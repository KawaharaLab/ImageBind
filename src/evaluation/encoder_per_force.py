from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fire import Fire

import clip
from imagebind import data

data_dir = "/home/mdxuser/sim/Genesis/data/"

ALL_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "right_fx",
    "right_fy",
    "right_fz",
    "dof_0",
    "dof_1",
    "dof_2",
    "dof_3",
    "dof_4",
    "dof_5",
    "dof_6",
    "dof_7",
    "dof_8",
]
PURE_FORCE_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "right_fx",
    "right_fy",
    "right_fz",
]
COMPACT_FORCE_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "right_fx",
    "right_fy",
    "right_fz",
    "dof_7",
    "dof_8",
]

def main(model_path):
    #========================== TODO: change to json ============================
    temperature: float = 0.2
    drop_path: float = 0.3
    num_blocks: int = 4
    out_embed_dim: int = 512
    data_len: int = 300
    mode = "pure"
    cnn = False  # CNN モデルを使うかどうか
    #==========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "pure":
        data_channels = 6
        use_cols = PURE_FORCE_COLS
    elif mode == "compact":
        data_channels = 8
        use_cols = COMPACT_FORCE_COLS
    else:
        data_channels = 15
        use_cols = ALL_COLS    

    if cnn == True:
        from imagebind.models.force_model_cnn import load_model
        force_encoder = load_model(out_dim=out_embed_dim).to(device).float()
    else:
        from imagebind.models.force_model import load_model
        force_encoder = load_model(
            pretrained=True, ckpt_path=model_path,
            drop_path=drop_path, num_blocks=num_blocks, out_embed_dim=out_embed_dim, data_channels=data_channels, data_len=data_len, temperature=temperature
        ).to(device).float()


    force_encoder.eval()
    text_encoder, _ = clip.load("ViT-B/16", device=device)

    # label の読み込み
    labels = []
    with open(data_dir + "scenarios_simple.txt", "r") as f:
        for line in f:
            labels.append(line.strip())
    labels_preprocessed = clip.tokenize(labels).to(device)
    with torch.no_grad():
        labels_encoded = text_encoder.encode_text(labels_preprocessed).float()
        labels_encoded /= labels_encoded.norm(dim=-1, keepdim=True)  # 正規化

    eval_path = data_dir + "eval.csv"
    eval_df = pd.read_csv(eval_path)

    # Confusion matrix 用に初期化
    # 真のラベル → 予測ラベル → カウント
    confusion = {t: {p: 0 for p in labels} for t in labels}

    n_correct = 0
    total = 0
    for _, row in eval_df.iterrows():
        force_csv = row["csv_path"]
        start = row["timestep_start"]*data_len//3000
        force_df = pd.read_csv(force_csv).iloc[::3000//data_len].reset_index(drop=True)
        correct = "False"
        force_array = force_df[use_cols].values.astype("float32")[start : start + data_len, :]
        # for col in range(len(ALL_COLS)):
        #     y = force_array[:, col]
        #     x = np.arange(len(y))
        #     not_nan = ~np.isnan(y)
        #     y_interp = np.interp(x, x[not_nan], y[not_nan])
        #     force_array[:, col] = y_interp
        # モデルの期待形状に合わせて必要なら転置 (ここではチャネル×時系列長)
        force_tensor = torch.from_numpy(force_array).T  # → (15, T)
        force_tensor = force_tensor.unsqueeze(0).to(device)  # → (1, 15, T)

        with torch.no_grad():
            fe = force_encoder(force_tensor)  # → (1, D)
            # labels_encoded も正規化済み (N_labels, D)
            cos_sim = fe @ labels_encoded.T  # → (1, N_labels)
            # 温度スケールを反映
            logits = cos_sim

        # 各ラベルのスコアを取り出し
        scores = logits.squeeze(0)  # Tensor of shape (N_labels,)

        # (2) 最も高いスコアのラベルを予測
        pred_idx = scores.argmax().item()
        pred_label = labels[pred_idx]
        # confusion matrix 更新
        true_label = row["label"]
        confusion[true_label][pred_label] += 1
        if pred_label == true_label:
            n_correct += 1
            correct = "True"

        # logits: Tensor of shape (1, N_labels)
        probs = torch.softmax(logits, dim=-1)  # → (1, N_labels), 全て0～1, 合計1
        print(probs)
        # (3) ファイルにも全スコアを追記
        with open(data_dir + "predictions_new.txt", "a") as f:
            line = f"{row['csv_path']},{start},{correct},{row['label']},{pred_label}"
            for p in probs[0]:
                line += f",{p.item():.4f}"
            f.write(line + "\n")

        # 各ラベルの確率を出力
        # for idx, label in enumerate(labels):
        #     print(f"  {label}: {probs[0, idx].item():.4f}")

        total += 1
        print(f"Predict={pred_label} True={row['label']} Total={total}, Correct={n_correct}")

    # ─── ループ終了後にマークダウン形式でサマリー出力 ───
    md_lines = []
    # ヘッダ行
    md_lines.append("| True \\ Predicted | " + " | ".join([f"{p}" for p in labels]) + " |")
    # 区切り行
    sep = "|:-------------------|" + "|".join([":----------------:" for _ in labels]) + "|"
    md_lines.append(sep)
    # 各行データ
    for t in labels:
        counts = [f"{confusion[t][p]:>3}" for p in labels]
        md_lines.append(f"| {t:<18} | " + " | ".join(counts) + " |")

    # ファイルに書き出し
    out_path = "data/prediction_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved prediction summary → {out_path}")

    print(f"Accuracy: {n_correct / total * 100:.2f}%")
    with open(data_dir + "predictions_new.txt", "a") as f:
        f.write(f"Accuracy: {n_correct / total * 100:.2f}%\n")


if __name__ == "__main__":
    Fire(main)
