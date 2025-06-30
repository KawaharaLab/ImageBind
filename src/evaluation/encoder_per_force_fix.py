from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fire import Fire

import clip
from imagebind import data
from imagebind.models.force_model import load_force_encoder

data_dir = "/home/mdxuser/sim/Genesis/data/"

USE_FORCE_COLS = [
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


def main(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_encoder = load_force_encoder(
        pretrained=True, ckpt_path=model_path
    )
    force_encoder.eval().to(device)
    text_encoder, _ = clip.load(
        "ViT-L/14@336px", device=device
    )

    # 学習時と同じ logit‐scale を使う
    temperature = 0.2
    exp_temp = torch.exp(torch.tensor(temperature)).to(device)
    # label の読み込み
    labels = []
    with open(data_dir + "scenarios_simple.txt", "r") as f:
        for line in f:
            labels.append(line.strip())
    labels_preprocessed = clip.tokenize(labels).to(device)
    with torch.no_grad():
        labels_encoded = text_encoder.encode_text(labels_preprocessed).float()

    eval_path = data_dir + "eval.csv"
    eval_df = pd.read_csv(eval_path)

    n_correct = 0
    total = 0
    for _, row in eval_df.iterrows():
        force_csv = row["csv_path"]
        start = row["timestep_start"]
        force_df = pd.read_csv(force_csv)
        correct = "False"
        force_array = force_df[USE_FORCE_COLS].values.astype("float32")[
            start : start + 3000, :
        ]  # 3000 samples
        for col in range(len(USE_FORCE_COLS)):
            y = force_array[:, col]
            x = np.arange(len(y))
            not_nan = ~np.isnan(y)
            y_interp = np.interp(x, x[not_nan], y[not_nan])
            force_array[:, col] = y_interp
        # モデルの期待形状に合わせて必要なら転置 (ここではチャネル×時系列長)
        force_tensor = torch.from_numpy(force_array).T  # → (15, T)
        force_tensor = force_tensor.unsqueeze(0).to(device)  # → (1, 15, T)

        with torch.no_grad():
            force_emb = force_encoder(force_tensor)  # → (1, D)
            # normalize して cosine similarity
            fe = force_emb / force_emb.norm(dim=-1, keepdim=True)
            # labels_encoded も正規化済み (N_labels, D)
            cos_sim = fe @ labels_encoded.T  # → (1, N_labels)
            # 温度スケールを反映
            logits = cos_sim * exp_temp  # → (1, N_labels)

        # 各ラベルのスコアを取り出し
        scores = logits.squeeze(0)  # Tensor of shape (N_labels,)

        # (2) 最も高いスコアのラベルを予測
        pred_idx = scores.argmax().item()
        pred_label = labels[pred_idx]
        if pred_label == row["label"]:
            n_correct += 1
            correct = "True"

        # logits: Tensor of shape (1, N_labels)
        probs = torch.softmax(logits, dim=-1)  # → (1, N_labels), 全て0～1, 合計1

        # (3) ファイルにも全スコアを追記
        with open(data_dir + "predictions_new.txt", "a") as f:
            line = f"{row['csv_path']},{start},{correct},{row['label']},{pred_label}"
            for p in probs[0]:
                line += f",{p.item():.4f}"
            f.write(line + "\n")

        # 各ラベルの確率を出力
        for idx, label in enumerate(labels):
            print(f"  {label}: {probs[0, idx].item():.4f}")

        total += 1
        print(f"Predict={pred_label}  Total={total}, Correct={n_correct}")

    print(f"Accuracy: {n_correct / total * 100:.2f}%")
    with open(data_dir + "predictions_new.txt", "a") as f:
        f.write(f"Accuracy: {n_correct / total * 100:.2f}%\n")


if __name__ == "__main__":
    Fire(main)
