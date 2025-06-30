from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from imagebind import data
from imagebind.models.imagebind_model import imagebind_huge

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = imagebind_huge(pretrained=True, path="/home/mdxuser/ImageBind/model_imagebind_force.pth")
    model = imagebind_huge(pretrained=True)
    model.eval().to(device)

    # 学習時と同じ logit‐scale を使う
    temperature = 0.2
    exp_temp = torch.exp(torch.tensor(temperature)).to(device)
    # label の読み込み
    labels = []
    with open(data_dir + "scenarios.txt", "r") as f:
        for line in f:
            labels.append(line.strip())
    labels_preprocessed = data.load_and_transform_text(labels, device).to(device)
    with torch.no_grad():
        labels_encoded = model.encode_text(labels_preprocessed)
        print("shape:", labels_encoded.shape)  # (N_labels, D)
    exit()
    eval_path = data_dir + "eval.csv"
    eval_df = pd.read_csv(eval_path)

    n_correct = 0
    total = 0

    # 各ラベルに対して最も高いスコアの力覚データを保持する辞書
    best_per_label = {
        label: {"score": float("-inf"), "csv_path": None, "start": None} for label in labels
    }
    # 各ラベルについて、各startのスコアを保持する辞書
    scores_per_label = {label: [] for label in labels}

    for _, row in eval_df.iterrows():
        total += 1
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
            force_emb = model.encode_force(force_tensor)  # → (1, D)
            fe = force_emb / force_emb.norm(dim=-1, keepdim=True)
            cos_sim = fe @ labels_encoded.T  # → (1, N_labels)
            logits = cos_sim  # → (1, N_labels)

            # 確率に変換
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # Tensor of shape (N_labels,)

        for idx, label in enumerate(labels):
            p = probs[idx].item()
            # 確率をリストに追加
            scores_per_label[label].append((start, p))
            if p > best_per_label[label]["score"]:
                best_per_label[label] = {
                    "score": p,
                    "csv_path": force_csv,
                    "start": start,
                }

    # 最も高いスコアの結果を出力
    for label, info in best_per_label.items():
        print(
            f"Label={label}  best_start={info['start']}  "
            f"csv={info['csv_path']}  probability={info['score']:.4f}"
        )

    # 各ラベルについて、全 start のスコア一覧を出力
    print("\n=== 全 start ごとのスコア一覧 ===")
    for label, entries in scores_per_label.items():
        print(f"\nLabel: {label}")
        for start, s in entries:
            print(f"  start={start}  score={s:.4f}")

    # ラベル同士の関連度（コサイン類似度行列）を計算
    labels_norm = labels_encoded / labels_encoded.norm(dim=1, keepdim=True)  # 正規化
    sim_matrix = labels_norm @ labels_norm.T  # (N_labels, N_labels)

    sim_df = pd.DataFrame(sim_matrix.cpu().numpy(), index=labels, columns=labels)
    print("=== Label-to-Label Cosine Similarity Matrix ===")
    print(sim_df)

    # 各ラベルに対して上位5つの関連ラベルを表示
    print("\n=== Top-5 Related Labels ===")
    for i, label in enumerate(labels):
        # 自己を除いて降順ソート
        sims = sim_matrix[i]
        topk = torch.topk(sims, k=6).indices.tolist()  # 6 取って先頭は自己
        related = [(labels[j], sims[j].item()) for j in topk if j != i][:5]
        print(f"{label}: {related}")

    print(f"\nAccuracy: {n_correct / total * 100:.2f}%")
    with open("data/predictions_new.txt", "a") as f:
        f.write(f"Accuracy: {n_correct / total * 100:.2f}%\n")

    # ─────────────── ファイル出力 ───────────────
    # CSV 形式で保存 (data/label_similarity.csv)
    out_path = "data/label_similarity.csv"
    sim_df.to_csv(out_path, index=True, encoding="utf-8-sig")
    print(f"Saved Label-to-Label Similarity Matrix → {out_path}")
    # ─────────────────────────────────────────


if __name__ == "__main__":
    main()
