import os

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")           # GUI なしでも動かす
import matplotlib.pyplot as plt
import umap

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
    model = imagebind_huge(pretrained=True, path="/home/mdxuser/ImageBind/data/model_imagebind_force.pth")
    model.eval().to(device)

    # # label の読み込み (必要なければ削除可)
    # labels = []
    # with open(data_dir + "scenarios.txt", "r") as f:
    #     for line in f:
    #         labels.append(line.strip())
    # labels_preprocessed = data.load_and_transform_text(labels, device).to(device)
    # with torch.no_grad():
    #     labels_encoded = model.encode_text(labels_preprocessed)

    eval_df = pd.read_csv(data_dir + "eval.csv")

    # ───────────── UMAP 用に force 埋め込みを収集 ─────────────
    force_feats = []
    labels_feats = []  # ラベルの埋め込みを保存するリスト
    labels = []
    for _, row in eval_df.iterrows():
        force_csv = row["csv_path"]
        start = row["timestep_start"]
        force_df = pd.read_csv(force_csv)
        arr = force_df[USE_FORCE_COLS].values.astype("float32")[start : start + 3000, :]
        # NaN 補間
        for col in range(arr.shape[1]):
            y = arr[:, col]
            x = np.arange(len(y))
            mask = ~np.isnan(y)
            arr[:, col] = np.interp(x, x[mask], y[mask])
        force_tensor = torch.from_numpy(arr).T.unsqueeze(0).to(device)  # → (1, 15, T)

        # エンコード＆正規化
        with torch.no_grad():
            emb = model.encode_force(force_tensor)  # → (1, D)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        force_feats.append(emb.cpu().numpy().reshape(-1))
        label_preprocessed = data.load_and_transform_text([row["label"]], device).to(device)
        with torch.no_grad():
            label_emb = model.encode_text(label_preprocessed)
            print(f"label_emb shape: {label_emb.shape}")  # (1, D)
            labels_feats.append(label_emb.cpu().numpy().reshape(-1))
        labels.append(row["label"])
    force_feats = np.stack(force_feats)  # (N_samples, D)
    labels_feats = np.stack(labels_feats)  # (N_samples, D)
    # ───────────── UMAP 次元削減 ─────────────
    reducer = umap.UMAP(random_state=42)
    embedding_force = reducer.fit_transform(force_feats)  # → (N, 2)
    embedding_labels = reducer.transform(labels_feats)    # → (N, 2)
    # embedding_labels_exclusive = reducer.fit_transform([labels_feats[0], labels_feats[1], labels_feats[2], labels_feats[4]])  # → (N, 2)
    # embedding_labels = reducer.transform(labels_feats)    # → (N, 2)
    # embedding_force = reducer.transform(force_feats)  # → (N, 2)
    # ──────────── ここを追加 ────────────
    # 各サンプルのラベルに対応する整数インデックスを作成
    unique_labels = sorted(set(labels))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    label_idxs = [label_to_idx[lbl] for lbl in labels]
    # ───────────────────────────────────

    # 散布図の描画
    scatter = plt.scatter(
        embedding_force[:, 0],
        embedding_force[:, 1],
        c=label_idxs,
        cmap="tab10",
        s=10,
        alpha=0.8,
    )
    plt.scatter(
        embedding_labels[:, 0],
        embedding_labels[:, 1],
        c=label_idxs,
        cmap="tab10",
        marker="X",
        s=100,
        linewidths=1,
        edgecolors="k",
        alpha=1.0,
    )

    # ──────────── 凡例：ラベルごとの色 ────────────
    import matplotlib.patches as mpatches
    # unique_labels はすでに定義済み
    color_handles = [
        mpatches.Patch(color=scatter.cmap(scatter.norm(idx)), label=lbl)
        for idx, lbl in enumerate(unique_labels)
    ]
    legend1 = plt.legend(
        handles=color_handles,
        title="Labels",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    plt.gca().add_artist(legend1)

    # ──────────── 凡例：Text Embeddings マーカー ────────────
    from matplotlib.lines import Line2D
    text_handle = Line2D(
        [0], [0],
        marker="X",
        color="w",
        markerfacecolor="k",
        markersize=8,
        label="Text Embeddings",
        linewidth=0,
    )
    # 凡例を枠外（プロット右側下部）に配置
    legend2 = plt.legend(
        handles=[text_handle],
        bbox_to_anchor=(1.05, 0.1),
        loc="lower left",
        borderaxespad=0.0,
    )
    plt.gca().add_artist(legend2)

    plt.title("UMAP of Force vs Text Embeddings", fontsize=16)
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "force_umap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved UMAP of force embeddings → {out_path}")


if __name__ == "__main__":
    main()
