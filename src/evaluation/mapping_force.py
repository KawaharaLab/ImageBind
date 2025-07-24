import os

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")  # GUI なしでも動かす
import matplotlib.pyplot as plt
import umap
from fire import Fire
import clip
from imagebind.models.force_model import load_model

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

def plotting(name, length="short", type="normal"):
    #========================== TODO: change to json ============================
    temperature: float = 0.2
    drop_path: float = 0.3
    num_blocks: int = 4
    out_embed_dim: int = 512
    data_len: int = 300
    mode = "pure"
    cnn = False
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
            pretrained=True, ckpt_path=f"/home/mdxuser/ImageBind/data/{mode}/{name}.pth",
            drop_path=drop_path, num_blocks=num_blocks, out_embed_dim=out_embed_dim, data_channels=data_channels, data_len=data_len, temperature=temperature
        ).to(device).float()


    force_encoder.eval().to(device)
    text_encoder, _ = clip.load("ViT-B/16", device=device)

    eval_df = pd.read_csv(data_dir + "eval.csv")

    # ───────────── UMAP 用に force 埋め込みを収集 ─────────────
    force_feats = []
    labels_feats = []  # ラベルの埋め込みを保存するリスト
    labels = []
    for _, row in eval_df.iterrows():
        force_csv = row["csv_path"]
        start = row["timestep_start"]
        force_df = pd.read_csv(force_csv)
        arr = force_df[use_cols].values.astype("float32")[start : start + data_len, :]
        # NaN 補間
        # for col in range(arr.shape[1]):
        #     y = arr[:, col]
        #     x = np.arange(len(y))
        #     mask = ~np.isnan(y)
        #     arr[:, col] = np.interp(x, x[mask], y[mask])
        force_tensor = torch.from_numpy(arr).T.unsqueeze(0).to(device)  # → (1, 15, T)

        # エンコード＆正規化
        with torch.no_grad():
            emb = force_encoder(force_tensor)  # → (1, D)
        force_feats.append(emb.cpu().numpy().reshape(-1))
        if length == "long":
            label_preprocessed = clip.tokenize([row["label"]]).to(device)
        else:
            label_preprocessed = clip.tokenize([row["label_short"]]).to(device)
        with torch.no_grad():
            label_emb = text_encoder.encode_text(label_preprocessed)
            print(f"label_emb shape: {label_emb.shape}")  # (1, D)
            labels_feats.append(label_emb.cpu().numpy().reshape(-1))
        if length == "long":
            labels.append(row["label"])  # 元のラベルを保存
        else:
            labels.append(row["label_short"])  # 短いラベルを保存
    force_feats = np.stack(force_feats)  # (N_samples, D)
    labels_feats = np.stack(labels_feats)  # (N_samples, D)
    # ───────────── UMAP 次元削減 ─────────────
    reducer = umap.UMAP(random_state=42)

    if type == "textbase":
        embedding_labels_exclusive = reducer.fit_transform(
            [labels_feats[0], labels_feats[1], labels_feats[2], labels_feats[4]]
        )  # → (N, 2)
        embedding_labels = reducer.transform(labels_feats)  # → (N, 2)
        embedding_force = reducer.transform(force_feats)  # → (N, 2)
    else:
        embedding_force = reducer.fit_transform(force_feats)  # → (N, 2)
        embedding_labels = reducer.transform(labels_feats)    # → (N, 2)
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
        s=5,
        alpha=0.8,
    )
    plt.scatter(
        embedding_labels[:, 0],
        embedding_labels[:, 1],
        c=label_idxs,
        cmap="tab10",
        marker="X",
        s=240,
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
        fontsize=14,
        title_fontsize=16,
    )
    plt.gca().add_artist(legend1)

    # ──────────── 凡例：Text Embeddings マーカー ────────────
    from matplotlib.lines import Line2D

    text_handle = Line2D(
        [0],
        [0],
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
        fontsize=14,
    )
    plt.gca().add_artist(legend2)

    # plt.title("UMAP of Force Embeddings", fontsize=16)
    # ──────────── 軸を非表示 ────────────
    plt.xticks([])  # x 軸目盛りを消す
    plt.yticks([])  # y 軸目盛りを消す
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    os.makedirs(f"data/{name}", exist_ok=True)
    if type == "textbase":
        out_path = os.path.join(f"data/{name}", f"force_umap_{length}_textbase.png")
    else:
        out_path = os.path.join(f"data/{name}", f"force_umap_{length}_normal.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved UMAP of force embeddings → {out_path}")
    plt.close()  # プロットを閉じる

def main(name):
    """
    Main function to run the plotting.
    :param name: Name of the model or dataset to use for plotting.
    """
    plotting(name, length="short", type="normal")
    plotting(name, length="long", type="normal")
    plotting(name, length="short", type="textbase")
    plotting(name, length="long", type="textbase")


if __name__ == "__main__":
    Fire(main)
