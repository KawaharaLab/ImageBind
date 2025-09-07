import os

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.use("Agg")  # Run without GUI
import matplotlib.pyplot as plt
import umap
from fire import Fire
from imagebind.models.force_model import load_model

BASE_DIR = "/home/user/ImageBind/"

data_dir = "/home/user/Genesis/data/eval/"

PURE_FORCE_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "left_tx",
    "left_ty",
    "left_tz",
    "right_fx",
    "right_fy",
    "right_fz",
    "right_tx",
    "right_ty",
    "right_tz",
]

COMPACT_FORCE_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "left_tx",
    "left_ty",
    "left_tz",
    "right_fx",
    "right_fy",
    "right_fz",
    "right_tx",
    "right_ty",
    "right_tz",
    "dof_7",
    "dof_8",
]

def plotting(name, length="short", type="normal"):
    #========================== TODO: change to json ============================
    temperature: float = 0.2
    drop_path: float = 0.3
    num_blocks: int = 6
    out_embed_dim: int = 768
    data_len: int = 80
    mode = "pure"  # "pure" / "compact"
    cnn = False
    #====================================================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "pure":
        data_channels = 12
        use_cols = PURE_FORCE_COLS
    elif mode == "compact":
        data_channels = 14
        use_cols = COMPACT_FORCE_COLS

    # Force encoder 読み込み（SigLIP 学習と同じ保存場所想定）
    # 例: data/models/pure/<run_name>.pth
    ckpt_path = f"{BASE_DIR}data/models/siglip/{mode}/{name}.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Force encoder checkpoint not found: {ckpt_path}")

    if cnn:
        from imagebind.models.force_model_cnn import load_model as load_cnn
        force_encoder = load_cnn(out_dim=out_embed_dim).to(device).float()
    else:
        force_encoder = load_model(
            pretrained=True,
            ckpt_path=ckpt_path,
            drop_path=drop_path,
            num_blocks=num_blocks,
            out_embed_dim=out_embed_dim,
            data_channels=data_channels,
            data_len=data_len,
            temperature=temperature,
        ).to(device).float()

    force_encoder.eval()

    eval_csv = os.path.join(data_dir, "eval_thin_10pct.csv")
    if not os.path.exists(eval_csv):
        raise FileNotFoundError(f"eval.csv not found: {eval_csv}")
    eval_df = pd.read_csv(eval_csv)

    required_cols = {"csv_path", "start", "emb_index"}
    missing = required_cols - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval.csv is missing required columns: {missing}")

    # UMAP 用リスト
    force_feats = []
    labels_feats = []  # Store text label embeddings
    labels = []
    skipped_short = 0
    skipped_missing = 0

    text_emb_dir = os.path.join(data_dir, "text_emb")
    if not os.path.isdir(text_emb_dir):
        raise FileNotFoundError(f"Text embedding directory not found: {text_emb_dir}")

    for _, row in eval_df.iterrows():
        csv_rel = row["csv_path"]
        start = int(row["start"])
        emb_index = row["emb_index"]
        force_csv_path = os.path.join(data_dir, "csv", csv_rel)
        if not os.path.exists(force_csv_path):
            skipped_missing += 1
            continue
        df_force = pd.read_csv(force_csv_path)
        if any(col not in df_force.columns for col in use_cols):
            skipped_missing += 1
            continue
        arr = df_force[use_cols].values.astype("float32")[start : start + data_len, :]
        if arr.shape[0] != data_len:
            skipped_short += 1
            continue

        # Force embedding
        force_tensor = torch.from_numpy(arr).T.unsqueeze(0).to(device)
        with torch.no_grad():
            f_emb = force_encoder(force_tensor)  # (1, D)
        force_feats.append(f_emb.cpu().numpy().reshape(-1))

        # Text embedding (precomputed SigLIP) の読み込み
        emb_path = os.path.join(text_emb_dir, f"{emb_index}.pt")
        if not os.path.exists(emb_path):
            skipped_missing += 1
            force_feats.pop()  # 対応埋め込みがないので取り消し
            continue
        t_emb = torch.load(emb_path, map_location="cpu")
        if t_emb.ndim == 2 and t_emb.shape[0] == 1:
            t_emb = t_emb.squeeze(0)
        labels_feats.append(t_emb.numpy())

        # ラベル（表示用）
        if length == "long" and "label" in row:
            labels.append(row.get("label", str(emb_index)))
        else:
            # 短いラベルがなければ emb_index をフォールバック
            labels.append(row.get("label_short", row.get("label", str(emb_index))))

    if len(force_feats) == 0:
        raise RuntimeError(
            f"No valid segments. skipped_short={skipped_short}, skipped_missing={skipped_missing}"
        )

    if skipped_short > 0 or skipped_missing > 0:
        print(
            f"Warning: skipped_short={skipped_short}, skipped_missing={skipped_missing}, kept={len(force_feats)}"
        )

    force_feats = np.stack(force_feats)  # (N, D)
    labels_feats = np.stack(labels_feats)  # (N, D)
    # UMAP dimensionality reduction
    reducer = umap.UMAP(random_state=42)

    if type == "textbase":
        unique_labels_feats = np.unique(labels_feats, axis=0)
        # embedding_labels_exclusive = reducer.fit_transform(unique_labels_feats)  # (N, 2)
        # embedding_labels = reducer.transform(labels_feats)  # (N, 2)
        embedding_labels = reducer.fit_transform(labels_feats)  # (N, 2)
        embedding_labels_exclusive = reducer.transform(unique_labels_feats)  # (N, 2)
        embedding_force = reducer.transform(force_feats)  # (N, 2)
    else:
        embedding_force = reducer.fit_transform(force_feats)  # (N, 2)
        embedding_labels = reducer.transform(labels_feats)    # (N, 2)
    # Map each sample label to an integer index
    unique_labels = sorted(set(labels))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    label_idxs = [label_to_idx[lbl] for lbl in labels]
    unique_labels_idxs = list(range(len(unique_labels)))
    # Plot
    scatter = plt.scatter(
        embedding_force[:, 0],
        embedding_force[:, 1],
        c=label_idxs,
        cmap="tab10",
        s=5,
        alpha=0.8,
    )
    plt.scatter(
        embedding_labels_exclusive[:, 0],
        embedding_labels_exclusive[:, 1],
        c=unique_labels_idxs,
        cmap="tab10",
        marker="X",
        s=240,
        linewidths=1,
        edgecolors="k",
        alpha=1.0,
    )

    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    color_handles = [
        mpatches.Patch(color=scatter.cmap(scatter.norm(idx)), label=lbl)
        for idx, lbl in enumerate(unique_labels)
    ]
    legend1 = plt.legend(
        handles=color_handles,
        title="Force embeddings",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=14,
        title_fontsize=16,
    )
    plt.gca().add_artist(legend1)

    text_handle = Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        markerfacecolor="k",
        markersize=8,
        label="Text embeddings",
        linewidth=0,
    )
    legend2 = plt.legend(
        handles=[text_handle],
        bbox_to_anchor=(1.05, 0.1),
        loc="lower left",
        borderaxespad=0.0,
        fontsize=14,
    )
    plt.gca().add_artist(legend2)

    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    os.makedirs(f"data/results/siglip/{mode}/{name}", exist_ok=True)
    if type == "textbase":
        out_path = os.path.join(f"data/results/siglip/{mode}/{name}", f"force_umap_{length}_textbase.png")
    else:
        out_path = os.path.join(f"data/results/siglip/{mode}/{name}", f"force_umap_{length}_normal.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved UMAP → {out_path}")
    plt.close()

def main(name):
    """
    Main function to run the plotting.
    :param name: Name of the model or dataset to use for plotting.
    """
    # plotting(name, length="short", type="normal")
    # plotting(name, length="long", type="normal")
    # plotting(name, length="short", type="textbase")
    plotting(name, length="long", type="textbase")


if __name__ == "__main__":
    Fire(main)
