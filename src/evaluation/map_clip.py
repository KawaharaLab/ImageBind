import os

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")  # Run without GUI
import matplotlib.pyplot as plt
import umap
from fire import Fire
import clip
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
    mode = "pure"
    cnn = False
    #==========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "pure":
        data_channels = 12
        use_cols = PURE_FORCE_COLS
    elif mode == "compact":
        data_channels = 14
        use_cols = COMPACT_FORCE_COLS

    if cnn == True:
        from imagebind.models.force_model_cnn import load_model
        force_encoder = load_model(out_dim=out_embed_dim).to(device).float()
    else:
        from imagebind.models.force_model import load_model
        force_encoder = load_model(
            pretrained=True, ckpt_path=f"{BASE_DIR}data/models/clip/{mode}/{name}.pth",
            drop_path=drop_path, num_blocks=num_blocks, out_embed_dim=out_embed_dim, data_channels=data_channels, data_len=data_len, temperature=temperature
        ).to(device).float()


    force_encoder.eval().to(device)
    text_encoder, _ = clip.load("ViT-L/14@336px", device=device)

    eval_df = pd.read_csv(data_dir + "eval.csv")

    # Collect force embeddings for UMAP
    force_feats = []
    labels_feats = []  # Store text label embeddings
    labels = []
    skipped = 0
    for _, row in eval_df.iterrows():
        force_csv = row["csv_path"]
        start = row["start"]
        force_df = pd.read_csv(data_dir + "csv/" + force_csv)
        arr = force_df[use_cols].values.astype("float32")[start : start + data_len, :]
        # Skip segments shorter than data_len to avoid positional embedding mismatch
        if arr.shape[0] != data_len:
            skipped += 1
            continue
        force_tensor = torch.from_numpy(arr).T.unsqueeze(0).to(device)  # (1, C, T)

        # Encode & normalize
        with torch.no_grad():
            emb = force_encoder(force_tensor)  # (1, D)
        force_feats.append(emb.cpu().numpy().reshape(-1))
        if length == "long":
            label_preprocessed = clip.tokenize([row["label"]]).to(device)
        else:
            label_preprocessed = clip.tokenize([row["label_short"]]).to(device)
        with torch.no_grad():
            label_emb = text_encoder.encode_text(label_preprocessed)
            labels_feats.append(label_emb.cpu().numpy().reshape(-1))
        if length == "long":
            labels.append(row["label"])
        else:
            labels.append(row["label_short"])

    if len(force_feats) == 0:
        raise RuntimeError(f"All segments were skipped (skipped={skipped}). Check eval.csv and segment lengths.")
    if skipped > 0:
        print(f"Warning: Skipped {skipped} segments due to insufficient length (< {data_len}).")

    force_feats = np.stack(force_feats)  # (N, D)
    labels_feats = np.stack(labels_feats)  # (N, D)
    # UMAP dimensionality reduction
    reducer = umap.UMAP(random_state=42)

    if type == "textbase":
        unique_labels_feats = np.unique(labels_feats, axis=0)
        embedding_labels_exclusive = reducer.fit_transform(
            unique_labels_feats
        )  # (N, 2)
        embedding_labels = reducer.transform(labels_feats)  # (N, 2)
        embedding_force = reducer.transform(force_feats)  # (N, 2)
    else:
        embedding_force = reducer.fit_transform(force_feats)  # (N, 2)
        embedding_labels = reducer.transform(labels_feats)    # (N, 2)
    # Map each sample label to an integer index
    unique_labels = sorted(set(labels))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    label_idxs = [label_to_idx[lbl] for lbl in labels]

    # Scatter plot
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

    import matplotlib.patches as mpatches

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
    os.makedirs(f"data/results/clip/{mode}/{name}", exist_ok=True)
    if type == "textbase":
        out_path = os.path.join(f"data/results/clip/{mode}/{name}", f"force_umap_{length}_textbase.png")
    else:
        out_path = os.path.join(f"data/results/clip/{mode}/{name}", f"force_umap_{length}_normal.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved UMAP of force embeddings → {out_path}")
    plt.close()

def main(name):
    """
    Main function to run the plotting.
    :param name: Name of the model or dataset to use for plotting.
    """
    # plotting(name, length="short", type="normal")
    plotting(name, length="long", type="normal")
    # plotting(name, length="short", type="textbase")
    plotting(name, length="long", type="textbase")


if __name__ == "__main__":
    Fire(main)
