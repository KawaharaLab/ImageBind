import os
import matplotlib
import numpy as np
import pandas as pd
import torch
import umap
from fire import Fire
import clip
# Assuming these custom models are in the path
from imagebind.models.force_model import load_model as load_force_transformer
from imagebind.models.force_model_cnn import load_model as load_force_cnn

import re
from pathlib import Path

# --- Constants and Configuration ---
matplotlib.use("Agg")  # GUI なしでも動かす
import matplotlib.pyplot as plt

data_dir = "/home/mdxuser/ImageBind/data/ESEP_data"
MODEL_PATH = '/home/mdxuser/ImageBind/data/ESEP_data/vital-cloud-2_epoch_100.pth'
TEST_PATH = '/home/mdxuser/Genesis/main/data/picked_up/simple_formatted_training_data_IB_2_test.csv'


ALL_COLS = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz", "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"]
PURE_FORCE_COLS = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz"]
COMPACT_FORCE_COLS = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz", "dof_7", "dof_8"]

# =================================================================================
#  REFACTORED DATA HANDLING FUNCTIONS
# =================================================================================

def load_and_prepare_force_data(csv_path: str, start_index: int, data_len: int, use_cols: list, device: torch.device) -> torch.Tensor:
    """
    Loads a segment of force data from a CSV and prepares it for the model.

    Args:
        csv_path (str): Path to the CSV file with raw force data.
        start_index (int): The starting timestep for the data window.
        data_len (int): The length of the data window (e.g., 100 timesteps).
        use_cols (list): A list of column names to extract from the CSV.
        device (torch.device): The device to move the tensor to ('cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor of shape (1, num_channels, data_len) ready for the model.
    """
    force_df = pd.read_csv(csv_path)
    # Extract the specified window and columns
    arr = force_df[use_cols].values.astype("float32")[start_index : start_index + data_len, :]

    # NOTE: NaN interpolation logic would go here if needed.
    # for col in range(arr.shape[1]):
    #     y = arr[:, col]
    #     # ... interpolation code ...
    #     arr[:, col] = np.interp(...)

    # Transpose to (channels, timesteps) and add a batch dimension -> (1, C, T)
    force_tensor = torch.from_numpy(arr).T.unsqueeze(0).to(device)
    return force_tensor

def generate_embeddings(
    eval_df: pd.DataFrame, force_encoder: torch.nn.Module, text_encoder: torch.nn.Module,
    text_label_column: str, use_cols: list, data_len: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Generates force and text embeddings from the evaluation data.

    This function iterates through the evaluation dataframe, processes each data point,
    and returns the collected embeddings and labels.

    Args:
        eval_df (pd.DataFrame): DataFrame with metadata (paths, labels).
        force_encoder (torch.nn.Module): The trained model to encode force data.
        text_encoder (torch.nn.Module): The trained model to encode text (e.g., CLIP).
        text_label_column (str): The name of the column containing text labels ("annotation" or "label").
        use_cols (list): List of force data columns to use.
        data_len (int): The length of the force data sequence.
        device (torch.device): The device for computation.

    Returns:
        tuple[np.ndarray, np.ndarray, list]: A tuple containing:
            - force_embeddings (np.ndarray): Array of force embeddings, shape (N, D).
            - text_embeddings (np.ndarray): Array of text embeddings, shape (N, D).
            - raw_labels (list): List of the original string labels.
    """
    force_embeddings = []
    text_embeddings = []
    raw_labels = []

    with torch.no_grad(): # Disable gradient calculation for efficiency
        for _, row in eval_df.iterrows():
            # --- 1. Process Force Data ---
            # INPUT: File path, start time, etc.
            force_tensor_input = load_and_prepare_force_data(
                csv_path=row["csv_path"],
                start_index=row["timestep_start"],
                data_len=data_len,
                use_cols=use_cols,
                device=device
            )
            # OUTPUT: A single force embedding vector from the model
            force_emb_output = force_encoder(force_tensor_input) # Shape: (1, D)
            force_embeddings.append(force_emb_output.cpu().numpy().flatten())

            # --- 2. Process Text Data ---
            # INPUT: A string label from the dataframe
            label_text_input = row[text_label_column]
            tokenized_text = clip.tokenize([label_text_input]).to(device)

            # OUTPUT: A single text embedding vector from the model
            text_emb_output = text_encoder.encode_text(tokenized_text) # Shape: (1, D)
            text_embeddings.append(text_emb_output.cpu().numpy().flatten())

            # --- 3. Collect Raw Label for Plotting ---
            raw_labels.append(label_text_input)

    # Convert lists of embeddings into 2D NumPy arrays
    return np.stack(force_embeddings), np.stack(text_embeddings), raw_labels


def plotting(name, length="short", type="normal", mark = True):
    # ========================== TODO: change to json ============================
    temperature: float = 0.2
    drop_path: float = 0.3
    num_blocks: int = 6
    out_embed_dim: int = 512
    data_len: int = 100
    mode = "pure"
    cnn = False
    # ==========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "pure":
        use_cols, data_channels = PURE_FORCE_COLS, 6
    elif mode == "compact":
        use_cols, data_channels = COMPACT_FORCE_COLS, 8
    else:
        use_cols, data_channels = ALL_COLS, 15

    # --- Model Loading ---
    if cnn:
        force_encoder = load_force_cnn(out_dim=out_embed_dim).to(device).float()
    else:
        force_encoder = load_force_transformer(
            pretrained=True, ckpt_path=MODEL_PATH,
            drop_path=drop_path, num_blocks=num_blocks, out_embed_dim=out_embed_dim, data_channels=data_channels, data_len=data_len, temperature=temperature
        ).to(device).float()
    force_encoder.eval()
    text_encoder, _ = clip.load("ViT-B/16", device=device)
    text_encoder.eval()

    # --- Data Loading ---
    eval_df = pd.read_csv(TEST_PATH)

    # --- Generate Embeddings (The new, clean data pipeline) ---
    text_label_column = "label" if length == "long" else "annotation"
    force_feats, labels_feats, labels = generate_embeddings(
        eval_df=eval_df,
        force_encoder=force_encoder,
        text_encoder=text_encoder,
        text_label_column=text_label_column,
        use_cols=use_cols,
        data_len=data_len,
        device=device
    )

    # --- UMAP Dimensionality Reduction ---
    reducer = umap.UMAP(random_state=42)
    if type == "textbase":
        # Fit UMAP on a subset of text embeddings to create a "language-based" map
        reducer.fit(labels_feats)
 
        embedding_labels = reducer.transform(labels_feats)
        embedding_force = reducer.transform(force_feats)
    else:
        # Fit UMAP on force embeddings to create a "force-based" map
        
        embedding_force = reducer.fit_transform(force_feats)
 
        embedding_labels = reducer.transform(labels_feats)

    # --- Plotting ---
    unique_labels = sorted(list(set(labels)))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    label_idxs = [label_to_idx[lbl] for lbl in labels]

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding_force[:, 0], embedding_force[:, 1], c=label_idxs, cmap="tab10", s=15, alpha=0.8, label="Force Embeddings")
    if mark:
        plt.scatter(embedding_labels[:, 0], embedding_labels[:, 1], c=label_idxs, cmap="tab10", marker="X", s=250, linewidths=1.5, edgecolors="k", alpha=1.0)
    
    # ... (The rest of the plotting code for legends and saving is unchanged) ...
    import matplotlib.patches as mpatches
    color_handles = [mpatches.Patch(color=scatter.cmap(scatter.norm(idx)), label=lbl) for idx, lbl in enumerate(unique_labels)]
    legend1 = plt.legend(handles=color_handles, title="Action Labels", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=14, title_fontsize=16)
    plt.gca().add_artist(legend1)


    from matplotlib.lines import Line2D
    text_handle = Line2D([0], [0], marker="X", color="w", markerfacecolor="gray", markersize=10, label="Text Embeddings", linewidth=0)
    legend2 = plt.legend(handles=[text_handle], bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0.0, fontsize=14)
    plt.gca().add_artist(legend2)
    
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


    if name is None:
        # Extract the wandb path
        name = MODEL_PATH.split('/')[-1].split('.')[0]

    os.makedirs(os.path.join(data_dir, name), exist_ok=True)
    if mark:
        out_path = os.path.join(f"{data_dir}/{name}", f"force_umap_{length}_{type}.png")
    else:
        out_path = os.path.join(f"{data_dir}/{name}", f"force_umap_{length}_{type}_unmarked.png")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved UMAP plot -> {out_path}")
    plt.close()

def main(option=0, name=None):
    # OPTION 0: Does all graphs
    # OPTION 1: Does only marked graphs
    # OPTION 2: Does only unmarked graphs
    if option == 0:
        plotting(name, length="short", type="normal", mark = True)
        plotting(name, length="short", type="textbase", mark = True)
        plotting(name, length="short", type="normal", mark = False)
        plotting(name, length="short", type="textbase", mark = False)
        
    elif option == 1:
        plotting(name, length="short", type="normal", mark = True)
        plotting(name, length="short", type="textbase", mark = True)
    else:
        plotting(name, length="short", type="normal", mark = False)
        plotting(name, length="short", type="textbase", mark = False)
        

   

if __name__ == "__main__":
    # Name is optional to select where to save to
    # Otherwise, it will save to the model name
    Fire(main)

    # Ex: uv run 