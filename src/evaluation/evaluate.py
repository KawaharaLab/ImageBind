import os
import torch
import clip
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

# Note: Your custom model loading function and Dataset class are needed.
# I've included them here for a complete, runnable script.

# === MODEL DEFINITION (Copied from your training script) ===
# This needs to match the architecture of your trained model.
# For simplicity, I am assuming `load_model` is available.
# If not, you'd paste your model's class definition here.
from imagebind.models.force_model import load_model

# === DATASET CLASS (Copied from your training script) ===
class ForceDataset(Dataset):
    def __init__(self, csv_path, data_len: int = 100, use_cols: list = None):
        if use_cols is None:
            use_cols = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz"]
        
        train_df = pd.read_csv(csv_path)
        self.data_len = data_len
        self.use_cols = use_cols

        unique_csvs = train_df["csv_path"].unique()
        data_cache = {
            path: pd.read_csv(path, usecols=self.use_cols).values.astype("float32")
            for path in tqdm(unique_csvs, desc="Caching data files")
        }

        self.force_segments = []
        self.annotations = []
        for _, row in train_df.iterrows():
            seg = data_cache[row["csv_path"]][row["timestep_start"] : row["timestep_start"] + data_len]
            if seg.shape[0] != data_len:
                continue
            self.force_segments.append(seg)
            self.annotations.append(row["annotation"])

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, idx):
        force_tensor = torch.from_numpy(self.force_segments[idx]).T
        annotation_text = self.annotations[idx]
        tokenized_annotation = clip.tokenize([annotation_text])[0]
        return force_tensor, tokenized_annotation, annotation_text

# === GRAPHING FUNCTIONS ===
def generate_similarity_heatmap(logits, output_path):
    """Saves a heatmap of the similarity matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(logits, annot=False, cmap='viridis')
    plt.title("Force-to-Text Similarity Matrix")
    plt.xlabel("Text Samples")
    plt.ylabel("Force Samples")
    plt.savefig(output_path)
    plt.close()
    print(f"Similarity heatmap saved to {output_path}")

def generate_tsne_plot(force_embeds, text_embeds, annotations, output_path, sample_size=500):
    """Saves a t-SNE plot of the force and text embeddings."""
    if len(force_embeds) > sample_size:
        # Subsample for faster t-SNE processing
        indices = np.random.choice(len(force_embeds), sample_size, replace=False)
        force_embeds = force_embeds[indices]
        text_embeds = text_embeds[indices]
        annotations = [annotations[i] for i in indices]

    all_embeds = np.vstack((force_embeds, text_embeds))
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=min(30, len(force_embeds) - 1))
    tsne_results = tsne.fit_transform(all_embeds)
    
    num_samples = len(force_embeds)
    force_points = tsne_results[:num_samples]
    text_points = tsne_results[num_samples:]
    
    plt.figure(figsize=(16, 10))
    # Plot force and text points
    plt.scatter(force_points[:, 0], force_points[:, 1], c='blue', alpha=0.6, label='Force Embeddings')
    plt.scatter(text_points[:, 0], text_points[:, 1], c='red', alpha=0.6, label='Text Embeddings')
    
    # Draw lines connecting corresponding pairs
    for i in range(num_samples):
        plt.plot([force_points[i, 0], text_points[i, 0]], 
                 [force_points[i, 1], text_points[i, 1]], 
                 'k-', alpha=0.1)
    
    plt.title('t-SNE Projection of Force and Text Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"t-SNE plot saved to {output_path}")

# === MAIN EVALUATION FUNCTION ===
def evaluate(args):
    """Main function to run the evaluation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load Data ---
    # Use a fixed seed for reproducibility of the train/val split
    torch.manual_seed(42)
    full_dataset = ForceDataset(csv_path=args.train_csv)
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 2. Load Models ---
    # Make sure these parameters match the configuration of your 'happy-water-2' run
    force_encoder = load_model(
        drop_path=0.2,
        num_blocks=6,
        out_embed_dim=512,
        data_channels=6,
        data_len=100
    ).to(device).float() # Use same params as training

    force_encoder.load_state_dict(torch.load(args.model_path, map_location=device))
    force_encoder.eval()

    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    
    criterion = torch.nn.CrossEntropyLoss()

    # --- 3. Run Evaluation Loop ---
    val_loss, top1_correct, top5_correct = 0, 0, 0
    all_force_embeds, all_text_embeds, all_annotations = [], [], []
    first_batch_logits = None

    with torch.no_grad():
        for i, (forces, texts_tok, texts_raw) in enumerate(tqdm(val_loader, desc="Evaluating")):
            forces, texts_tok = forces.to(device), texts_tok.to(device)
            
            force_emb = force_encoder(forces)
            text_emb = clip_model.encode_text(texts_tok).float()
            
            force_emb /= force_emb.norm(dim=-1, keepdim=True)
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
            
            logits = (force_emb @ text_emb.T) / 0.07
            labels = torch.arange(len(forces), device=device)
            loss = (criterion(logits, labels) + criterion(logits.T, labels)) / 2.0
            val_loss += loss.item()
            
            # --- Accuracy Calculation ---
            top5_preds = torch.topk(logits, k=5, dim=1).indices
            top1_preds = top5_preds[:, 0]
            top1_correct += (top1_preds == labels).float().sum().item()
            top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=1).float().sum().item()

            # --- Store data for graphs ---
            all_force_embeds.append(force_emb.cpu().numpy())
            all_text_embeds.append(text_emb.cpu().numpy())
            all_annotations.extend(texts_raw)
            if i == 0:
                first_batch_logits = logits.cpu().numpy()

    # --- 4. Calculate and Print Metrics ---
    num_val_samples = len(val_dataset)
    avg_loss = val_loss / len(val_loader)
    top1_acc = top1_correct / num_val_samples
    top5_acc = top5_correct / num_val_samples

    print("\n--- Final Evaluation Metrics ---")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy:  {top1_acc:.4f} ({int(top1_correct)}/{num_val_samples})")
    print(f"Top-5 Accuracy:  {top5_acc:.4f} ({int(top5_correct)}/{num_val_samples})")

    # --- 5. Generate and Save Graphs ---
    all_force_embeds = np.concatenate(all_force_embeds)
    all_text_embeds = np.concatenate(all_text_embeds)
    
    generate_similarity_heatmap(first_batch_logits, os.path.join(args.output_dir, "similarity_heatmap.png"))
    generate_tsne_plot(all_force_embeds, all_text_embeds, all_annotations, os.path.join(args.output_dir, "tsne_plot.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Force-Text contrastive model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth) file.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the formatted training CSV file.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save output graphs.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio, must match training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    evaluate(args)