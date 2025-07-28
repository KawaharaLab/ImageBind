import os
import math

import torch
import clip
import wandb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from imagebind.models.force_model import load_model
from tqdm import tqdm # Import tqdm

# === CONFIGURABLE PATHS ===
BASE_PATH = "/home/mdxuser/ImageBind/src"
TRAIN_CSV = "/home/mdxuser/Genesis/data/formatted_training_data_IB.csv"
MODEL_DIR = "data/contrastive_force_test"

PURE_FORCE_COLS = ["left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz"]

# === DATASET ===
class ForceDataset(Dataset):
    def __init__(self, data_len: int = 100, use_cols: list = PURE_FORCE_COLS):
        train_df = pd.read_csv(TRAIN_CSV)
        self.data_len = data_len
        self.use_cols = use_cols

        unique_csvs = train_df["csv_path"].unique()
        data_cache = {
            path: pd.read_csv(path, usecols=self.use_cols).values.astype("float32")
            for path in unique_csvs
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
        force_tensor = torch.from_numpy(self.force_segments[idx]).T  # (C, T)
        annotation = clip.tokenize([self.annotations[idx]])[0]
        return force_tensor, annotation

# === CUSTOM SCHEDULER ===
class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, peak_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [peak_lr] * len(optimizer.param_groups)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [lr * (self.last_epoch + 1) / self.warmup_epochs for lr in self.base_lrs]
        else:
            decay_ratio = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)))
            return [max(lr * decay_ratio, 1e-5) for lr in self.base_lrs]

# === TRAIN ===
def train():
    wandb.login(key="3f9edde5e58f6c9eab6123b18cf61030047ba716")
    wandb.init(project="imagebind_force_test", config={
        "batch_size": 64,
        "epochs": 400,
        "warmup_epochs": 20,
        "peak_lr": 1e-3,
        "drop_path": 0.35,
        "num_blocks": 6,
        "out_embed_dim": 512,
        "data_len": 100,
        "temperature": 0.07,
        "mode": "normal",
        "validation_split": 0.1 # <-- NEW: Add validation split to config
    })

    cfg = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. SPLIT YOUR DATA ===
    dataset = ForceDataset(data_len=cfg.data_len, use_cols=PURE_FORCE_COLS)

    # Calculate split sizes
    val_size = int(cfg.validation_split * len(dataset))
    train_size = len(dataset) - val_size

    # Perform the split
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create separate DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4) # <-- NEW: Validation loader

    # ... (Model, optimizer, scheduler, and criterion setup remains the same) ...
    force_encoder = load_model(
        drop_path=cfg.drop_path,
        num_blocks=cfg.num_blocks,
        out_embed_dim=cfg.out_embed_dim,
        data_channels=6,
        data_len=cfg.data_len,
        temperature=cfg.temperature
    ).to(device).float()

    clip_encoder, _ = clip.load("ViT-B/16", device=device)
    optimizer = torch.optim.Adam(force_encoder.parameters(), lr=cfg.peak_lr, weight_decay=1e-4)
    scheduler = CustomLRScheduler(optimizer, cfg.warmup_epochs, cfg.epochs, cfg.peak_lr)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(MODEL_DIR, exist_ok=True)
    run_name = wandb.run.name

    for epoch in range(cfg.epochs):
        # --- TRAINING LOOP ---
        force_encoder.train() # Set model to training mode
        scheduler.step()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [TRAIN]") # <-- MODIFIED
        for i, (forces, texts) in enumerate(pbar):
            forces, texts = forces.to(device), texts.to(device)
            optimizer.zero_grad()
            force_emb = force_encoder(forces)
            with torch.no_grad():
                text_emb = clip_encoder.encode_text(texts).float()

            force_emb = force_emb / force_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            logits = (force_emb @ text_emb.T) / cfg.temperature
            labels = torch.arange(len(texts), device=device)
            loss = (criterion(logits, labels) + criterion(logits.T, labels)) / 2.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(force_encoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        # === 2. ADD A VALIDATION LOOP ===
        force_encoder.eval() # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad(): # Disable gradient calculations
            for forces, texts in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [VAL]"): # <-- NEW
                forces, texts = forces.to(device), texts.to(device)
                force_emb = force_encoder(forces)
                text_emb = clip_encoder.encode_text(texts).float() # No torch.no_grad() needed here

                force_emb = force_emb / force_emb.norm(dim=-1, keepdim=True)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

                logits = (force_emb @ text_emb.T) / cfg.temperature
                labels = torch.arange(len(texts), device=device)
                loss = (criterion(logits, labels) + criterion(logits.T, labels)) / 2.0
                val_loss += loss.item()

        # === 3. LOG THE VALIDATION LOSS ===
        wandb.log({
            "train_loss": epoch_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader), # <-- MODIFIED
            "epoch": epoch + 1,
            "learning_rate": scheduler.get_last_lr()[0],
        })

        if (epoch + 1) % 50 == 0:
            torch.save(force_encoder.state_dict(), f"{MODEL_DIR}/{run_name}_epoch_{epoch+1}.pth")

    torch.save(force_encoder.state_dict(), f"{MODEL_DIR}/{run_name}_final.pth")
    print(f"Finished training. Model saved to {MODEL_DIR}/{run_name}_final.pth")

if __name__ == "__main__":
    train()
