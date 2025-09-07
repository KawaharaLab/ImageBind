import math
import os

import fire
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

import wandb
from imagebind.models.force_model import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F


DATA_TYPE = "train"

DATA_DIR = f"/home/user/Genesis/data/{DATA_TYPE}/"

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
    "right_tz"
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


class SigmoidLoss(nn.Module):
    """
    Implementation of Sigmoid Loss proposed in SigLIP.
    Includes learnable temperature (t) and bias (b).
    """
    def __init__(self, initial_t_prime=0.0, initial_b=0.0):
        super(SigmoidLoss, self).__init__()
        # Register as nn.Parameter to make it part of the optimizer's parameters
        self.t_prime = nn.Parameter(torch.tensor(initial_t_prime))
        self.b = nn.Parameter(torch.tensor(initial_b))

    def forward(self, force_embeddings, text_embeddings):
        """
        Args:
            force_embeddings (torch.Tensor): Output of the force encoder [n, dim]
            text_embeddings (torch.Tensor): Output of the text encoder [n, dim]
        """
        n = force_embeddings.shape[0]
        device = force_embeddings.device

        # 2. Compute temperature (t) and bias (b)
        # Use exp to ensure t remains positive
        t = torch.exp(self.t_prime)

        # 3. Compute logits
        # (zimg @ ztxt.T) * t + b
        logits = (force_embeddings @ text_embeddings.T) * t + self.b
        
        # 4. Create label matrix (-1 with diagonal elements as 1)
        # 2 * eye(n) - ones(n)
        labels = 2 * torch.eye(n, device=device) - torch.ones(n, n, device=device)

        # 5. Compute loss
        # -sum(log_sigmoid(labels * logits)) / n
        loss = -torch.sum(F.logsigmoid(labels * logits)) / n
        
        return loss


class CustomLRScheduler(_LRScheduler):
    def __init__(self, peak_lr, warmup_epochs, total_epochs, optimizer, last_epoch=-1):
        # Define custom attributes required by get_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        # Define base_lrs to match the number of parameter groups in the optimizer for robustness
        self.base_lrs = [peak_lr] * len(optimizer.param_groups)

        # Finally, call the parent class's __init__
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]  # Linear increase during warmup
        else:
            decay_ratio = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.total_epochs - self.warmup_epochs)
                )
            )
            return [max(base_lr * decay_ratio, 1e-5) for base_lr in self.base_lrs]


class ForceDataset(Dataset):
    def __init__(
        self,
        data_len: int = 80,
        use_cols: list = PURE_FORCE_COLS,
    ):
        super().__init__()
        self.data_len = data_len
        self.use_cols = use_cols

        train_csv = os.path.join(DATA_DIR, "train_thin_20pct.csv")
        train_df = pd.read_csv(train_csv)

        self.annotations_emb = []
        self.force_segments = []
        
        unique_csv_paths = train_df["csv_path"].unique()
        
        data_cache = {
            path: pd.read_csv(DATA_DIR + "csv/" + path, usecols=self.use_cols).values.astype("float32")
            for path in unique_csv_paths
        }
        print(f"Loaded {len(data_cache)} unique CSV files into memory.")

        for _, row in train_df.iterrows():
            csv_path = row["csv_path"]
            start_id = row["start"]
            emb_path = f"{DATA_DIR}text_emb/{row['emb_index']}.pt"
            force_segment = data_cache[csv_path][start_id : start_id + self.data_len, :]
            if force_segment.shape[0] != self.data_len:
                print(f"Warning: Skipping segment from {csv_path} at start_id {start_id} due to shape mismatch.")
                continue

            text_emb = torch.load(emb_path, map_location="cpu")
            self.force_segments.append(force_segment)
            self.annotations_emb.append(text_emb)

        if not self.force_segments:
            raise RuntimeError(f"No usable pairs in {DATA_DIR}")

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, idx):
        force_array = self.force_segments[idx]
        force_tensor = torch.from_numpy(force_array).T
        
        annotation_emb = self.annotations_emb[idx]

        return force_tensor, annotation_emb


def main(
    epochs: int = 100,
    warmup_epochs: int = 20,
    batch_size: int = 512,
    gradient_clipping: float = 1.0,
    temperature: float = 0.2,
    weight_decay = None,
    peak_lr: float = 5e-4,
    drop_path: float = 0.3,
    num_blocks: int = 6,
    out_embed_dim: int = 768,
    data_len: int = 80,
    mode: str = "pure",
):
    if mode == "pure":
        data_channels = 12
        use_cols = PURE_FORCE_COLS
        project_name = "icra_siglip"
    elif mode == "compact":
        data_channels = 14
        use_cols = COMPACT_FORCE_COLS
        project_name = "icra_siglip_width"

    wandb.login(key="c85b817c62f441243d232b381088358e72fa2b19")
    wandb.init(
        project=project_name,
        config={
            "model": mode,
            "batch_size": batch_size,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "gradient_clipping": gradient_clipping,
            "temperature": temperature,
            "weight_decay": weight_decay,
            "peak_lr": peak_lr,
            "drop_path": drop_path,
            "num_blocks": num_blocks,
            "out_embed_dim": out_embed_dim,
            "data_len": data_len,
            "train_thinout": 20,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_encoder = load_model(
        # pretrained=True, ckpt_path="/home/mdxuser/ImageBind/data/normal/skilled-sky-26.pth",
        drop_path=drop_path, num_blocks=num_blocks, out_embed_dim=out_embed_dim, data_channels=data_channels, data_len=data_len, temperature=temperature
    ).to(device).float()
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    train_dataset = ForceDataset(data_len=data_len, use_cols=use_cols)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    # optimizer = torch.optim.Adam(force_encoder.parameters(), lr=peak_lr, weight_decay=weight_decay)
    criterion = SigmoidLoss().to(device)
    import itertools
    all_params = itertools.chain(force_encoder.parameters(), criterion.parameters())
    optimizer = torch.optim.Adam(all_params, lr=peak_lr)

    scheduler = CustomLRScheduler(peak_lr, warmup_epochs, epochs, optimizer)

    os.makedirs(f"data/models/siglip/{mode}", exist_ok=True)
    run_name = wandb.run.name
    for epoch in range(epochs):
        force_encoder.train()
        scheduler.step()
        train_loss = 0
        for i, (forces, annotations_emb) in enumerate(train_loader):
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]")
            optimizer.zero_grad()

            forces = forces.to(device, dtype=torch.float32)
            Force_e = force_encoder(forces)
            Force_e = Force_e / Force_e.norm(dim=-1, keepdim=True)

            Text_e = annotations_emb.to(device)
            loss = criterion(Force_e, Text_e)
            if torch.isnan(loss):
                print(f"NaN detected at Epoch [{epoch + 1}], Step [{i}]")
                return
            loss.backward()
            if gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(force_encoder.parameters(), gradient_clipping)
            optimizer.step()
            train_loss += loss.item()

        wandb.log(
            {
                "train_loss": train_loss / len(train_loader),
                "epoch": epoch + 1,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if epoch % 50 == 0:
            torch.save(
                force_encoder.state_dict(), f"data/models/siglip/{mode}/{run_name}_epoch_{epoch}.pth"
            )
    torch.save(force_encoder.state_dict(), f"data/models/siglip/{mode}/{run_name}.pth")
    print(f"Training complete. Model saved as {run_name}.pth")


if __name__ == "__main__":
    fire.Fire(main)  # Allows command line arguments to override defaults
