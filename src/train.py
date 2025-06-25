import os
import glob
import random

import torch
import wandb
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import torchvision.transforms as T
from PIL import Image
import fire
import numpy as np

from imagebind.models.imagebind_model import imagebind_huge_imu
from imagebind.data import load_and_transform_text

USE_FORCE_COLS = [
    "left_fx", "left_fy", "left_fz", 
    "right_fx", "right_fy", "right_fz",
    "dof_0", "dof_1", "dof_2", "dof_3", "dof_4", "dof_5", "dof_6", "dof_7", "dof_8"
]

class ForceDataset(Dataset):
    def __init__(self,
                 device,
                 data_dir: str = "/home/mdxuser/sim/Genesis/data/",
                 force_len: int = 3000,
    ):

        super().__init__()
        self.device = device
        self.force_len = force_len

        train_csv = os.path.join(data_dir, "train.csv")
        train_df = pd.read_csv(train_csv)
        self.pairs = []
        for _, row in train_df.iterrows():
            csv_path = row['csv_path']
            start_id = row['timestep_start']
            annotation = row['analysis_result']
            self.pairs.append((csv_path, start_id, annotation))

        if not self.pairs:
            raise RuntimeError(f"No usable pairs in {data_dir}")

    # -------------- Dataset インタフェース --------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        csv_path, start_id, annotation = self.pairs[idx]

        # ---------------- FORCE: 必要 15列だけ読む ----------------
        force_df = pd.read_csv(csv_path)
        force_cols = USE_FORCE_COLS
        force_array = force_df[force_cols].values.astype('float32')[start_id:start_id + self.force_len, :]
        for col in range(len(force_cols)):
            # NaN を 0 に置き換え
            y = force_array[:, col]
            x = np.arange(len(y))
            not_nan = ~np.isnan(y)
            y_interp = np.interp(x, x[not_nan], y[not_nan])
            force_array[:, col] = y_interp
        force_tensor = torch.from_numpy(force_array).T        # → (15, T)
        force_tensor = force_tensor.to(self.device)  # デバイスに転送

        return force_tensor, annotation

def main(
    epochs: int = 8,
    warmup_epochs: int = 2, #TODO
    batch_size: int = 16,
    gradient_clipping: float = 1.0,
    data_dir: str = "/home/mdxuser/sim/Genesis/data/",
    temperature: float = 0.2,
    weight_decay: float = 0.5,
    peak_lr: float = 5e-4, # TODO
    pretrained: bool = True,
):
    model = imagebind_huge_imu(pretrained=True)
    project_name = "imagebind_force"
    wandb.login(key="c85b817c62f441243d232b381088358e72fa2b19")
    wandb.init(
        project=project_name,
        config={
            "model": "imagebind_huge",
            "pretrained": True,
            "batch_size": batch_size,
            "epochs": epochs,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device).float()
    exp_temp = torch.tensor(temperature, dtype=torch.float32).exp().to(device)

    for name, param in model.named_parameters():
        if "force" in name:
            # print(f"Training parameter: {name}")
            param.requires_grad = True
        else:
            param.requires_grad = False

    train_dataset = ForceDataset(data_dir=data_dir, device=device, force_len=3000)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (forces, annotations) in enumerate(train_loader):
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]")
            optimizer.zero_grad()
            forces = forces.to(device, dtype=torch.float32)
            annotations = load_and_transform_text(annotations, device)
            annotations = annotations.to(device)
            print(f"forces shape: {forces.shape}")
            print(f"annotations shape: {annotations.shape}")
            Force_f = model.encode_force(forces)
            Text_f = model.encode_text(annotations)
            print(f"Force_f shape: {Force_f.shape}")
            print(f"Text_f shape: {Text_f.shape}")
            Force_e = Force_f / Force_f.norm(dim=-1, keepdim=True)
            Text_e = Text_f / Text_f.norm(dim=-1, keepdim=True)
            logits = (Force_e @ Text_e.T) * exp_temp
            labels = torch.arange(len(annotations)).to(device)
            loss_force = criterion(logits, labels)
            loss_text = criterion(logits.T, labels)
            loss = (loss_text + loss_force) / 2
            if torch.isnan(loss):
                print(f"NaN detected at Epoch [{epoch + 1}], Step [{i}]")
                return
            loss.backward()
            if gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            train_loss += loss.item()
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "epoch": epoch + 1,
                    "step": i + 1,
                }
            )
            if i % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
    torch.save(model.state_dict(), f"model_{project_name}.pth")
    print(f"Training complete. Model saved as model_{project_name}.pth")

if __name__ == "__main__": 
    fire.Fire(main)  # Allows command line arguments to override defaults