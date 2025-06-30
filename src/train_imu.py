import glob
import os
import random

import cv2
import fire
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import wandb
from imagebind.models.imagebind_model import imagebind_huge_imu

# ---------------- 前処理（動画用） ----------------
IMG_TRANSFORM = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

USE_IMU_COLS = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accl_x",
    "accl_y",
    "accl_z",
]


class IMUDataset(Dataset):
    def __init__(
        self,
        device,
        data_dir: str,
        txt: str = "data/train.txt",
        imu_len: int = 2000,
        frame_sampling: str = "center",
        transform=IMG_TRANSFORM,
    ):
        """
        data_dir/imu   : {uid}_{chunk}.csv
        data_dir/video_feature : {uid}_{chunk}.pt
        """
        super().__init__()
        self.device = device
        self.imu_len = imu_len
        self.frame_sampling = frame_sampling.lower()
        self.transform = transform

        imu_dir = os.path.join(data_dir, "imu")
        video_dir = os.path.join(data_dir, "video_feature")
        with open(txt, "r") as f:
            uidchunks = [line.strip() for line in f if line.strip()]
        self.pairs = []
        for uidchunk in uidchunks:
            csv_path = os.path.join(imu_dir, uidchunk + ".csv")
            pt_path = os.path.join(video_dir, uidchunk + ".pt")
            if os.path.exists(pt_path) and os.path.exists(csv_path):
                self.pairs.append((csv_path, pt_path))
            else:
                print(f"[WARN] skip: not found {uidchunk}")

        if not self.pairs:
            raise RuntimeError(f"No usable pairs in {data_dir}")

    # -------------- Dataset インタフェース --------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        csv_path, pt_path = self.pairs[idx]

        # ---------------- IMU: 必要 6 列だけ読む ----------------
        imu_df = pd.read_csv(csv_path)
        # gyro_x, gyro_y, gyro_z, accl_x, accl_y, accl_z の 6 列だけ抽出
        imu_cols = ["gyro_x", "gyro_y", "gyro_z", "accl_x", "accl_y", "accl_z"]
        imu_array = imu_df[imu_cols].values.astype("float32")
        for col in range(len(imu_cols)):
            # NaN を 0 に置き換え
            y = imu_array[:, col]
            x = np.arange(len(y))
            not_nan = ~np.isnan(y)
            y_interp = np.interp(x, x[not_nan], y[not_nan])
            imu_array[:, col] = y_interp
        imu_tensor = torch.from_numpy(imu_array).T  # → (6, T)

        imu_tensor = imu_tensor.to(self.device)  # デバイスに転送
        # ---------------- VIDEO: 全フレーム取得 ----------------
        video_tensor = torch.load(pt_path).to(self.device)
        video_tensor = video_tensor.mean(dim=0)  # 平均化して [T, 1024] → [1024]
        return video_tensor, imu_tensor  # vision_tensor, imu_tensor


def main(
    epochs: int = 8,
    warmup_epochs: int = 2,  # TODO
    batch_size: int = 512,
    gradient_clipping: float = 1.0,
    data_dir: str = "/large/ego4d/preprocessed/",  # TODO
    temperature: float = 0.2,
    weight_decay: float = 0.5,
    peak_lr: float = 5e-4,
    pretrained: bool = True,
):
    model = imagebind_huge_imu(pretrained=True)
    project_name = "imagebind_imu"
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
    print(f"Using device: {device}")
    model.to(device).float()
    exp_temp = torch.tensor(temperature, dtype=torch.float32).exp().to(device)

    for name, param in model.named_parameters():
        if "imu" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    train_dataset = IMUDataset(
        data_dir=data_dir, device=device, imu_len=2000, frame_sampling="center"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (videos, imus) in enumerate(train_loader):
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]")
            optimizer.zero_grad()
            videos = videos.to(device, dtype=torch.float32)
            imus = imus.to(device, dtype=torch.float32)
            if videos.isnan().any():
                print(f"NaN detected in videos at Epoch [{epoch + 1}], Step [{i + 1}]")
                return
            if imus.isnan().any():
                print(f"NaN detected in imus at Epoch [{epoch + 1}], Step [{i + 1}]")
                return
            Imu_f = model.encode_imu(imus)
            Image_e = videos / videos.norm(dim=-1, keepdim=True)
            Imu_e = Imu_f / Imu_f.norm(dim=-1, keepdim=True)
            logits = (Image_e @ Imu_e.T) * exp_temp

            labels = torch.arange(len(videos)).to(device)
            loss_image = criterion(logits, labels)
            loss_imu = criterion(logits.T, labels)
            loss = (loss_image + loss_imu) / 2
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
