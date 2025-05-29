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

from imagebind.models.imagebind_model import imagebind_huge_imu

# ---------------- 前処理（動画用） ----------------
IMG_TRANSFORM = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

USE_IMU_COLS = [
    "gyro_x", "gyro_y", "gyro_z",
    "accl_x", "accl_y", "accl_z",
]

class IMUDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 imu_len: int = 2000,
                 frame_sampling: str = "center",
                 transform=IMG_TRANSFORM):
        """
        data_dir/imu   : {uid}_{chunk}.csv
        data_dir/video : {uid}_{chunk}.mp4
        """
        super().__init__()
        self.imu_len = imu_len
        self.frame_sampling = frame_sampling.lower()
        self.transform = transform

        imu_dir   = os.path.join(data_dir, "imu")
        video_dir = os.path.join(data_dir, "video")

        self.pairs = []
        for csv_path in sorted(glob.glob(os.path.join(imu_dir, "*.csv"))):
            uidchunk = os.path.splitext(os.path.basename(csv_path))[0]
            mp4_path = os.path.join(video_dir, uidchunk + ".mp4")
            if os.path.exists(mp4_path):
                self.pairs.append((csv_path, mp4_path))
            else:
                print(f"[WARN] skip: video not found {mp4_path}")

        if not self.pairs:
            raise RuntimeError(f"No usable pairs in {data_dir}")

    # -------------- Dataset インタフェース --------------
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        csv_path, mp4_path = self.pairs[idx]

        # ---------------- IMU: 必要 6 列だけ読む ----------------
        df = pd.read_csv(csv_path, usecols=USE_IMU_COLS)
        imu = torch.tensor(df.values, dtype=torch.float32)  # [L, 6]

        # 長さ固定
        if imu.shape[0] >= self.imu_len:
            imu = imu[:self.imu_len]
        else:
            pad = torch.zeros(self.imu_len - imu.shape[0], 6)
            imu = torch.cat([imu, pad], dim=0)

        # ---------------- VIDEO: 全フレーム取得 ----------------
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {mp4_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.transform(Image.fromarray(frame))  # [3,224,224]
            frames.append(image)
        cap.release()
        print(mp4_path, len(frames))
        if len(frames) == 0:
            raise RuntimeError(f"No frames could be read from {mp4_path}")

        video_tensor = torch.stack(frames, dim=1)  # [3, video_frames, 224, 224]

        return video_tensor, imu         # vision_tensor, imu_tensor

def main(
    epochs: int = 8,
    warmup_epochs: int = 2, #TODO
    batch_size: int = 128,
    gradient_clipping: float = 1.0,
    data_dir: str = "/large/ego4d/preprocessed/", #TODO
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
    model.to(device)
    exp_temp = torch.tensor(temperature, dtype=torch.float32).exp().to(device)

    for name, param in model.named_parameters():
        if "imu" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    train_dataset = IMUDataset(data_dir=data_dir, imu_len=2000, frame_sampling="center")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, (videos, imus) in enumerate(train_loader):
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]")
            optimizer.zero_grad()
            videos = videos.to(device, dtype=torch.float32)
            imus = imus.to(device, dtype=torch.float32)
            print(f"imus shape: {imus.shape}, videos shape: {videos.shape}")
            with torch.no_grad():
                Image_f = model.encode_image(videos)
            Imu_f = model.encode_imu(imus)
            Image_e = Image_f / Image_f.norm(dim=-1, keepdim=True)
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


if __name__ == "__main__": 
    fire.Fire(main)  # Allows command line arguments to override defaults