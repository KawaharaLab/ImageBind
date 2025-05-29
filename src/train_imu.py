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

from imagebind.models.imagebind_model import imagebind_huge_imu




class IMUDataset(Dataset):
    """
    ディレクトリ構成
    ---------------
    <data_dir>/
        imu/    foo_0.csv,  bar_12.csv, ...
        video/  foo_0.mp4,  bar_12.mp4, ...
    """
    def __init__(
        self,
        data_dir: str,
        imu_len: int = 2000,
        frame_sampling: str = "center",   # "center" or "random"
        transform=None,
    ):
        super().__init__()
        self.imu_len = imu_len
        self.frame_sampling = frame_sampling.lower()
        self.transform = transform or T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        imu_dir   = os.path.join(data_dir, "imu")
        video_dir = os.path.join(data_dir, "video")

        csv_paths = sorted(glob.glob(os.path.join(imu_dir, "*.csv")))
        self.pairs = []
        for csv_path in csv_paths:
            fname   = os.path.basename(csv_path)
            uid_ch  = os.path.splitext(fname)[0]                 # foo_12
            mp4_path = os.path.join(video_dir, uid_ch + ".mp4")
            if os.path.exists(mp4_path):
                self.pairs.append((csv_path, mp4_path))
            else:
                print(f"[WARN] video not found for {fname}, skip.")

        if len(self.pairs) == 0:
            raise RuntimeError(f"No imu–video pairs found in {data_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        csv_path, mp4_path = self.pairs[idx]

        # ------------------- IMU -------------------
        df = pd.read_csv(csv_path,
                         usecols=[
                             "gyro_x", "gyro_y", "gyro_z",
                             "accl_x", "accl_y", "accl_z",
                         ])
        imu = torch.tensor(df.values, dtype=torch.float32)      # [N, 6]
        # 長さを固定（truncate / pad）
        if imu.shape[0] >= self.imu_len:
            imu = imu[:self.imu_len]
        else:
            pad = torch.zeros(self.imu_len - imu.shape[0], 6)
            imu = torch.cat([imu, pad], dim=0)                  # [L, 6]

        # ------------------- VIDEO -----------------
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {mp4_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.frame_sampling == "random":
            fidx = random.randint(0, total_frames - 1)
        else:                                 # center
            fidx = total_frames // 2

        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        success, frame = cap.read()
        cap.release()
        if not success:
            raise RuntimeError(f"Failed to read frame {fidx} from {mp4_path}")

        # BGR → RGB → PIL → transform
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img   = Image.fromarray(frame)
        img   = self.transform(img)           # [3, 224, 224] float32

        return img, imu                      # (vision_tensor, imu_tensor)

def main(
    epochs: int = 8,
    warmup_epochs: int = 2, #TODO
    batch_size: int = 512,
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
        name="imagebind_imu",
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
        for i, (images, imus) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device, dtype=torch.float32)
            imus = imus.to(device, dtype=torch.long)
            with torch.no_grad():
                Image_f = model.encode_image(images)
            Imu_f = model.encode_imu(imus)
            Image_e = Image_f / Image_f.norm(dim=-1, keepdim=True)
            Imu_e = Imu_f / Imu_f.norm(dim=-1, keepdim=True)
            logits = (Image_e @ Imu_e.T) * exp_temp

            labels = torch.arange(len(images)).to(device)
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

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )