import math
import os

import fire
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

import clip
import wandb
from imagebind.models.force_model import load_force_encoder

USE_FORCE_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "right_fx",
    "right_fy",
    "right_fz",
    "dof_0",
    "dof_1",
    "dof_2",
    "dof_3",
    "dof_4",
    "dof_5",
    "dof_6",
    "dof_7",
    "dof_8",
]

USE_PURE_FORCE_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "right_fx",
    "right_fy",
    "right_fz",
]


class CustomLRScheduler(_LRScheduler):
    def __init__(self, peak_lr, warmup_epochs, total_epochs, optimizer, last_epoch=-1):
        # 先にget_lrで必要となる独自の属性を定義する
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        # optimizerのパラメータグループの数に合わせてbase_lrsを定義すると、より堅牢になります
        self.base_lrs = [peak_lr] * len(optimizer.param_groups)

        # 最後に親クラスの__init__を呼び出す
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]  # ウォームアップ期間中は線形増加
        else:
            decay_ratio = 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.total_epochs - self.warmup_epochs)
                )
            )
            return [base_lr * decay_ratio for base_lr in self.base_lrs]


class ForceDataset(Dataset):
    def __init__(
        self,
        # device はここから削除します
        data_dir: str = "/home/mdxuser/sim/Genesis/data/",
        force_len: int = 3000,
    ):
        super().__init__()
        self.force_len = force_len

        train_csv = os.path.join(data_dir, "train_simple.csv")
        train_df = pd.read_csv(train_csv)
        
        # --- ここからが改善点 ---
        self.annotations = []
        self.force_segments = []
        
        # 1. ユニークなCSVパスを取得
        unique_csv_paths = train_df["csv_path"].unique()
        
        # 2. 全てのCSVを一度だけ読み、辞書にキャッシュする
        data_cache = {
            path: pd.read_csv(path, usecols=USE_FORCE_COLS).values.astype("float32")
            for path in unique_csv_paths
        }
        print(f"Loaded {len(data_cache)} unique CSV files into memory.")

        # 3. 各サンプルをメモリ上のデータへの参照として保持
        for _, row in train_df.iterrows():
            csv_path = row["csv_path"]
            start_id = row["timestep_start"]
            
            # メモリ上のNumPy配列から直接スライスして追加
            force_segment = data_cache[csv_path][start_id : start_id + self.force_len, :]
            
            self.force_segments.append(force_segment)
            self.annotations.append(row["analysis_result_simple"])
        # --- 改善点ここまで ---

        if not self.force_segments:
            raise RuntimeError(f"No usable pairs in {data_dir}")

    def __len__(self):
        return len(self.force_segments)

    def __getitem__(self, idx):
        force_array = self.force_segments[idx]
        annotation_text = self.annotations[idx]
        
        # torch.from_numpy は高速
        force_tensor = torch.from_numpy(force_array).T  # → (15, T)
        
        # TokenizeはCPUで行う
        annotation = clip.tokenize(annotation_text)[0]

        # GPUへの転送はここでは行わない！
        return force_tensor, annotation


def main(
    epochs: int = 1000,
    warmup_epochs: int = 20,
    batch_size: int = 128,
    gradient_clipping: float = 1.0,
    data_dir: str = "/home/mdxuser/sim/Genesis/data/",
    temperature: float = 0.07,
    weight_decay: float = 0.05,
    peak_lr: float = 1e-4,
):
    project_name = "imagebind_force_simple"
    model_name = "al_ru"
    wandb.login(key="c85b817c62f441243d232b381088358e72fa2b19")
    wandb.init(
        project=project_name,
        config={
            "model": model_name,
            "batch_size": batch_size,
            "epochs": epochs,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_encoder = load_force_encoder(pretrained=True, ckpt_path="/home/mdxuser/ImageBind/data/al_ru/magic-wood-9.pth").to(device).float()
    CLIP_encoder, _ = clip.load("ViT-L/14@336px", device=device)
    # device = torch.device("cpu")
    print(f"Using device: {device}")
    exp_temp = torch.tensor(temperature, dtype=torch.float32).exp().to(device)

    train_dataset = ForceDataset(data_dir=data_dir, force_len=3000)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    optimizer = torch.optim.Adam(force_encoder.parameters(), lr=peak_lr, weight_decay=weight_decay)

    scheduler = CustomLRScheduler(peak_lr, warmup_epochs, epochs, optimizer)

    criterion = torch.nn.CrossEntropyLoss()
    os.makedirs(f"data/{model_name}", exist_ok=True)
    run_name = wandb.run.name
    for epoch in range(epochs):
        force_encoder.train()
        CLIP_encoder.eval()  # CLIPのエンコーダは評価モードに設定
        scheduler.step()
        train_loss = 0
        for i, (forces, annotations) in enumerate(train_loader):
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]")
            optimizer.zero_grad()
            forces = forces.to(device, dtype=torch.float32)
            annotations = annotations.to(device)
            print(f"forces shape: {forces.shape}")
            print(f"annotations shape: {annotations.shape}")
            Force_f = force_encoder(forces)
            with torch.no_grad():
                Text_f = CLIP_encoder.encode_text(annotations).float()
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
                torch.nn.utils.clip_grad_norm_(force_encoder.parameters(), gradient_clipping)
            optimizer.step()
            train_loss += loss.item()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
        wandb.log(
            {
                "train_loss": train_loss / len(train_loader),
                "epoch": epoch + 1,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        if epoch % 10 == 0:
            torch.save(
                force_encoder.state_dict(), f"data/{model_name}/{run_name}_epoch_{epoch + 101}.pth"
            )
    torch.save(force_encoder.state_dict(), f"data/{model_name}/{run_name}.pth")
    print(f"Training complete. Model saved as {run_name}.pth")


if __name__ == "__main__":
    fire.Fire(main)  # Allows command line arguments to override defaults
