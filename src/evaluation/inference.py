from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fire import Fire

import clip
from imagebind import data

data_dir = "/home/mdxuser/sim/Genesis/data/"


PURE_FORCE_COLS = [
    "left_fx",
    "left_fy",
    "left_fz",
    "right_fx",
    "right_fy",
    "right_fz",
]


def main(model_path):
    #========================== TODO: change to json ============================
    temperature: float = 0.2
    drop_path: float = 0.3
    num_blocks: int = 4
    out_embed_dim: int = 512
    data_len: int = 300
    #==========================================================================
    device = torch.device("cpu")
    data_channels = 6
    use_cols = PURE_FORCE_COLS

    from imagebind.models.force_model import load_model
    force_encoder = load_model(
        pretrained=True, ckpt_path=model_path,
        drop_path=drop_path, num_blocks=num_blocks, out_embed_dim=out_embed_dim, data_channels=data_channels, data_len=data_len, temperature=temperature
    ).to(device).float()


    force_encoder.eval()

    eval_path = data_dir + "eval.csv"
    eval_df = pd.read_csv(eval_path)

    # Confusion matrix 用に初期化
    # 真のラベル → 予測ラベル → カウント
    inference_times = []
    for _, row in eval_df.iterrows():
        force_csv = row["csv_path"]
        start = row["timestep_start"]*data_len//3000
        force_df = pd.read_csv(force_csv).iloc[::3000//data_len].reset_index(drop=True)
        force_array = force_df[use_cols].values.astype("float32")[start : start + data_len, :]
        now = time.time()
        force_tensor = torch.from_numpy(force_array).T  # → (15, T)
        force_tensor = force_tensor.unsqueeze(0).to(device)  # → (1, 15, T)

        with torch.no_grad():
            fe = force_encoder(force_tensor)  # → (1, D)
        then = time.time()
        inference_times.append(then - now)

    inference_times = np.array(inference_times)
    print(f"Average inference time: {inference_times.mean():.4f} seconds")

if __name__ == "__main__":
    Fire(main)
