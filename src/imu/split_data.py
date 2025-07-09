import random
from pathlib import Path

import torch

video_feature_path = "/large/ego4d/preprocessed/video_feature"
video_feature_dir = Path(video_feature_path)

# ランダムに9:1で分割して train.txt, eval.txt を出力
all_files = [f.name for f in video_feature_dir.glob("*.pt")]

for i in range(len(all_files)):
    print(all_files[i])
    torch.load(video_feature_dir / all_files[i]).to("cuda")  # 動作確認のために読み込む

#     all_files[i] = all_files[i].replace(".pt", "")
# random.shuffle(all_files)
# split_idx = int(len(all_files) * 0.9)
# train_files = all_files[:split_idx]
# eval_files = all_files[split_idx:]

# with open("data/train.txt", "w") as f:
#     f.write("\n".join(train_files))
# with open("data/eval.txt", "w") as f:
#     f.write("\n".join(eval_files))
