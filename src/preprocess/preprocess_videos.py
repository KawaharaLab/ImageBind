import json
import os
import math
import subprocess
import glob
from collections import defaultdict

import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image

from imagebind.data import load_and_transform_video_data
from imagebind.models.imagebind_model import imagebind_huge

IMG_TRANSFORM = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

#各動画をIMUに合わせて切り抜く
def cut_videos_to_imu(imu_dir: str,
                      video_dir: str,
                      output_dir: str,
                      codec_copy: bool = True):

    os.makedirs(output_dir, exist_ok=True)

    # すべての 2000 行 IMU チャンクを列挙
    csv_paths = sorted(glob.glob(os.path.join(imu_dir, "*.csv")))

    for csv_path in csv_paths:
        # ----------- ファイル名から uid と chunk 番号を取得 -------------
        fname = os.path.basename(csv_path)
        if '_' not in fname:
            print(f"[SKIP] アンダースコアが無い: {fname}")
            continue

        uid, chunk_part = fname.rsplit('_', 1)        # 末尾の '_' で分割
        chunk_id = os.path.splitext(chunk_part)[0]    # .csv を外す
        out_mp4 = os.path.join(output_dir, f"{uid}_{chunk_id}.mp4")

        if os.path.exists(out_mp4):
            print(f"[SKIP] {out_mp4} already exists.")
            continue
        video_path = os.path.join(video_dir, f"{uid}.mp4")
        if not os.path.exists(video_path):
            print(f"[WARN] video が無い: {video_path}")
            continue

        # ------------------------ IMU 読み込み -------------------------
        imu_chunk = pd.read_csv(csv_path)
        imu_chunk.sort_values("canonical_timestamp_ms", inplace=True)

        start_ms = imu_chunk["canonical_timestamp_ms"].iloc[0]
        end_ms   = imu_chunk["canonical_timestamp_ms"].iloc[-1]
        dur_ms   = end_ms - start_ms
        if dur_ms <= 0:
            print(f"[WARN] duration<=0 ({fname})")
            continue

        start_sec = start_ms / 1000.0
        dur_sec   = dur_ms   / 1000.0

        # ----------------------- ffmpeg コマンド -----------------------
        if codec_copy:
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_sec:.3f}",
                "-i",  video_path,
                "-t",  f"{dur_sec:.3f}",
                "-c",  "copy",
                out_mp4
            ]
        else:  # フレーム単位できっちり切りたい場合は再エンコード
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_sec:.3f}",
                "-i",  video_path,
                "-t",  f"{dur_sec:.3f}",
                "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                "-c:a", "copy",
                out_mp4
            ]

        print(f"[ffmpeg] {uid}_{chunk_id}: start={start_sec:.3f}s dur={dur_sec:.3f}s")
        try:
            subprocess.run(cmd,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT,
                           check=True)
        except subprocess.CalledProcessError:
            print(f"[ERR] ffmpeg 失敗: {out_mp4}")

def encode_videos():
    failures = []
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # For debugging on CPU
    model = imagebind_huge(pretrained=True).eval().to(device)
    video_dir = "/large/ego4d/preprocessed/video"
    output_dir = "/large/ego4d/preprocessed/video_feature"
    os.makedirs(output_dir, exist_ok=True)
    for mp4_path in sorted(glob.glob(os.path.join(video_dir, "*.mp4"))):
        uidchunk = os.path.splitext(os.path.basename(mp4_path))[0]
        uidchunk = "000cd456-ff8d-499b-b0c1-4acead128a8b_0"
        mp4_path = os.path.join(video_dir, f"{uidchunk}.mp4")
        # if os.path.exists(os.path.join(output_dir, f"{uidchunk}.pt")):
        #     continue
        print(f"[INFO] Encoding {uidchunk}...")
        try:
            video = load_and_transform_video_data(
                [mp4_path], device=device, clips_per_video=10
            )
            video = video.to(device)
            print("video", video.shape)
            with torch.no_grad():
                video_feature = model.encode_image(video)
            out_path = os.path.join(output_dir, f"{uidchunk}.pt")
            torch.save(video_feature, out_path)
            del video_feature
            del video
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ERR] Failed to encode {uidchunk}: {e}")
            failures.append(uidchunk)
        exit()
    if failures:
        with open("video_encoding_failures.txt", "w") as f:
            for uidchunk in failures:
                f.write(f"{uidchunk}\n")

    
if __name__ == "__main__":
    imu_chunks_dir    = "/large/ego4d/preprocessed/imu"          # 2000 行 IMU CSV がある場所
    full_video_dir    = "/large/ego4d/v2/full_scale"   # 元動画 <uid>.mp4
    clipped_video_dir = "/large/ego4d/preprocessed/video"
    # os.makedirs(clipped_video_dir, exist_ok=True)
    encode_videos()  # 動画をエンコードして特徴量を抽出する
    # cut_videos_to_imu(imu_chunks_dir, full_video_dir, clipped_video_dir)
    
