import json
import os
import math
import subprocess
import glob
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

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
        out_mp4 = os.path.join(output_dir, f"{uid}_{chunk_id}.mp4")

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
    
    
if __name__ == "__main__":
    imu_chunks_dir    = "/large/ego4d/preprocessed/imu"          # 2000 行 IMU CSV がある場所
    full_video_dir    = "/large/ego4d/v2/full_scale"   # 元動画 <uid>.mp4
    clipped_video_dir = "/large/ego4d/preprocessed/video"

    cut_videos_to_imu(imu_chunks_dir, full_video_dir, clipped_video_dir)
    
