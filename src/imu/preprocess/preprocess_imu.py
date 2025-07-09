import glob
import json
import math
import os
import subprocess
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd


# ego4d.jsonを見て、scenarioが複数あるビデオを削除
def remove_videos_with_multiple_scenarios(json_file, uid_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    with open(uid_file, "r") as f:
        content = f.read()
        video_uids = content.split()
    for video in data["videos"]:
        video_uid = video["video_uid"]
        scenarios = video["scenarios"]
        if len(scenarios) > 1:
            if video_uid in video_uids:
                video_uids.remove(video_uid)
    return video_uids


# IMUを2000行ずつで分ける
# この時タイムスタンプが飛んでいないことを確認
def split_imu_data(imu_path, output_dir, gaps, uid):
    imu_data = pd.read_csv(imu_path)
    # canonical_timestamp_msでソート
    imu_data.sort_values(by="canonical_timestamp_ms", inplace=True)
    imu_data["timediff"] = imu_data["canonical_timestamp_ms"].diff().fillna(0)
    gaps.append(imu_data["timediff"].mean())
    imu_data["discontinuity"] = imu_data["timediff"] > 30
    imu_data["group"] = imu_data["discontinuity"].cumsum()
    chunk_counter = 0

    for group_id, df_group in imu_data.groupby("group"):
        if len(df_group) < 2000:
            continue
        num_chunks = len(df_group) // 2000
        for i in range(num_chunks):
            imu = df_group.iloc[i * 2000 : (i + 1) * 2000].copy()
            time_diff = imu["canonical_timestamp_ms"].diff().fillna(0)
            gap_mean = time_diff.mean()
            print(
                f"Chunk {chunk_counter} mean gap: {gap_mean}ms, max gap: {time_diff.max()}ms, min gap: {time_diff.min()}ms"
            )
            imu.to_csv(f"{output_dir}/{uid}_{chunk_counter}.csv", index=False)
            chunk_counter += 1
    # 2000行ずつに分割
    return gaps
    # imu.to_csv(f"{output_dir}/imu_chunk_{i // 2000}.csv", index=False)


def main():
    json_file = "data/ego4d.json"
    uid_file = "data/uids.txt"

    # JSONファイルから複数のシナリオを持つビデオを削除
    filtered_uids = remove_videos_with_multiple_scenarios(json_file, uid_file)
    # 結果を新しいファイルに保存
    output_file = "data/filtered_uids.txt"
    with open(output_file, "w") as f:
        f.write(" ".join(filtered_uids))
    gaps = []
    for uid in filtered_uids:
        video_path = f"/large/ego4d/v2/full_scale/{uid}.mp4"
        imu_path = f"/large/ego4d/v2/imu/{uid}.csv"
        gaps = split_imu_data(imu_path, "/large/ego4d/preprocessed/imu", gaps, uid)
        print("")


if __name__ == "__main__":
    main()
