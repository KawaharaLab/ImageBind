import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def plot_imu(csv_path, out_png="imu_plot.png"):
    """
    csv_path : canonical_timestamp_ms, gyro_x, gyro_y, gyro_z,
               accl_x, accl_y, accl_z を含む CSV
    out_png  : 保存先 PNG ファイル
    """
    df = pd.read_csv(csv_path)

    # 時間軸（秒）。そのまま ms を使いたいなら /1000 を削除
    t = (df["canonical_timestamp_ms"] - df["canonical_timestamp_ms"].iloc[0]) / 1000.0

    # ----------- 描画 -----------
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), sharex=True)  # X 軸は共有

    gyro_cols = ["gyro_x", "gyro_y", "gyro_z"]
    accl_cols = ["accl_x", "accl_y", "accl_z"]

    for idx, col in enumerate(gyro_cols):
        ax = axes[0, idx]
        ax.plot(t, df[col], color="tab:blue")
        ax.set_title(col)
        ax.set_ylabel("deg/s" if idx == 0 else "")  # 1 枚目だけラベル
        ax.grid(True)

    for idx, col in enumerate(accl_cols):
        ax = axes[1, idx]
        ax.plot(t, df[col], color="tab:orange")
        ax.set_title(col)
        ax.set_xlabel("time [s]")  # 下段だけ X ラベル
        ax.set_ylabel("m/s²" if idx == 0 else "")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"saved → {os.path.abspath(out_png)}")


# ---------------------- 使い方 ----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python plot_imu.py <imu_chunk.csv> [out.png]")
        sys.exit(1)

    csv_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "imu_plot.png"
    plot_imu(csv_file, out_file)
