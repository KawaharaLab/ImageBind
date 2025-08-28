import torch
import pandas as pd
from pathlib import Path
import torch.nn.functional as F
import numpy as np

from imagebind.models.imagebind_model import imagebind_huge
from imagebind import data

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    model = imagebind_huge(pretrained=True)
    model.eval().to(device)

    # label の読み込み
    labels = []
    with open("data/scenarios.txt", "r") as f:
        for line in f:
            labels.append(line.strip())
    labels_preprocessed = data.load_and_transform_text(labels, device).to(device)
    with torch.no_grad():
        labels_encoded = model.encode_text(labels_preprocessed)

    imu_path = "/large/ego4d/preprocessed/imu/"

    # eval.txt に書かれたファイル名のみで評価を行う
    with open("data/eval.txt", "r") as f:
        eval_files = [l.strip() for l in f if l.strip()]

    correct_pairs = {}
    correct_pairs_df = pd.read_csv("data/uid_scenario_imu.csv")
    for _, row in correct_pairs_df.iterrows():
        correct_pairs[row['video_uid']] = row['scenario']

    n_correct = 0
    total = 0
    for fname in eval_files:
        imu_file = imu_path + "/" + fname + ".csv"
        uid = fname.split("_")[0]
        correct = "False"
        imu_df = pd.read_csv(imu_file)
        # gyro_x, gyro_y, gyro_z, accl_x, accl_y, accl_z の 6 列だけ抽出
        imu_cols = ['gyro_x', 'gyro_y', 'gyro_z', 'accl_x', 'accl_y', 'accl_z']
        imu_array = imu_df[imu_cols].values.astype('float32')
        for col in range(len(imu_cols)):
            y = imu_array[:, col]
            x = np.arange(len(y))
            not_nan = ~np.isnan(y)
            y_interp = np.interp(x, x[not_nan], y[not_nan])
            imu_array[:, col] = y_interp
        # モデルの期待形状に合わせて必要なら転置 (ここではチャネル×時系列長)
        imu_tensor = torch.from_numpy(imu_array).T       # → (6, T)
        imu_tensor = imu_tensor.unsqueeze(0)              # → (1, 6, T)
        # imu_tensor = F.interpolate(
        #     imu_tensor, scale_factor=2, mode='linear', align_corners=False
        # )                                                 # → (1, 6, 2*T)
        # print("imu_tensor shape:", imu_tensor.shape)
        # imu_tensor = imu_tensor[:, :, :2000]              # → (1, 6, 2000)
        imu_tensor = imu_tensor.to(device)

        with torch.no_grad():
            imu_encoded = model.encode_imu(imu_tensor)
        similarities = torch.softmax(imu_encoded @ labels_encoded.T, dim=-1)
        most_similar_index = similarities.argmax().item()
        most_similar_label = labels[most_similar_index]
        if most_similar_label == correct_pairs[uid]:
            n_correct += 1
            correct = "True"

        print(f"{fname}: predict={most_similar_label}, true={correct_pairs[uid]}")
        with open("data/predictions_new.txt", "a") as f:
            f.write(f"{fname}: {correct} predict={most_similar_label}, true={correct_pairs[uid]}\n")

        total += 1
        print(f"Total: {total}, Correct: {n_correct}")

    print(f"Accuracy: {n_correct/total*100:.2f}%")
    with open("data/predictions_new.txt", "a") as f:
        f.write(f"Accuracy: {n_correct/total*100:.2f}%\n")

if __name__ == "__main__":
    main()