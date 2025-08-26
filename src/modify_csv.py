import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

SEGMENT_LENGTH = 270  # Length of each segment in steps
BASE_PATH = "/home/mdxuser/Genesis/"
MATERIAL = "Elastic"  # Material type

DATA_PATH = os.path.join(BASE_PATH, "main/data/picked_up_4/")

def main(df, steps_df):

    # Partition boundaries
    partitions = steps_df['step start'].values.tolist()
    partitions.append(partitions[-1] + 100)  # Extend the last partition to include the end
    # partitions = [0, 50, 200, 350, 550, 730, 910, 1150, 1420, 1585, 1750, 1850]

    # Prepare list to collect each interpolated partition
    interpolated_parts = []    

    # For each partition:
    for i in range(len(partitions) - 1):
        start, end = partitions[i]+1, partitions[i+1]
        
        # Extract the subset
        segment = df[(df['step'] >= start) & (df['step'] <= end)]
        # print(segment)
        # print(f"Segment {i}: step range {start} to {end} → {len(segment)} rows")
        
        # # Original x (step) and new x (270 evenly spaced steps from start to end)
        original_x = segment['step'].values
        new_x = np.linspace(i*SEGMENT_LENGTH+1, (i+1)*SEGMENT_LENGTH, SEGMENT_LENGTH)
        # print(original_x)
        # print(new_x)
        
        # # Dictionary to store interpolated data
        interp_segment = {'step': new_x}
        
        # # Interpolate each column
        for col in df.columns:
            if col == 'step':
                continue
            f = interp1d(original_x, segment[col].values, kind='linear', fill_value="extrapolate")
            interp_segment[col] = f(new_x)
        
        # # Convert to DataFrame and append
        interpolated_parts.append(pd.DataFrame(interp_segment))

    # Concatenate all interpolated segments
    df_upsampled = pd.concat(interpolated_parts, ignore_index=True)

    # Save or use
    # df_upsampled.to_csv(f"test_upsample.csv", index=False)
    return df_upsampled


df = pd.read_csv("/home/mdxuser/Genesis/main/data/picked_up_4/csv/5_HTP/Elastic/hard/5_HTP_Elastic_hard.csv")
steps_df = pd.read_csv("/home/mdxuser/Genesis/main/data/picked_up_4/annotations/5_HTP_Elastic_hard_annotations.csv")

if __name__ == "__main__":
    os.makedirs(os.path.join(DATA_PATH, "upsampled"), exist_ok=True)
    # for all non-empty directories in BASE_PATH/main/data/picked_up_4/csv/, extract name and deformation
    for obj_name in os.listdir(os.path.join(DATA_PATH, "csv")):
        for deformation in os.listdir(os.path.join(DATA_PATH, "csv", obj_name, MATERIAL)):
                print(f"Processing {obj_name} with deformation {deformation}")
                df = pd.read_csv(os.path.join(DATA_PATH, "csv", obj_name, MATERIAL, deformation, f"{obj_name}_{MATERIAL}_{deformation}.csv"))
                steps_df = pd.read_csv(os.path.join(DATA_PATH, "annotations", f"{obj_name}_{MATERIAL}_{deformation}_annotations.csv"))
                df_upsampled = main(df, steps_df)
                # Save the upsampled DataFrame
                output_path = os.path.join(DATA_PATH, "upsampled")
                df_upsampled.to_csv(os.path.join(output_path, f"{obj_name}_{MATERIAL}_{deformation}_upsampled.csv"), index=False)