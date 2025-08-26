import os
import pandas as pd

BASE_PATH = '/home/mdxuser/Genesis/main/data/picked_up_4'
ANNOTATION_PATH = f'{BASE_PATH}/annotations'
CSV_BASE_DIR = f'{BASE_PATH}/csv'
OUTPUT_CSV = f"/home/mdxuser/ImageBind/src/train_upsampled_simple.csv"

format1_rows = []

for filename in os.listdir(ANNOTATION_PATH):
    if filename.endswith(".csv"):
        annotation_filepath = os.path.join(ANNOTATION_PATH, filename)
        df = pd.read_csv(annotation_filepath)

        # Extract object, grasp type, and strength from filename
        # Format: Object_GraspType_Strength_annotations.csv
        base_name = filename.replace("_annotations.csv", "")
        parts = base_name.split("_")
        
        # Extract object name (everything before the last two components)
        obj_name = "_".join(parts[:-2])
        grasp_type = parts[-2]
        strength = parts[-1]

        # Construct CSV path
        csv_file = f"{obj_name}_{grasp_type}_{strength}.csv"
        full_csv_path = os.path.join(CSV_BASE_DIR, obj_name, grasp_type, strength, csv_file)

        for _, row in df.iterrows():
            timestep_start = row["step start"]
            annotation = row["annotation"]

            format1_rows.append({
                "csv_path": full_csv_path,
                "timestep_start": _ * 270,
                "annotation": annotation
            })

# Save to CSV
format1_df = pd.DataFrame(format1_rows)
format1_df.to_csv(OUTPUT_CSV, index=False)

print(f"Converted {len(format1_rows)} annotations to {OUTPUT_CSV}")
