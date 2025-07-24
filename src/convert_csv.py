import os
import pandas as pd

ANNOTATION_PATH = '/Users/hh/Desktop/genesis/genesis_forked/Genesis/main/data/picked_up_5/annotations'
CSV_BASE_DIR = '/Users/hh/Desktop/genesis/genesis_forked/Genesis/main/data/picked_up_5/csv'
OUTPUT_CSV = "/Users/hh/Desktop/ImageBind/ImageBind/data/train.csv"

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
                "timestep_start": timestep_start,
                "annotation": annotation
            })

# Save to CSV
format1_df = pd.DataFrame(format1_rows)
format1_df.to_csv(OUTPUT_CSV, index=False)

print(f"Converted {len(format1_rows)} annotations to {OUTPUT_CSV}")
