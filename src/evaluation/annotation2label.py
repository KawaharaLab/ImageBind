import pandas as pd

data_dir = "/home/mdxuser/ImageBind/data/YCB_0824/"

df = pd.read_csv(data_dir + "eval.csv")

# Ensure 'label' column exists in the DataFrame
df['label'] = None

for idx, row in df.iterrows():
    annotation = row["annotation"]

    if "place" in annotation:
        df.at[idx, "label"] = "place"
    elif "grasp" in annotation:
        df.at[idx, "label"] = "grasp"
    elif "drop" in annotation:
        df.at[idx, "label"] = "drop"
    elif ", no slip" in annotation:
        df.at[idx, 'label'] = "no slip"
    elif "slip" in annotation:
        if "quickly" in annotation:
            df.at[idx, "label"] = "slip quickly"
        else:
            df.at[idx, "label"] = "slip slowly"
    elif "empty" in annotation:
        df.at[idx, "label"] = "empty"
    else:
        df.at[idx, "label"] = "hold"

df.to_csv(data_dir + "eval.csv", index=False)