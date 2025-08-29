import pandas as pd

data_dir = "/home/user/Genesis/data/YCB_0824/"

df = pd.read_csv(data_dir + "train.csv")

# Ensure 'label' column exists in the DataFrame
df['label'] = None

for idx, row in df.iterrows():
    annotation = row["annotation"]

    if "place" in annotation:
        df.at[idx, "label"] = "Places the object."
    elif "grasp" in annotation:
        df.at[idx, "label"] = "Grabs the object."
    elif "touch" in annotation:
        df.at[idx, "label"] = "Touches the object."
    elif "drop" in annotation:
        df.at[idx, "label"] = "Drops the object."
    elif ", no slip" in annotation:
        df.at[idx, 'label'] = "Holds the object."
    elif "slip" in annotation:
        if "quickly" in annotation:
            df.at[idx, "label"] = "The object slips quickly."
        else:
            df.at[idx, "label"] = "The object slips slowly."
    elif "empty" in annotation:
        df.at[idx, "label"] = "No contact."
    else:
        df.at[idx, "label"] = "Holds the object."

df.to_csv(data_dir + "train.csv", index=False)