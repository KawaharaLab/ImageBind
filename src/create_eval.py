import pandas as pd

# Load your large CSV file
df = pd.read_csv("/home/mdxuser/Genesis/main/data/picked_up/formatted_training_data_IB.csv")

# Sample 1000 random rows without replacement
sampled_df = df.sample(n=2000, random_state=42)

# Write the sampled rows to a new CSV file
sampled_df.to_csv("/home/mdxuser/Genesis/main/data/picked_up/eval.csv", index=False)
