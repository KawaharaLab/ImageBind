import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_val(data_path, output_dir, test_size=0.1, random_state=42):
    """
    Splits a dataset into training and validation sets.

    :param data_path: Path to the input CSV file containing the dataset.
    :param output_dir: Directory where the train and val CSV files will be saved.
    :param test_size: Proportion of the dataset to include in the validation split (default: 0.1).
    :param random_state: Random seed for reproducibility (default: 42).
    """
    # Load the dataset
    data = pd.read_csv(data_path)

    # Split the dataset into training and validation sets
    train_data, val_data = train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the splits to CSV files
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "eval.csv")
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)

    print(f"Training data saved to: {train_path}")
    print(f"Validation data saved to: {val_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument("data_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the train and val CSV files.")
    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Proportion of data for validation (default: 0.1)."
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed for reproducibility (default: 42)."
    )

    args = parser.parse_args()
    split_train_val(
        data_path=args.data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
