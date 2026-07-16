"""
============================================================
AI House Price Prediction System
Module : Data Preprocessing
Author : Lisika K
============================================================
"""

from pathlib import Path
import pandas as pd

# ==========================================================
# Project Paths
# ==========================================================

# Current file -> src/preprocessing.py
CURRENT_FILE = Path(__file__).resolve()

# Project root -> House_Price_Prediction
PROJECT_ROOT = CURRENT_FILE.parent.parent

# Dataset folder
DATASET_FOLDER = PROJECT_ROOT / "dataset"

# Dataset files
DATA_PATH = DATASET_FOLDER / "house_data.csv"
CLEAN_DATA_PATH = DATASET_FOLDER / "clean_house_data.csv"


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    print(f"\nDataset Path : {DATA_PATH}")

    if not DATA_PATH.exists():
        print("\n❌ ERROR : house_data.csv not found!")
        print("\nPlease place the dataset here:")
        print(DATA_PATH)
        return None

    df = pd.read_csv(DATA_PATH)

    print("\n✅ Dataset Loaded Successfully!")

    return df


# ==========================================================
# Dataset Information
# ==========================================================

def dataset_info(df):

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names")
    print("-" * 60)
    print(df.columns.tolist())

    print("\nData Types")
    print("-" * 60)
    print(df.dtypes)


# ==========================================================
# Display First Rows
# ==========================================================

def display_data(df):

    print("\n" + "=" * 60)
    print("FIRST FIVE RECORDS")
    print("=" * 60)

    print(df.head())


# ==========================================================
# Missing Values
# ==========================================================

def check_missing_values(df):

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()

    print(missing)

    print(f"\nTotal Missing Values : {missing.sum()}")


# ==========================================================
# Duplicate Records
# ==========================================================

def check_duplicates(df):

    print("\n" + "=" * 60)
    print("DUPLICATE RECORDS")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(f"Duplicate Records : {duplicates}")


# ==========================================================
# Statistical Summary
# ==========================================================

def statistical_summary(df):

    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("=" * 60)

    print(df.describe())


# ==========================================================
# Remove Duplicate Records
# ==========================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"\nDuplicate Records Removed : {before-after}")

    return df


# ==========================================================
# Save Clean Dataset
# ==========================================================

def save_dataset(df):

    DATASET_FOLDER.mkdir(exist_ok=True)

    df.to_csv(CLEAN_DATA_PATH, index=False)

    print("\n✅ Clean Dataset Saved Successfully!")
    print(f"Location : {CLEAN_DATA_PATH}")


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 60)
    print("AI HOUSE PRICE PREDICTION SYSTEM")
    print("DATA PREPROCESSING MODULE")
    print("=" * 60)

    data = load_dataset()

    if data is None:
        return

    dataset_info(data)

    display_data(data)

    check_missing_values(data)

    check_duplicates(data)

    statistical_summary(data)

    data = remove_duplicates(data)

    save_dataset(data)

    print("\n🎉 Data Preprocessing Completed Successfully!")


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()