"""
============================================================
AI House Price Prediction System
Module : Feature Engineering
Author : Lisika K
============================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "dataset" / "clean_house_data.csv"

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():
    """Load the cleaned dataset."""

    if not DATA_PATH.exists():
        print("❌ Clean dataset not found!")
        print("Please run preprocessing.py first.")
        return None

    df = pd.read_csv(DATA_PATH)

    print("✅ Clean Dataset Loaded Successfully!")

    return df


# ==========================================================
# Feature Engineering
# ==========================================================

def prepare_features(df):
    """
    Remove unnecessary columns and prepare
    features (X) and target (y).
    """

    columns_to_drop = [
        "date",
        "street",
        "city",
        "statezip",
        "country"
    ]

    # Remove columns only if they exist
    existing_columns = [col for col in columns_to_drop if col in df.columns]

    df = df.drop(columns=existing_columns)

    # Target variable
    y = df["price"]

    # Input features
    X = df.drop("price", axis=1)

    print("\n✅ Feature Selection Completed!")

    print("\nSelected Features:")
    for feature in X.columns:
        print(f"• {feature}")

    return X, y


# ==========================================================
# Train-Test Split
# ==========================================================

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\n========================================")
    print("TRAIN TEST SPLIT")
    print("========================================")

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    print(f"\nTraining Shape : {X_train.shape}")
    print(f"Testing Shape  : {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 60)
    print("AI HOUSE PRICE PREDICTION SYSTEM")
    print("FEATURE ENGINEERING")
    print("=" * 60)

    data = load_dataset()

    if data is None:
        return

    X, y = prepare_features(data)

    split_dataset(X, y)

    print("\n========================================")
    print("✅ Feature Engineering Completed!")
    print("========================================")


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()