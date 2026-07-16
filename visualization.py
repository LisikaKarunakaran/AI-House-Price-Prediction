"""
============================================================
AI House Price Prediction System
Module : Exploratory Data Analysis (EDA)
Author : Lisika K
============================================================
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "dataset" / "clean_house_data.csv"

IMAGE_FOLDER = PROJECT_ROOT / "images"

IMAGE_FOLDER.mkdir(exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    if not DATA_PATH.exists():
        print("❌ Clean dataset not found!")
        print("Run preprocessing.py first.")
        return None

    df = pd.read_csv(DATA_PATH)

    print("✅ Dataset Loaded Successfully!")

    return df

# ==========================================================
# Correlation Heatmap
# ==========================================================

def correlation_heatmap(df):

    plt.figure(figsize=(12, 8))

    correlation = df.select_dtypes(include="number").corr()

    sns.heatmap(
        correlation,
        annot=False,
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        IMAGE_FOLDER / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Price Distribution
# ==========================================================

def price_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["price"],
        bins=40,
        kde=True
    )

    plt.title("House Price Distribution")

    plt.xlabel("Price")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        IMAGE_FOLDER / "price_distribution.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Bedrooms Count
# ==========================================================

def bedroom_count(df):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        x="bedrooms",
        data=df
    )

    plt.title("Bedroom Count")

    plt.xlabel("Bedrooms")

    plt.ylabel("Count")

    plt.tight_layout()

    plt.savefig(
        IMAGE_FOLDER / "bedroom_count.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Living Area vs Price
# ==========================================================

def living_area_vs_price(df):

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        x="sqft_living",
        y="price",
        data=df
    )

    plt.title("Living Area vs House Price")

    plt.xlabel("Living Area (sqft)")

    plt.ylabel("Price")

    plt.tight_layout()

    plt.savefig(
        IMAGE_FOLDER / "living_area_price.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Price Boxplot
# ==========================================================

def price_boxplot(df):

    plt.figure(figsize=(8, 4))

    sns.boxplot(
        x=df["price"]
    )

    plt.title("House Price Boxplot")

    plt.tight_layout()

    plt.savefig(
        IMAGE_FOLDER / "price_boxplot.png",
        dpi=300
    )

    plt.close()

# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 60)
    print("AI HOUSE PRICE PREDICTION SYSTEM")
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    data = load_dataset()

    if data is None:
        return

    print("\nGenerating Charts...\n")

    correlation_heatmap(data)
    print("✅ correlation_heatmap.png")

    price_distribution(data)
    print("✅ price_distribution.png")

    bedroom_count(data)
    print("✅ bedroom_count.png")

    living_area_vs_price(data)
    print("✅ living_area_price.png")

    price_boxplot(data)
    print("✅ price_boxplot.png")

    print("\n==========================================")
    print("EDA Completed Successfully!")
    print(f"Images saved in:\n{IMAGE_FOLDER}")
    print("==========================================")

# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()








































































































