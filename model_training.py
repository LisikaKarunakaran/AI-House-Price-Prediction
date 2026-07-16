"""
============================================================
AI House Price Prediction System
Module : Model Training & Evaluation
Author : Lisika K
============================================================
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "dataset" / "clean_house_data.csv"

MODEL_FOLDER = PROJECT_ROOT / "models"
MODEL_FOLDER.mkdir(exist_ok=True)

# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset():

    if not DATA_PATH.exists():
        print("❌ Clean dataset not found!")
        return None

    df = pd.read_csv(DATA_PATH)

    print("✅ Dataset Loaded Successfully!")

    return df


# ==========================================================
# Prepare Features
# ==========================================================

def prepare_data(df):

    columns_to_drop = [
        "date",
        "street",
        "city",
        "statezip",
        "country"
    ]

    existing_columns = [col for col in columns_to_drop if col in df.columns]

    df = df.drop(columns=existing_columns)

    X = df.drop("price", axis=1)

    y = df["price"]

    print("\nTraining Features")
    print("-" * 50)
    print(X.columns.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = mse ** 0.5

    return r2, mae, mse, rmse


# ==========================================================
# Main Function
# ==========================================================

def main():

    print("=" * 60)
    print("AI HOUSE PRICE PREDICTION SYSTEM")
    print("MODEL TRAINING")
    print("=" * 60)

    data = load_dataset()

    if data is None:
        return

    X_train, X_test, y_train, y_test, feature_names = prepare_data(data)

    models = {

        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        )
    }

    best_model = None
    best_name = ""
    best_score = float("-inf")

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    for name, model in models.items():

        model.fit(X_train, y_train)

        r2, mae, mse, rmse = evaluate(
            model,
            X_test,
            y_test
        )

        print(f"\n{name}")
        print(f"R² Score : {r2:.4f}")
        print(f"MAE      : {mae:.2f}")
        print(f"MSE      : {mse:.2f}")
        print(f"RMSE     : {rmse:.2f}")

        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name

    # Save Best Model
    joblib.dump(
        best_model,
        MODEL_FOLDER / "best_model.pkl"
    )

    # Save Feature Names
    joblib.dump(
        feature_names,
        MODEL_FOLDER / "feature_names.pkl"
    )

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print(f"Model Name : {best_name}")
    print(f"R² Score   : {best_score:.4f}")

    print("\nFiles Saved")

    print("✔ best_model.pkl")
    print("✔ feature_names.pkl")

    print("\nLocation")

    print(MODEL_FOLDER)


# ==========================================================
# Driver Code
# ==========================================================

if __name__ == "__main__":
    main()