"""Train a RandomForest regressor to predict Sales_INR for the Superstore dataset.

Requirements:
- Use a relative path to Dataset/Superstore.csv
- Convert Sales to INR using Sales * 83
- Save the trained model as models/sales_model.pkl
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "Dataset" / "Superstore.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # Read CSV using relative project path only
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Convert Sales to INR
    df["Sales_INR"] = df["Sales"] * 83

    # Use only columns that actually exist in the dataset for a robust model build.
    possible_feature_columns = [
        "Category",
        "Sub-Category",
        "Region",
        "Segment",
        "Ship Mode",
        "State",
    ]
    feature_columns = [
        col for col in possible_feature_columns
        if col in df.columns and df[col].nunique() <= 200
    ]

    if not feature_columns:
        feature_columns = ["Category", "Region", "Segment"]

    # Prepare features
    model_df = df[feature_columns + ["Sales_INR"]].copy()
    for col in feature_columns:
        model_df[col] = model_df[col].astype(str)

    model_df = pd.get_dummies(model_df, columns=feature_columns)

    X = model_df.drop(columns=["Sales_INR"])
    y = model_df["Sales_INR"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=None,
        min_samples_leaf=1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"Model RMSE on INR sales: ₹{rmse:,.2f}")
    print(f"Model R^2 score: {r2:.4f}")

    model_dir = Path(__file__).resolve().parent
    model_path = model_dir / "sales_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
