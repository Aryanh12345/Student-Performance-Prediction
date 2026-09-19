from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_PATH = ROOT / "models" / "student_score_model.pkl"

FEATURES = ["Age", "Study_Hours", "Attendance(%)"]
TARGET = "Test_Score"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    # Treat physically impossible values as missing so the model can impute them.
    df.loc[~df["Study_Hours"].between(0, 24), "Study_Hours"] = np.nan
    df.loc[~df["Attendance(%)"].between(0, 100), "Attendance(%)"] = np.nan
    return df


def train():
    df = clean_data(pd.read_csv(DATA_PATH))
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )),
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"R2:   {r2_score(y_test, pred):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, pred):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, pred) ** 0.5:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
