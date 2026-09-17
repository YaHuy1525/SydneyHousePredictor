import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from src.features import build_model_pipeline, engineer_features, LOG_TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES

DATA_PATH = Path("data/processed_dataset.csv")
MODEL_PATH = Path("models/best_model.joblib")


def train_and_save():
    df = pd.read_csv(DATA_PATH, parse_dates=["sale_date"])
    df = engineer_features(df)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df[LOG_TARGET]  # train on log scale

    pipeline = build_model_pipeline(
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
        scale_numerics=False,  # GBR doesn't need scaling
    )
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
