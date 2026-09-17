import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ── Column groups ──────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "car_spaces",
    "land_size_m2",
    "floor_area_m2",
    "property_age",
    "distance_to_cbd_km",
    "bed_bath_ratio",
]

CATEGORICAL_FEATURES = ["suburb", "property_type"]

TARGET = "sale_price_aud"
LOG_TARGET = "log_sale_price"


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, reference_year: int = 2024) -> pd.DataFrame:
    df = df.copy()

    # Property age (years since built)
    df["property_age"] = reference_year - df["year_built"]

    # Bedroom-to-bathroom ratio (proxy for dwelling configuration)
    df["bed_bath_ratio"] = (
        df["bedrooms"] / df["bathrooms"].replace(0, np.nan)
    )

    # Price per m² — only available when land_size_m2 is present
    if "sale_price_aud" in df.columns:
        df["price_per_m2"] = np.where(
            df["land_size_m2"].notna(),
            df["sale_price_aud"] / df["land_size_m2"],
            np.nan,
        )

    # Log-transform the target (fitted models use log scale internally)
    if "sale_price_aud" in df.columns:
        df[LOG_TARGET] = np.log(df["sale_price_aud"])

    return df


# ── Preprocessing pipeline factory ────────────────────────────────────────────

def build_preprocessor(
    numeric_features: list = None,
    categorical_features: list = None,
    scale_numerics: bool = True,
) -> ColumnTransformer:
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numerics:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipeline(model, scale_numerics: bool = True) -> Pipeline:
    preprocessor = build_preprocessor(scale_numerics=scale_numerics)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ── Metric helpers ─────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
