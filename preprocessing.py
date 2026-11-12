from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from .config import RANDOM_STATE
except ImportError:  # pragma: no cover
    from config import RANDOM_STATE


def load_dataset(path) -> pd.DataFrame:
    path = str(path)
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Dataset not found. Put marketing_campaign.csv in the project folder "
            "or set CUSTOMER_CLUSTERING_DATA_PATH to the CSV file path."
        ) from exc
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    return df


def entropy_from_series(series: pd.Series, bins: int = 10) -> float:
    clean = series.dropna()
    if clean.empty or clean.nunique() <= 1:
        return 0.0

    if pd.api.types.is_numeric_dtype(clean):
        try:
            binned = pd.qcut(clean, q=min(bins, clean.nunique()), duplicates="drop")
            probs = binned.value_counts(normalize=True).values
        except Exception:
            probs = clean.value_counts(normalize=True).values
    else:
        probs = clean.astype(str).value_counts(normalize=True).values

    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    return float(scipy_entropy(probs, base=2))


def iqr_filter(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)
    return df.loc[mask].reset_index(drop=True)


def standard_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    return scaled, scaler.mean_, scaler.scale_


def pca_reduce(X: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(X)
    return reduced, pca.components_


def preprocess_data(
    df: pd.DataFrame,
    entropy_threshold: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(df["Income"].median())

    spend_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]

    if "Year_Birth" in df.columns and "Age" not in df.columns:
        df["Age"] = 2014 - df["Year_Birth"]

    if all(col in df.columns for col in spend_cols):
        df["Total_Spend"] = df[spend_cols].sum(axis=1)
        for col in spend_cols:
            pct_col = f"Pct_{col.replace('Mnt', '')}"
            df[pct_col] = np.where(df["Total_Spend"] > 0, df[col] / df["Total_Spend"], 0.0)

    if "Dt_Customer" in df.columns:
        date_col = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
        ref_date = date_col.max()
        if pd.isna(ref_date):
            ref_date = pd.Timestamp("2014-12-31")
        date_col = date_col.fillna(ref_date)
        df["Customer_Days"] = (ref_date - date_col).dt.days.astype(float)

    if "Education" in df.columns:
        edu_map = {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
        df["Education"] = df["Education"].map(edu_map).fillna(2).astype(int)

    if "Marital_Status" in df.columns:
        single_like = {"Single", "Divorced", "Widow", "Alone", "YOLO", "Absurd"}
        df["Marital_Single"] = df["Marital_Status"].apply(lambda x: 1 if x in single_like else 0).astype(int)

    drop_cols = ["ID", "Year_Birth", "Dt_Customer", "Marital_Status", "Z_CostContact", "Z_Revenue"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        raise ValueError("Dataset must contain at least 3 numeric features for PCA clustering.")

    df = df[numeric_cols].copy()
    df = df.fillna(df.median(numeric_only=True))

    filter_cols = [col for col in ["Income", "Age", "Total_Spend", "Total_Spending"] if col in df.columns]
    if filter_cols:
        df = iqr_filter(df, filter_cols)

    entropy_scores = {col: entropy_from_series(df[col]) for col in df.columns}
    entropy_df = (
        pd.DataFrame({"feature": list(entropy_scores.keys()), "entropy": list(entropy_scores.values())})
        .sort_values("entropy", ascending=False)
        .reset_index(drop=True)
    )

    selected = [col for col, ent in entropy_scores.items() if ent > entropy_threshold]
    if len(selected) < 3:
        selected = entropy_df["feature"].head(3).tolist()

    df_selected = df[selected].copy()
    scaled, _, _ = standard_scale(df_selected.to_numpy(dtype=float))
    reduced, _ = pca_reduce(scaled, n_components=3)
    reduced_df = pd.DataFrame(reduced, columns=["PC1", "PC2", "PC3"])
    return df_selected, entropy_df, reduced_df
