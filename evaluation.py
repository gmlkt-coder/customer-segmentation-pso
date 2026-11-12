from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def compute_sse(data: np.ndarray, centers: np.ndarray) -> float:
    distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
    return float(np.min(distances**2, axis=1).sum())


def evaluate_clustering(labels: np.ndarray, data: np.ndarray) -> Dict[str, float]:
    unique = np.unique(labels)
    if len(unique) < 2:
        return {"Inertia": np.nan, "Silhouette": np.nan, "DBI": np.nan, "CHI": np.nan}

    centers = np.vstack([data[labels == lab].mean(axis=0) for lab in unique])
    return {
        "Inertia": compute_sse(data, centers),
        "Silhouette": float(silhouette_score(data, labels)),
        "DBI": float(davies_bouldin_score(data, labels)),
        "CHI": float(calinski_harabasz_score(data, labels)),
    }


def summarize_runs(results: Dict[str, List[Dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for model_name, metrics_list in results.items():
        frame = pd.DataFrame(metrics_list)
        row = {"Model": model_name}
        for metric in ["Inertia", "Silhouette", "DBI", "CHI"]:
            row[f"{metric}_Mean"] = frame[metric].mean()
            row[f"{metric}_Std"] = frame[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)

