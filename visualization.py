from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_bar_chart(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Silhouette Score and DBI Comparison")

    panels = [
        ("Silhouette", "Silhouette_Mean", "Silhouette_Std", "#2a9d8f"),
        ("DBI", "DBI_Mean", "DBI_Std", "#e76f51"),
    ]

    x = np.arange(len(summary))
    for ax, (title, mean_col, std_col, color) in zip(axes, panels):
        ax.bar(x, summary[mean_col], yerr=summary[std_col].fillna(0), color=color, alpha=0.9, capsize=4)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["Model"], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_convergence_chart(curves: Dict[str, List[List[float]]], output_path: Path) -> None:
    if not curves:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"PSO-KM": "#3498db", "IHPSO-KM": "#2ecc71", "MOPSO-KM": "#e74c3c"}

    for model_name, histories in curves.items():
        if not histories:
            continue
        max_len = max(len(h) for h in histories)
        aligned = []
        for hist in histories:
            arr = np.asarray(hist, dtype=float)
            if len(arr) < max_len:
                arr = np.pad(arr, (0, max_len - len(arr)), mode="edge")
            aligned.append(arr)
        mean_curve = np.mean(np.vstack(aligned), axis=0)
        ax.plot(mean_curve, label=model_name, color=colors.get(model_name))

    ax.set_title("Convergence Curves")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best SSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_3d_scatter(data_3d: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap="tab10", s=18, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(scatter, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

