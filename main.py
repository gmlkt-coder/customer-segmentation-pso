from __future__ import annotations

from pathlib import Path
from typing import Dict, List

try:
    from .algorithms import FitResult, run_model
    from .config import DATA_PATH, N_RUNS, RANDOM_STATE
    from .evaluation import summarize_runs
    from .preprocessing import load_dataset, preprocess_data
    from .visualization import save_3d_scatter, save_bar_chart, save_convergence_chart
except ImportError:  # pragma: no cover
    from algorithms import FitResult, run_model
    from config import DATA_PATH, N_RUNS, RANDOM_STATE
    from evaluation import summarize_runs
    from preprocessing import load_dataset, preprocess_data
from visualization import save_3d_scatter, save_bar_chart, save_convergence_chart


def writable_output_path(path: Path) -> Path:
    for idx in range(100):
        candidate = path if idx == 0 else path.with_name(f"{path.stem}_{idx}{path.suffix}")
        try:
            with candidate.open("a+b"):
                pass
            return candidate
        except PermissionError:
            continue
    raise PermissionError(f"No writable output path available for {path.name}. Close locked output files and try again.")


def run_experiment() -> None:
    output_dir = Path(__file__).resolve().parent
    df = load_dataset(DATA_PATH)
    df_selected, entropy_df, data_3d_df = preprocess_data(df, entropy_threshold=0.05)
    data_3d = data_3d_df.to_numpy(dtype=float)

    print("Entropy ranking:")
    print(entropy_df.to_string(index=False))
    print("\nSelected features after entropy filtering:")
    print(df_selected.columns.tolist())
    print("\nPCA shape:", data_3d.shape)

    model_names = ["KMeans-Random", "KMeans++", "PSO-KM", "IHPSO-KM", "MOPSO-KM"]
    all_metrics: Dict[str, List[Dict[str, float]]] = {name: [] for name in model_names}
    all_histories: Dict[str, List[List[float]]] = {name: [] for name in ["PSO-KM", "IHPSO-KM", "MOPSO-KM"]}
    representative: Dict[str, FitResult] = {}

    for run_id in range(N_RUNS):
        seed = RANDOM_STATE + run_id
        for name in model_names:
            result_metrics, fit_result = run_model(name, data_3d, seed)
            all_metrics[name].append(result_metrics)
            if name in all_histories:
                all_histories[name].append(fit_result.history)
            if run_id == 0:
                representative[name] = fit_result

    summary = summarize_runs(all_metrics).sort_values("Silhouette_Mean", ascending=False).reset_index(drop=True)

    print("\nFinal summary over 10 runs:")
    print(summary.to_string(index=False))
    summary_path = writable_output_path(output_dir / "clustering_summary.csv")
    bar_chart_path = writable_output_path(output_dir / "silhouette_dbi_comparison.png")
    convergence_path = writable_output_path(output_dir / "convergence_curves.png")
    scatter_path = writable_output_path(output_dir / "best_model_3d_scatter.png")

    summary.to_csv(summary_path, index=False)
    save_bar_chart(summary, bar_chart_path)
    save_convergence_chart(all_histories, convergence_path)

    best_model = summary.iloc[0]["Model"]
    best_result = representative.get(best_model)
    if best_result is None:
        _, best_result = run_model(best_model, data_3d, RANDOM_STATE)
    save_3d_scatter(data_3d, best_result.labels, scatter_path, f"3D Clustering Result - {best_model}")

    print("\nSaved files:")
    print(f"- {summary_path}")
    print(f"- {bar_chart_path}")
    print(f"- {convergence_path}")
    print(f"- {scatter_path}")


if __name__ == "__main__":
    try:
        run_experiment()
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        print(f"Error: {exc}")
