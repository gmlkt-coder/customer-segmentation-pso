from __future__ import annotations

from main import run_experiment


if __name__ == "__main__":
    try:
        run_experiment()
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        print(f"Error: {exc}")
