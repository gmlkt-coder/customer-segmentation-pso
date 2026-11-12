from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.environ.get("CUSTOMER_CLUSTERING_DATA_PATH", BASE_DIR / "marketing_campaign_clean.csv"))

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

N_CLUSTERS = 3
N_RUNS = 10
MAX_ITER = 80
N_PARTICLES = 30
RANDOM_STATE = 42
