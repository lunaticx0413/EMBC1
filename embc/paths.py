"""Filesystem locations. Every external location can be overridden by an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DATA = Path(os.environ.get("EMBC_DATA", ROOT / "data"))          # derived data (gitignored)
RUNS = Path(os.environ.get("EMBC_RUNS", ROOT / "runs"))          # trained models, per-event outputs (gitignored)
RESULTS = Path(os.environ.get("EMBC_RESULTS", ROOT / "results"))   # small summary tables (versioned)

ZUCO2_NR = Path(os.environ.get(
    "EMBC_ZUCO2_NR", "/run/media/lunaticx/Documents/eeg2text/datasets/ZuCo/v2/task2-NR-2.0/Matlab files"))
MPNET = Path(os.environ.get(
    "EMBC_MPNET", "/run/media/lunaticx/Documents/hf_home/hub/models--sentence-transformers--all-mpnet-base-v2/"
                  "snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130"))
NLTK_DATA = Path(os.environ.get("EMBC_NLTK_DATA", ROOT / ".cache" / "nltk_data"))

CHANPOS = ASSETS / "zuco2_chanpos.csv"                            # 105 channels in rawData order; Cz is constant
SPLITS = ASSETS / "splits"                                        # lobo7 contiguous-block text folds (copied)


def continuous_path(subject):
    return DATA / "continuous" / f"{subject}.npz"


def fold_dir(fold):
    return DATA / f"fold{fold}"
