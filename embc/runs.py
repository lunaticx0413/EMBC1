"""Run bookkeeping: configuration fingerprints, completed-run validation, training batches.

A run directory is complete only when metrics.json exists. It is written last, and it records the
configuration that produced the run. Rerunning with a different configuration in the same directory is
refused instead of silently reusing the old outputs.
"""
import hashlib
import json
from pathlib import Path

# arguments that determine a run's results (device and paths do not)
RUN_KEYS = ("exp", "fold", "subject_fold", "k_subject_folds", "cond", "seed", "epochs", "patience", "batch", "lr",
            "wd", "tau", "select_gallery", "max_train_events")


class RunConfigMismatch(RuntimeError):
    pass


def file_digest(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:16]


def run_config(args, data_files):
    """args: dict of parsed arguments; data_files: input files whose content defines the run."""
    cfg = {k: args.get(k) for k in RUN_KEYS}
    cfg["data"] = {Path(p).name: file_digest(p) for p in sorted(map(str, data_files))}
    return cfg


def existing_run_state(out, config):
    """'missing' if no completed run exists in out, 'done' if one exists with this exact config.
    Raises RunConfigMismatch if a completed run was produced by a different configuration."""
    f = Path(out) / "metrics.json"
    if not f.exists():
        return "missing"
    old = json.loads(f.read_text()).get("config")
    if old == config:
        return "done"
    if old is None:
        raise RunConfigMismatch(f"{f} has no recorded configuration; remove it or pass --overwrite")
    diff = sorted(k for k in set(old) | set(config) if old.get(k) != config.get(k))
    raise RunConfigMismatch(f"{f} was produced by a different configuration (differs in: {', '.join(diff)}); "
                            "use another --exp or pass --overwrite")


def batch_slices(n, batch, min_size=2):
    """Training batches over n shuffled events, keeping a final partial batch when it has >= min_size events
    (BatchNorm needs more than one sample per batch in training mode)."""
    out = [(lo, min(lo + batch, n)) for lo in range(0, n, batch)]
    return [(lo, hi) for lo, hi in out if hi - lo >= min_size]
