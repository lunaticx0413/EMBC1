"""Bookkeeping: atomic writes, run configuration fingerprints, completed-run validation, training batches.

A run directory is complete only when metrics.json exists. It is written last, atomically, and records the
configuration that produced the run: the arguments that determine its results plus content digests of
every input (fold data, reader folds, the stage-1a EEG of EEG conditions) and of the code that trains and
evaluates. A completed run whose configuration differs from the requested one is refused instead of
silently reused, and the report refuses runs whose recorded inputs or code no longer match the current ones.
"""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path

from . import events, paths
from .inputs import EEG_CONDITIONS

# arguments that determine a run's results (device and paths do not)
RUN_KEYS = ("exp", "fold", "subject_fold", "k_subject_folds", "cond", "seed", "epochs", "patience", "batch", "lr",
            "wd", "tau", "select_gallery", "max_train_events")
GRID_KEYS = ("subject_fold", "cond", "seed")          # vary within one experiment; the other keys are its protocol
# code that turns the inputs into a run's outputs (stage-1a code is covered by the stage-1a data digests)
CODE_FILES = ("embc/events.py", "embc/inputs.py", "embc/lexicon.py", "embc/metrics.py", "embc/model.py",
              "embc/runs.py", "scripts/train_eval.py")


class RunConfigMismatch(RuntimeError):
    pass


def write_atomic(path, text):
    """Write text through a temporary file in the same directory, so path never holds a partial file."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@functools.lru_cache(maxsize=None)
def _digest(path, ino, mtime_ns, size, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:16]


def file_digest(path):
    """sha256[:16] of a file's content, cached per process while the file is unchanged."""
    st = os.stat(path)
    return _digest(str(path), st.st_ino, st.st_mtime_ns, st.st_size)


def continuous_digest(subject):
    """Content digest of a reader's stage-1a npz. build_continuous.py records it in <SUBJ>_info.json;
    for readers built before it did, it is computed once here and recorded there."""
    out = paths.subject_outputs(subject)
    info = json.loads(out["info"].read_text())
    if "npz_sha256" not in info:
        info["npz_sha256"] = file_digest(out["npz"])
        write_atomic(out["info"], json.dumps(info, indent=1))
    return info["npz_sha256"]


def input_digests(fold, k_subject_folds, cond):
    """Digests of everything besides the arguments that determines a run's results: {"data": ..., "code": ...}."""
    fdir = paths.fold_dir(fold)
    files = [fdir / "events.csv", fdir / "types.npz", *sorted(fdir.glob("galleries_*.npz")),
             paths.DATA / f"subject_folds_k{k_subject_folds}.json"]
    data = {f.name: file_digest(f) for f in files}
    if cond in EEG_CONDITIONS:
        data.update({f"continuous/{s}.npz": continuous_digest(s) for s in events.subjects_available()})
    return dict(data=data, code={p: file_digest(paths.ROOT / p) for p in CODE_FILES})


def run_config(args, inputs):
    """args: dict of parsed train_eval arguments; inputs: input_digests(...) of the run."""
    cfg = {k: args.get(k) for k in RUN_KEYS}
    cfg.update(inputs)
    return cfg


def protocol(cfg):
    """The part of a run configuration that every run of one experiment shares."""
    return {k: cfg.get(k) for k in RUN_KEYS if k not in GRID_KEYS}


def run_dir(a):
    return paths.RUNS / a.exp / f"fold{a.fold}" / f"sf{a.subject_fold}" / a.cond / f"seed{a.seed}"


def read_metrics(out):
    """metrics.json of a completed run, or None if there is none. Raises RunConfigMismatch if unreadable."""
    f = Path(out) / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as err:
        raise RunConfigMismatch(f"{f} is unreadable ({err}); remove it or pass --overwrite") from None


def config_diff(old, new):
    """Names of the entries that differ; digest tables are compared per file ("data:events.csv")."""
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"{k}:{f}" for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]
        elif a != b:
            out.append(k)
    return out


def existing_run_state(out, config):
    """'missing' if no completed run exists in out, 'done' if one exists with this exact config.
    Raises RunConfigMismatch if a completed run was produced by a different configuration."""
    m = read_metrics(out)
    if m is None:
        return "missing"
    old = m.get("config")
    if old is None:
        raise RunConfigMismatch(f"{Path(out) / 'metrics.json'} has no recorded configuration; remove it or pass --overwrite")
    diff = config_diff(old, config)
    if not diff:
        return "done"
    raise RunConfigMismatch(f"{Path(out) / 'metrics.json'} was produced by a different configuration (differs in: "
                            f"{', '.join(diff)}); use another --exp or pass --overwrite")


def stale_reason(out):
    """None if out holds a completed run whose recorded inputs and code match the current ones, else why not."""
    try:
        m = read_metrics(out)
    except RunConfigMismatch as err:
        return str(err)
    if m is None:
        return "not completed"
    cfg = m.get("config")
    if not cfg:
        return "no recorded configuration"
    cur = input_digests(cfg["fold"], cfg["k_subject_folds"], cfg["cond"])
    diff = config_diff({k: cfg.get(k) for k in cur}, cur)
    return f"inputs or code changed since training (differs in: {', '.join(diff)})" if diff else None


def train_parser():
    """Arguments of scripts/train_eval.py; scripts/run_status.py parses the same ones to get the same defaults."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--subject_fold", type=int, default=0)
    ap.add_argument("--k_subject_folds", type=int, default=3)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--select_gallery", default="random_N100")
    ap.add_argument("--max_train_events", type=int, default=None, help="smoke tests only")
    ap.add_argument("--exp", default="main", help="experiment name; separates e.g. smoke runs from main runs")
    ap.add_argument("--overwrite", action="store_true", help="replace a completed run with another config")
    ap.add_argument("--device", default="cuda")
    return ap


def selection_gallery(galleries, name):
    """(rows, G) of the model-selection gallery among the validation galleries. Refuses an unknown gallery or
    one that covers no validation event (its MRR would be NaN and no epoch could ever be selected)."""
    if name not in galleries:
        raise SystemExit(f"unknown --select_gallery {name}; available: {', '.join(sorted(galleries))}")
    rows, G = galleries[name]
    if len(rows) == 0:
        raise SystemExit(f"selection gallery {name} covers no validation event")
    return rows, G


def batch_slices(n, batch, min_size=2):
    """Training batches over n shuffled events, keeping a final partial batch when it has >= min_size events
    (a 1-event batch has a single in-batch candidate, so its InfoNCE loss and gradient are zero)."""
    out = [(lo, min(lo + batch, n)) for lo in range(0, n, batch)]
    return [(lo, hi) for lo, hi in out if hi - lo >= min_size]
