"""Which runs of a training grid still need training, checked in one process (no torch import).

Prints "<subject_fold> <cond> <seed>" for every run to train. A completed run produced by a different
configuration is reported on stderr and makes the exit status 1; with --overwrite it is listed for
retraining instead. Any other argument is passed to train_eval.py's parser, so the configurations
compared are exactly the ones train_eval.py would record. The prior condition is deterministic and
runs for seed 0 only.

    python scripts/run_status.py --sfolds 0 1 2 --conds raw res --seeds 0 1 2 [--exp main --fold 1 ...]
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embc import events as ev  # noqa: E402
from embc import paths, runs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfolds", type=int, nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    g, rest = ap.parse_known_args()
    parser = runs.train_parser()
    todo, done, refused, folds_ready = [], 0, [], set()
    for sf in g.sfolds:
        for c in g.conds:
            for s in g.seeds:
                if c == "prior" and s != 0:
                    continue
                a = parser.parse_args(rest + ["--subject_fold", str(sf), "--cond", c, "--seed", str(s)])
                if (a.fold, a.k_subject_folds) not in folds_ready:          # as train_eval.py does
                    E = pd.read_csv(paths.fold_dir(a.fold) / "events.csv", usecols=["subject"])
                    ev.subject_folds(sorted(E.subject.unique()), a.k_subject_folds)
                    folds_ready.add((a.fold, a.k_subject_folds))
                config = runs.run_config(vars(a), runs.input_digests(a.fold, a.k_subject_folds, c))
                try:
                    state = runs.existing_run_state(runs.run_dir(a), config)
                except runs.RunConfigMismatch as err:
                    if not a.overwrite:
                        refused.append(str(err))
                        continue
                    state = "missing"
                if state == "done":
                    done += 1
                else:
                    todo.append(f"{sf} {c} {s}")
    print(f"run status: {len(todo)} to train, {done} completed, {len(refused)} refused", file=sys.stderr)
    if refused:
        print("refusing to reuse runs (set OVERWRITE=1 to retrain them, or use another EXP):\n  "
              + "\n  ".join(refused), file=sys.stderr)
        raise SystemExit(1)
    print("\n".join(todo))


if __name__ == "__main__":
    main()
