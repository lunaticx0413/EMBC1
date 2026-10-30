"""Word events (first fixation on each word), reading measures, partitions and epoch window validity."""
import json
import re

import numpy as np
import pandas as pd

from . import paths

PRE, POST = 10, 60                       # epoch -100 .. +600 ms at 100 Hz (identical for every condition)
MIN_VALID = 0.9


# a stage-1a output of reader <name>, or its temporary file (".tmp<pid>", or ".tmp.npz" for the npz)
_STAGE1A_FILE = re.compile(r"^([A-Za-z0-9]+)(?:%s)(?:\.tmp\d*(?:\.npz)?)?$"
                           % "|".join(re.escape(s) for s in paths.STAGE1A_SUFFIXES.values()))


def _stage1a_readers():
    d = paths.DATA / "continuous"
    if not d.exists():
        return set()
    return {m.group(1) for p in d.iterdir() if p.is_file() and (m := _STAGE1A_FILE.match(p.name))}


def subjects_available():
    """Readers whose stage-1a outputs are complete (an interrupted build is not listed)."""
    return sorted(s for s in _stage1a_readers() if paths.subject_complete(s))


def incomplete_subjects():
    """Readers with some stage-1a output but no completion marker (interrupted build). Files that are not
    stage-1a outputs (e.g. .directory, notes.txt, *.bak) are ignored."""
    return sorted(s for s in _stage1a_readers() if not paths.subject_complete(s))


def reading_measures(fix):
    """FFD/GD/TRT (ms) and fixation count per (subject, sentence, word) from the fixation table itself.

    ZuCo's FFD/GD/TRT fields are in 500 Hz samples; durations here come from rawEEG piece lengths.
    GD = summed duration of the run of consecutive fixations that starts with the first fixation.
    """
    out = []
    for (s, i), g in fix.sort_values("fix_ord").groupby(["subject", "sent_idx"], sort=False):
        wp, dur = g.word_pos.to_numpy(), g.dur_ms.to_numpy()
        firsts = {}
        for k, w in enumerate(wp):
            firsts.setdefault(w, k)
        for w, k0 in firsts.items():
            k = k0
            while k + 1 < len(wp) and wp[k + 1] == w:
                k += 1
            on_w = wp == w
            out.append((s, i, w, dur[k0], dur[k0:k + 1].sum(), dur[on_w].sum(), int(on_w.sum())))
    return pd.DataFrame(out, columns=["subject", "sent_idx", "word_pos", "ffd_ms", "gd_ms", "trt_ms", "nfix_ours"])


def window_valid_fraction(onset, bad, seg_id):
    """Fraction of the epoch window samples that are inside the onset's segment and not bad."""
    n = len(bad)
    idx = onset[:, None] + np.arange(-PRE, POST)[None, :]
    inside = (idx >= 0) & (idx < n)
    c = np.clip(idx, 0, n - 1)
    ok = inside & ~bad[c] & (seg_id[c] == seg_id[np.clip(onset, 0, n - 1)][:, None])
    return ok.mean(1)


def subject_folds(subjects, k=3, seed=2027):
    """Fixed (seeded) subject folds, cached in the data dir so every run uses the same assignment."""
    f = paths.DATA / f"subject_folds_k{k}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        d = json.loads(f.read_text())
        if sorted(d["subjects"]) == sorted(subjects):
            return d["folds"]
    perm = list(np.random.default_rng(seed).permutation(sorted(subjects)))
    folds = [sorted(perm[i::k]) for i in range(k)]
    f.write_text(json.dumps({"subjects": sorted(subjects), "seed": seed, "folds": folds}, indent=1))
    return folds


def text_partition(fold):
    sp = json.loads((paths.SPLITS / f"fold{fold}.json").read_text())["part_sent_idx"]
    m = {}
    for part in ("train", "val", "test"):
        for i in sp[part]:
            m[int(i)] = part
    return m
