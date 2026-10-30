"""Stage 1: per-subject continuous stream, fixation table and oculomotor residualization.

Writes data/continuous/<SUBJ>.npz (raw, res, resfull as float16; bad mask; sentence segments; gaze trace;
fixation intercept kernel), <SUBJ>_fix.csv (all word fixations), <SUBJ>_words.json and <SUBJ>_info.json.
Each file is written atomically; <SUBJ>_info.json is written last and marks the reader as complete.
It records the npz content digest that training runs fingerprint.
Resumable: complete readers are skipped; a partially written reader is rebuilt.

    python scripts/build_continuous.py [--subjects YAC,YAG] [--ridge 1e-3]
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embc import paths  # noqa: E402
from embc.design import residualize  # noqa: E402
from embc.io_zuco import read_subject, subject_from_path  # noqa: E402
from embc.runs import file_digest, write_atomic  # noqa: E402
from embc.stream import build_stream, design_inputs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--max_sentences", type=int, default=None, help="smoke tests only")
    ap.add_argument("--force", action="store_true", help="rebuild finished subjects")
    a = ap.parse_args()
    out = paths.DATA / "continuous"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(paths.ZUCO2_NR / "*_NR.mat")))
    if a.subjects:
        keep = set(a.subjects.split(","))
        files = [p for p in files if subject_from_path(p) in keep]
    for p in files:
        s = subject_from_path(p)
        if paths.subject_complete(s) and not a.force:
            print(f"{s}: done, skipping", flush=True)
            continue
        outs = paths.subject_outputs(s)
        outs["info"].unlink(missing_ok=True)                 # drop the marker before touching any output
        t0 = time.time()
        sents = set(range(a.max_sentences)) if a.max_sentences else None
        subj = read_subject(p, sents)
        Y, bad, seg_id, seg, gaze, fix, words = build_stream(subj)
        valid = ~bad
        events, cont = design_inputs(fix, gaze, valid)
        r = residualize(Y, valid, seg_id, events, cont, ridge=a.ridge)
        info = dict(subject=s, n_sentences=len(seg), n_samples=int(len(Y)), n_fixations=int(len(fix)),
                    n_pieces=subj["n_pieces"], n_unmatched=subj["n_unmatched"],
                    n_gap_events=int(len(events.get("gap", {}).get("onset", []))),
                    gaze_coverage=float(np.isfinite(gaze[:, 0])[valid].mean()),
                    et_offset_sd_ms_max=float(np.nanmax([x["et_offset_sd"] for x in subj["sentences"]])),
                    ridge_rel=a.ridge, **r["info"], seconds=round(time.time() - t0, 1))
        tmp = str(outs["npz"]) + ".tmp.npz"
        np.savez(tmp, raw=Y.astype(np.float16), res=r["res"].astype(np.float16),
                 resfull=r["resfull"].astype(np.float16), bad=bad, seg=seg, gaze=gaze.astype(np.float32),
                 kernel=r["kernel"].astype(np.float32), kernel_lags=r["kernel_lags"])
        info["npz_sha256"] = file_digest(tmp)                 # identifies this build in run configurations
        os.replace(tmp, outs["npz"])
        fix.to_csv(str(outs["fix"]) + ".tmp", index=False)
        os.replace(str(outs["fix"]) + ".tmp", outs["fix"])
        write_atomic(outs["words"], json.dumps({str(k): v for k, v in words.items()}))
        write_atomic(outs["info"], json.dumps(info, indent=1))        # completion marker, last
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
