"""Stage 2: event table, content-word filter, type embeddings and fixed candidate galleries for one text fold.

Writes data/fold<k>/events.csv (every first fixation of every word, with eligibility flags), types.npz
(type strings, UPOS, nchar, zipf, train counts, mpnet embeddings), galleries_<kind>_N<N>.npz for the
val and test events, and summary.json (counts, gallery match quality).

    python scripts/build_lexicon.py --fold 1 [--N 100 250] [--min_train_count 1]
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embc import events as ev  # noqa: E402
from embc import lexicon as lx  # noqa: E402
from embc import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--N", type=int, nargs="+", default=[100])
    ap.add_argument("--min_train_count", type=int, default=1)
    ap.add_argument("--fn_thr", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--widen", type=float, default=3, help="matched: sample from widen*(N-1) nearest")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = paths.fold_dir(a.fold)
    out.mkdir(parents=True, exist_ok=True)
    cdir = paths.DATA / "continuous"
    subjects = ev.subjects_available()
    part_of = ev.text_partition(a.fold)

    # sentence tokens: identical across readers is checked, first reader's copy is used
    words, mismatch = {}, 0
    for s in subjects:
        for k, toks in json.loads((cdir / f"{s}_words.json").read_text()).items():
            k = int(k)
            if k in words and words[k] != toks:
                mismatch += 1
            words.setdefault(k, toks)
    upos = lx.tag_sentences(words, paths.NLTK_DATA)
    stop = lx.stopwords()

    fixes = pd.concat([pd.read_csv(cdir / f"{s}_fix.csv", keep_default_na=False, na_values=[""])
                       for s in subjects], ignore_index=True)
    fixes["word"] = fixes.word.astype(str)
    meas = ev.reading_measures(fixes)
    E = fixes[fixes.first_fix.astype(str) == "True"].merge(meas, on=["subject", "sent_idx", "word_pos"], how="left")
    E["partition"] = E.sent_idx.map(part_of).fillna("none")
    E["type"] = [lx.clean_word(words[i][w]) for i, w in zip(E.sent_idx, E.word_pos)]
    E["upos"] = [upos[i][w] for i, w in zip(E.sent_idx, E.word_pos)]
    E["content"] = [lx.is_content(t, u, stop) for t, u in zip(E.type, E.upos)]

    # window validity from each subject's bad mask and segments
    vf = np.zeros(len(E))
    for s in subjects:
        z = np.load(paths.continuous_path(s))
        bad, seg = z["bad"], z["seg"]
        seg_id = np.repeat(np.arange(len(seg)), seg[:, 2])
        m = (E.subject == s).to_numpy()
        vf[m] = ev.window_valid_fraction(E.onset.to_numpy()[m], bad, seg_id)
    E["valid_frac"] = vf

    # type table over content tokens of all sentences (text only, counted once per sentence occurrence)
    tok_rows = [(i, lx.clean_word(t), upos[i][j]) for i, toks in words.items() for j, t in enumerate(toks)]
    tok_rows = [r for r in tok_rows if lx.is_content(r[1], r[2], stop)]
    types = sorted({t for _, t, _ in tok_rows})
    tid = {t: k for k, t in enumerate(types)}
    pos_count = Counter((t, u) for _, t, u in tok_rows)
    maj = {}
    for (t, u), c in sorted(pos_count.items(), key=lambda kv: -kv[1]):
        maj.setdefault(t, u)
    train_count = Counter(t for i, t, _ in tok_rows if part_of.get(i) == "train")
    from wordfreq import zipf_frequency
    attrs = dict(upos=np.array([maj[t] for t in types]), nchar=np.array([len(t) for t in types], float),
                 zipf=np.array([zipf_frequency(t, "en") for t in types], float))
    tc = np.array([train_count.get(t, 0) for t in types])
    emb_path = paths.DATA / "type_emb_mpnet.npz"
    if emb_path.exists() and list(np.load(emb_path)["types"]) == types:
        emb = np.load(emb_path)["emb"]
    else:
        emb = lx.embed_types(types, paths.MPNET, device=a.device)
        np.savez(emb_path, types=np.array(types), emb=emb)
    np.savez(out / "types.npz", types=np.array(types), upos=attrs["upos"], nchar=attrs["nchar"],
             zipf=attrs["zipf"], train_count=tc, emb=emb)

    attrs["logtc"] = np.log1p(tc).astype(float)
    pool = np.flatnonzero(tc >= a.min_train_count)
    in_pool = np.zeros(len(types), bool)
    in_pool[pool] = True
    E["type_id"] = [tid.get(t, -1) for t in E.type]
    # text attributes for galleries/diagnostics only (the residualization ran upstream without them)
    E["nchar"] = E.type.str.len()
    E["zipf"] = [zipf_frequency(t, "en") for t in E.type]
    E["train_count"] = [train_count.get(t, 0) for t in E.type]
    E["eligible"] = (E.content & (E.type_id >= 0) & (E.valid_frac >= ev.MIN_VALID)
                     & (E.partition != "none")).to_numpy()
    E.loc[E.eligible, "eligible"] = in_pool[E.type_id[E.eligible]]
    E = E.sort_values(["subject", "sent_idx", "word_pos"]).reset_index(drop=True)
    E.insert(0, "event_id", np.arange(len(E)))
    E.to_csv(out / "events.csv", index=False)

    pool_attrs = dict(attrs, **{f"{k}_sd": float(attrs[k][pool].std()) for k in lx.MATCH_ATTRS})
    summary = dict(fold=a.fold, subjects=subjects, n_subjects=len(subjects), token_mismatch_sentences=mismatch,
                   n_first_fixations=int(len(E)), n_content=int(E.content.sum()), n_types=len(types),
                   pool_size=int(len(pool)), min_train_count=a.min_train_count, fn_thr=a.fn_thr, widen=a.widen,
                   eligible_by_partition=E[E.eligible].partition.value_counts().to_dict(),
                   excluded_low_valid=int((E.content & (E.valid_frac < ev.MIN_VALID)).sum()),
                   excluded_not_in_pool=int((E.content & (E.type_id >= 0) & ~in_pool[E.type_id.clip(0)]).sum()),
                   galleries={})
    target_events = E[E.eligible & E.partition.isin(["val", "test"])]
    for N in a.N:
        for kind in ("random", "matched", "balanced"):
            rng = np.random.default_rng([a.seed, a.fold, N, ("random", "matched", "balanced").index(kind)])
            G = np.zeros((len(target_events), N), np.int32)
            fb, imb = 0, []
            for r, t in enumerate(target_events.type_id.to_numpy()):
                G[r], info = lx.build_gallery(int(t), pool, pool_attrs, emb, N, kind, rng, fn_thr=a.fn_thr,
                                              widen=a.widen)
                fb += info["pos_fallback"]
                imb.append(info.get("imbalance", 0))
            np.savez(out / f"galleries_{kind}_N{N}.npz", event_id=target_events.event_id.to_numpy(), G=G,
                     imbalance=np.array(imb, np.int32))
            q = lx.gallery_quality(G, attrs, emb)
            q["pos_fallback_rate"] = fb / max(G[:, 1:].size, 1)
            if kind == "balanced":
                imb = np.abs(np.array(imb))
                q.update(mean_abs_imbalance=float(imb.mean()), frac_imbalance_gt10=float((imb > 10).mean()),
                         n_exact=int((imb <= lx.EXACT_IMBALANCE).sum()), n_events=int(len(imb)))
            summary["galleries"][f"{kind}_N{N}"] = q
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=int))
    print(json.dumps(summary, indent=1, default=int))


if __name__ == "__main__":
    main()
