"""Stage 3: train one encoder for one input condition and evaluate fixed-gallery word retrieval.

Partition (unseen subject + unseen text): train = training subjects x training block texts; model selection
= training subjects x validation block (val MRR, random gallery); test = held-out subjects x test block.
cond "prior" trains nothing and ranks candidates by their training-text frequency.

Outputs in runs/fold<f>/sf<k>/<cond>/seed<s>/: model.pt, z.npz (embeddings of train/val/test events for
probes), test_<gallery>.csv (per-event tie-aware metrics), metrics.json, train_log.json.

    python scripts/train_eval.py --fold 1 --subject_fold 0 --cond res --seed 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embc import events as ev  # noqa: E402
from embc import lexicon as lx  # noqa: E402
from embc import metrics as mt  # noqa: E402
from embc import paths  # noqa: E402
from embc.inputs import load_condition  # noqa: E402
from embc.model import EEGEncoder, VectorEncoder, info_nce  # noqa: E402


def partitions(E, fold_subjects, sf):
    test_s = set(fold_subjects[sf])
    insub = E.subject.isin(test_s)
    el = E.eligible.astype(bool)
    return dict(train=E[el & ~insub & (E.partition == "train")], val=E[el & ~insub & (E.partition == "val")],
                test=E[el & insub & (E.partition == "test")])


def event_frame(events, m):
    return pd.DataFrame(dict(event_id=events.event_id.to_numpy(), subject=events.subject.to_numpy(),
                             sent_idx=events.sent_idx.to_numpy(), type_id=events.type_id.to_numpy(), **m))


def evaluate(Z, T, galleries, events, scorer=None):
    """Per-event metrics for every gallery. scorer(G) overrides the embedding scores (frequency prior)."""
    rows, summ = {}, {}
    for name, (r, G) in galleries.items():
        S = scorer(G) if scorer is not None else mt.score_galleries(Z[r], T, G)
        df = event_frame(events.iloc[r], mt.rank_metrics(*mt.gallery_ranks(S), G.shape[1]))
        rows[name] = df
        summ[name] = summarize(df, G.shape[1])
    return rows, summ


def summarize(df, N):
    s = {"n_events": int(len(df)), "chance": mt.chance(N)}
    for k in ("R@1", "R@5", "R@10", "RR", "nDCG"):
        _, sm = mt.subject_means(df[k].to_numpy(), df.subject.to_numpy())
        m, lo, hi = mt.bootstrap_ci(sm)
        s[k] = dict(mean=m, lo=lo, hi=hi)
    return s


def prior_scores(T_count, G):
    return np.log1p(T_count[G]).astype(np.float32)


def main():
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
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    fdir = paths.fold_dir(a.fold)
    out = paths.RUNS / f"fold{a.fold}" / f"sf{a.subject_fold}" / a.cond / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    E = pd.read_csv(fdir / "events.csv", keep_default_na=False, na_values=[""])
    ty = np.load(fdir / "types.npz")
    T = ty["emb"].astype(np.float32)
    subjects = sorted(E.subject.unique())
    folds = ev.subject_folds(subjects, a.k_subject_folds)
    P = partitions(E, folds, a.subject_fold)
    if a.max_train_events:
        P["train"] = P["train"].sample(min(a.max_train_events, len(P["train"])), random_state=a.seed)
    gal = {k: lx.load_galleries(fdir, P[k].event_id.to_numpy()) for k in ("val", "test")}
    info = dict(vars(a), test_subjects=folds[a.subject_fold], n_train=len(P["train"]), n_val=len(P["val"]),
                n_test=len(P["test"]))

    if a.cond == "prior":
        rows, res = evaluate(None, None, gal["test"], P["test"], scorer=lambda G: prior_scores(ty["train_count"], G))
        for name, df in rows.items():
            df.to_csv(out / f"test_{name}.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(dict(info=info, test=res), indent=1))
        print(json.dumps({k: v["RR"] for k, v in res.items()}))
        return

    allE = pd.concat([P["train"], P["val"], P["test"]])
    X, M = load_condition(a.cond, allE, seed=a.seed)
    ntr, nva = len(P["train"]), len(P["val"])
    dev = torch.device(a.device)
    if M is None:                                                    # vector input (gaze)
        Xtr = X[:ntr]
        mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
        sd[~(sd > 0)] = 1.0
        Xn = np.nan_to_num((X - mu) / sd).astype(np.float32)
        Xt = torch.from_numpy(Xn).to(dev)
        Mt = None
        model = VectorEncoder(X.shape[1], T.shape[1])
    else:
        Xtr, Mtr = X[:ntr], M[:ntr]
        s1 = np.zeros(X.shape[1]); s2 = np.zeros(X.shape[1]); cnt = 0
        for i in range(0, ntr, 2048):
            xb = Xtr[i:i + 2048].astype(np.float64)
            mb = Mtr[i:i + 2048][:, None, :]
            s1 += (xb * mb).sum((0, 2)); s2 += (xb * xb * mb).sum((0, 2)); cnt += mb.sum()
        mu = s1 / cnt
        sd = np.sqrt(np.maximum(s2 / cnt - mu ** 2, 1e-12))
        Xt = torch.from_numpy(X).to(dev)                              # float16 on device
        Mt = torch.from_numpy(M).to(dev)
        mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)[None, :, None]
        sd_t = torch.tensor(sd, dtype=torch.float32, device=dev)[None, :, None]
        model = EEGEncoder(X.shape[1], X.shape[2], T.shape[1])
    del X
    model.to(dev)

    def batch_input(idx):
        if Mt is None:
            return Xt[idx]
        x = (Xt[idx].float() - mu_t) / sd_t
        return (x * Mt[idx][:, None, :]).clamp(-10, 10)

    def embed(lo, hi):
        model.eval()
        zs = []
        with torch.no_grad():
            for i in range(lo, hi, 1024):
                zs.append(model(batch_input(torch.arange(i, min(i + 1024, hi), device=dev))).float().cpu().numpy())
        return np.concatenate(zs) if zs else np.zeros((0, T.shape[1]), np.float32)

    Tt = torch.from_numpy(T).to(dev)
    tid = torch.from_numpy(allE.type_id.to_numpy()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_state, bad_epochs, log = -1.0, None, 0, []
    Gsel = gal["val"][a.select_gallery][1]
    for epoch in range(a.epochs):
        model.train()
        perm = torch.randperm(ntr, device=dev)
        tot, nb = 0.0, 0
        for i in range(0, ntr - a.batch + 1, a.batch):
            idx = perm[i:i + a.batch]
            loss = info_nce(model(batch_input(idx)), tid[idx], Tt, a.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        Zv = embed(ntr, ntr + nva)
        g, e = mt.gallery_ranks(mt.score_galleries(Zv, T, Gsel))
        val_mrr = float(mt.rank_metrics(g, e, Gsel.shape[1])["RR"].mean())
        log.append(dict(epoch=epoch, loss=tot / max(nb, 1), val_mrr=val_mrr))
        print(json.dumps(log[-1]), flush=True)
        if val_mrr > best:
            best, bad_epochs = val_mrr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                break
    model.load_state_dict(best_state)
    Ztr, Zva, Zte = embed(0, ntr), embed(ntr, ntr + nva), embed(ntr + nva, len(allE))
    rows, summ = evaluate(Zte, T, gal["test"], P["test"])
    for name, df in rows.items():
        df.to_csv(out / f"test_{name}.csv", index=False)
    _, summ_val = evaluate(Zva, T, gal["val"], P["val"])
    torch.save(dict(state=best_state, config=info, norm=dict(mu=np.asarray(mu), sd=np.asarray(sd))), out / "model.pt")
    np.savez(out / "z.npz", train_ids=P["train"].event_id.to_numpy(), val_ids=P["val"].event_id.to_numpy(),
             test_ids=P["test"].event_id.to_numpy(), z_train=Ztr.astype(np.float16), z_val=Zva.astype(np.float16),
             z_test=Zte.astype(np.float16))
    info.update(best_val_mrr=best, epochs_run=len(log), seconds=round(time.time() - t0, 1))
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    (out / "metrics.json").write_text(json.dumps(dict(info=info, val=summ_val, test=summ), indent=1))
    print(json.dumps({k: v["RR"] for k, v in summ.items()}))


if __name__ == "__main__":
    main()
