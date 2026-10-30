"""Word types, content-word filter, frozen type embeddings and fixed-size candidate galleries.

Text attributes here (POS, length, frequency) are used only for target filtering, gallery construction and
diagnostics. They never enter the EEG residualization (see design.py).
"""
import re
from pathlib import Path

import numpy as np

CONTENT_UPOS = ("NOUN", "VERB", "ADJ", "ADV")
MATCH_ATTRS = ("nchar", "zipf", "logtc")
EXACT_IMBALANCE = 1                     # balanced_exact: targets whose gallery is frequency-rank balanced
_WORD = re.compile(r"[a-z]+(?:[-'][a-z]+)*")
_PTB = {"NN": "NOUN", "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN", "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
        "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "MD": "AUX"}


def clean_word(w):
    """Surface token -> lowercase type with surrounding punctuation removed."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", w.strip()).lower()


def ptb_to_upos(tag):
    if tag in _PTB:
        return _PTB[tag]
    if tag.startswith("VB"):
        return "VERB"
    return "OTHER"


def stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return frozenset(ENGLISH_STOP_WORDS)


TAGGER_URL = ("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/"
              "averaged_perceptron_tagger_eng.zip")


def ensure_tagger(nltk_dir):
    """Make the English perceptron tagger (JSON weights) available under nltk_dir."""
    import io
    import urllib.request
    import zipfile
    from pathlib import Path

    import nltk
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng/")
        return nltk
    except LookupError:
        pass
    if not nltk.download("averaged_perceptron_tagger_eng", download_dir=str(nltk_dir), quiet=True):
        # nltk's downloader rejects some proxy setups; fetch the same official package directly.
        dest = Path(nltk_dir) / "taggers"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(TAGGER_URL, timeout=60).read())) as zf:
            for m in zf.namelist():
                if m.endswith(".json") and m.startswith("averaged_perceptron_tagger_eng/") and ".." not in m:
                    (dest / m).parent.mkdir(parents=True, exist_ok=True)
                    (dest / m).write_bytes(zf.read(m))
    nltk.data.find("taggers/averaged_perceptron_tagger_eng/")
    return nltk


def tag_sentences(words_by_sent, nltk_dir):
    """{sent_idx: [surface tokens]} -> {sent_idx: [UPOS]} (PTB perceptron tagger in sentence context)."""
    nltk = ensure_tagger(nltk_dir)
    out = {}
    for k, toks in words_by_sent.items():
        stripped = [re.sub(r"^[^\w]+|[^\w]+$", "", t) or t for t in toks]
        out[k] = [ptb_to_upos(tag) for _, tag in nltk.pos_tag(stripped)]
    return out


def is_content(typ, upos, stop):
    return upos in CONTENT_UPOS and len(typ) >= 2 and _WORD.fullmatch(typ) is not None and typ not in stop


def embed_types(types, model_dir, device="cuda", batch=256):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(model_dir), device=device, local_files_only=True)
    E = m.encode(list(types), batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                 show_progress_bar=False)
    return E.astype(np.float32)


# ---------------------------------------------------------------- galleries

def _false_negative_mask(emb, target, pool, thr):
    return (emb[pool] @ emb[target]) >= thr


def build_gallery(target, pool, attrs, emb, n, kind, rng, fn_thr=0.85, widen=3):
    """One gallery of n type ids, column 0 = target.

    pool   int array of eligible type ids (the fixed candidate dictionary)
    attrs  dict of arrays over all types: upos (str), nchar, zipf, logtc (log1p training-text count), and
           their pool SDs nchar_sd, zipf_sd, logtc_sd
    kind   "random": uniform over pool; "matched": same UPOS, nearest in standardized (nchar, zipf, logtc)
           space; sampled from the widen*(n-1) nearest to avoid a deterministic hardest-negative set.
           logtc matters: corpus repetition (Wikipedia biographies) makes a training-frequency prior far
           above chance on a random gallery, and any trained model can learn that prior.
           "balanced": same UPOS, negatives split evenly above/below the target's training count (ties are
           neutral); info["imbalance"] records the residual imbalance for frequent targets.
    Returns (gallery array, info dict).
    """
    cand = pool[pool != target]
    cand = cand[~_false_negative_mask(emb, target, cand, fn_thr)]
    info = {"n_cand": int(len(cand)), "pos_fallback": 0}
    if len(cand) < n - 1:
        raise ValueError(f"pool too small for target {target}: {len(cand)} < {n - 1}")
    if kind == "random":
        neg = rng.choice(cand, n - 1, replace=False)
    elif kind == "matched":
        d = sum(np.abs(attrs[k][cand] - attrs[k][target]) / attrs[f"{k}_sd"] for k in MATCH_ATTRS)
        same = attrs["upos"][cand] == attrs["upos"][target]
        d = d + np.where(same, 0.0, 1e6)                    # other POS only if same-POS supply runs out
        order = np.argsort(d, kind="stable")
        # sample among the widen*(n-1) nearest, but never let that window spill into other POS while the
        # same-POS supply alone can fill the gallery
        near = order[:max(n - 1, min(int(widen * (n - 1)), int(same.sum())))]
        neg = cand[rng.choice(near, n - 1, replace=False)]
        info["pos_fallback"] = int((attrs["upos"][neg] != attrs["upos"][target]).sum())
    elif kind == "balanced":
        # Frequency-prior-neutral: same UPOS; negatives tied with the target's training count are neutral,
        # the rest are split evenly between more and less frequent types (nearest in nchar/zipf on each
        # side), so a training-frequency ranking puts the target mid-gallery in expectation. Very frequent
        # targets may lack more-frequent partners; the remaining imbalance is recorded.
        same = attrs["upos"][cand] == attrs["upos"][target]
        if same.sum() >= n - 1:
            cand = cand[same]
        info["pos_fallback"] = 0 if same.sum() >= n - 1 else int(n - 1 - same.sum())
        d = sum(np.abs(attrs[k][cand] - attrs[k][target]) / attrs[f"{k}_sd"] for k in ("nchar", "zipf"))
        t = attrs["logtc"][target]
        lt = attrs["logtc"][cand]
        groups = {"eq": lt == t, "up": lt > t, "dn": lt < t}
        size = {k: int(m.sum()) for k, m in groups.items()}
        k_side = min(size["up"], size["dn"], (n - 1) // 2)            # balanced core
        k_eq = min(size["eq"], n - 1 - 2 * k_side)                     # neutral ties next
        r = n - 1 - 2 * k_side - k_eq                                  # unavoidable imbalance
        k_up = k_side + min(size["up"] - k_side, r)
        k_dn = n - 1 - k_eq - k_up

        def nearest(m, k):
            g, dg = cand[m], d[m]
            near = np.argsort(dg, kind="stable")[:max(k, min(int(widen * k), len(g)))]
            return g[rng.choice(near, k, replace=False)] if k > 0 else g[:0]

        neg = np.concatenate([nearest(groups["eq"], k_eq), nearest(groups["up"], k_up),
                              nearest(groups["dn"], k_dn)])
        info["imbalance"] = int(k_dn - k_up)
    else:
        raise ValueError(kind)
    return np.concatenate([[target], neg]).astype(np.int32), info


def gallery_quality(G, attrs, emb):
    """Mean |delta| between target and negatives on the matched attributes, and text similarity."""
    t, neg = G[:, :1], G[:, 1:]
    cos = [np.einsum("nd,nkd->nk", emb[G[i:i + 512, 0]], emb[G[i:i + 512, 1:]]).mean(1)
           for i in range(0, len(G), 512)]
    return dict(
        abs_dnchar=float(np.abs(attrs["nchar"][neg] - attrs["nchar"][t]).mean()),
        abs_dzipf=float(np.abs(attrs["zipf"][neg] - attrs["zipf"][t]).mean()),
        abs_dlogtc=float(np.abs(attrs["logtc"][neg] - attrs["logtc"][t]).mean()),
        same_upos=float((attrs["upos"][neg] == attrs["upos"][t]).mean()),
        mean_cos=float(np.concatenate(cos).mean()),
    )


def load_galleries(fdir, event_ids):
    """{name: (rows, G)} for every gallery file of a fold, aligned to event_ids (rows index into event_ids).

    balanced files also yield balanced_exact_N<N>: only the events whose balanced gallery achieved
    |imbalance| <= EXACT_IMBALANCE, where a training-frequency prior is neutral by construction."""
    import pandas as pd
    out = {}
    event_ids = np.asarray(event_ids)
    for f in sorted(Path(fdir).glob("galleries_*_N*.npz")):
        z = np.load(f)
        pos = pd.Series(np.arange(len(z["event_id"])), index=z["event_id"]).loc[event_ids].to_numpy()
        name = f.stem.replace("galleries_", "")
        out[name] = (np.arange(len(event_ids)), z["G"][pos])
        if name.startswith("balanced_"):
            keep = np.flatnonzero(np.abs(z["imbalance"][pos]) <= EXACT_IMBALANCE)
            out[name.replace("balanced_", "balanced_exact_")] = (keep, z["G"][pos][keep])
    return out
