# Deconfounded semantic decoding from natural-reading EEG

Code for the IEEE EMBC 2027 study *Does neural information survive oculomotor and language-prior controls?*
It asks whether word-level semantic retrieval from ZuCo 2.0 natural-reading EEG survives text-free
oculomotor residualization, attribute-matched and frequency-balanced candidate galleries, and retrained
non-neural controls.

- Research plan, decisions and pre-specified gates: [PLAN.md](PLAN.md)
- Current status and next action: [STATUS.md](STATUS.md)
- Original proposal (specification): `IEEE EMBC 2027 研究方案说明文档（修订版 Idea 1）.md`

## Setup

Python ≥ 3.10 with numpy, scipy, pandas, h5py, scikit-learn, torch (CUDA), sentence-transformers, nltk,
wordfreq, tabulate and pytest. On the development machine this is `~/venvs/e2t`. External inputs are set in
`embc/paths.py` and can be overridden by environment variables:

| Variable | Default content |
|---|---|
| `EMBC_ZUCO2_NR` | ZuCo 2.0 task2-NR `Matlab files` (`results<SUBJ>_NR.mat`) |
| `EMBC_MPNET` | local `sentence-transformers/all-mpnet-base-v2` snapshot |
| `EMBC_DATA`, `EMBC_RUNS`, `EMBC_RESULTS` | derived data, trained runs and summary tables (default: `data/`, `runs/`, `results/` here) |

The NLTK perceptron tagger is fetched into `.cache/nltk_data` on first use.

## Run

```bash
python -m pytest -q                                   # unit tests (no data needed)
PY=python bash scripts/run_mvp.sh                     # full MVP, text fold 1 (PLAN.md section 5)
STAGES="train" CONDS="raw res" SEEDS=0 bash scripts/run_mvp.sh   # a subset
EXP=smoke TRAIN_ARGS="--epochs 2 --max_train_events 3000" STAGES="train report" bash scripts/run_mvp.sh
OVERWRITE=1 STAGES=train bash scripts/run_mvp.sh      # retrain runs whose configuration changed
```

Runs are stored under `runs/<EXP>/fold<k>/sf<j>/<cond>/seed<s>/` (default `EXP=main`). `metrics.json` is
written last, atomically, and records the run configuration: the arguments, content digests of the input
data (fold files, reader folds and, for EEG conditions, every reader's stage-1a npz) and digests of the
training and evaluation code (`embc/runs.py`, `CODE_FILES`). Before training, `scripts/run_status.py` checks
the whole grid in one process: a completed run is skipped only when its configuration matches, and a run
produced by a different configuration stops the pipeline with the reason on the console unless
`OVERWRITE=1`. `TRAIN_ARGS` is refused under `EXP=main`. The report refuses stale runs (inputs or code
changed since training), runs of mixed protocols within one `EXP` and an empty experiment; it counts probes
only when computed from the run's current embeddings, and injection recovery only when it covers every reader
of the current stage-1a build. Stage-1a readers count as built only once `<SUBJ>_info.json`, written last,
exists.

Stages: `build` (continuous EEG, fixation table, residualization) → `lexicon` (events, types, galleries) →
`train` (one encoder per input condition, reader fold and seed) → `probes` → `inject` → `diag` → `report`
(`results/TABLES_fold1.md`, gates in `results/summary_fold1.json`; other experiments get an `<EXP>_` prefix).
Injection recovery and diagnostics are written per text fold (`results/injection_recovery_fold<k>.csv` with its
`.json` record, `collinearity_fold<k>.json`, `validity_*_fold<k>.*`).

## Layout

```
embc/io_zuco.py   ZuCo 2.0 HDF5 reader: 100 Hz sentence EEG, sample-matched fixation onsets, rawET gaze
embc/stream.py    per-reader stream, fixation table, oculomotor design inputs (no text columns read)
embc/design.py    time-expanded FIR ridge residualization; lexical covariates rejected
embc/events.py    first-fixation events, reading measures, reader folds, window validity
embc/lexicon.py   content-word filter, type embeddings, random / matched / balanced galleries
embc/inputs.py    decoder inputs per condition (raw, res, res_ovl, ocfit, *_shift, noise, gaze)
embc/model.py     light CNN encoder, gaze MLP, InfoNCE
embc/metrics.py   tie-aware R@K / MRR / nDCG, chance, subject bootstrap, sign-flip, permutation null
embc/runs.py      atomic writes, run configuration fingerprints (data + code), completed-run validation
scripts/          stage entry points, run_status.py (grid check) and run_mvp.sh
tests/            unit tests
assets/           lobo7 text folds and channel positions (copied from the parent project)
```

ZuCo-derived data and trained models are not committed (see `.gitignore`).
