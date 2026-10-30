# Status

Updated 2026-10-30. **State: MVP pipeline implemented and smoke-tested; no full experiment has run, so there are
no scientific results yet.**

## Done

- Proposal read and converted into [PLAN.md](PLAN.md): data constraints, deviations, run order, fixed gates.
- Data audit: the release is already EOG-regressed + MARA-cleaned, has no EOG channels, has usable `rawET`, has
  an unusable `allFixations`, and stores FFD/GD/TRT in samples (PLAN.md section 2).
- Stage 1a: continuous streams and residualization for all 18 readers (`data/continuous/`, 2.2 GB; 0 unmatched
  fixations; ocular nuisance explains 1.1–3.6% of variance; cond(X) 16–40).
- Stage 1b, text fold 1: 77,338 first fixations; 29,548 content-word events; pool of 1,021 types; eligible
  events 19,841 train / 1,465 selection / 1,803 test; galleries random / matched / balanced at N = 100 and 250.

## Checks

| Check | Result |
|---|---|
| `python -m pytest -q` | 43 passed |
| Smoke run of every stage (2 epochs, 3,000 training events, reader folds 0–1, scratch output) | all stages ran without error; numbers not interpretable |
| Review fixes (2026-10-29): config-validated run reuse, selection-gallery rows, partial/undersized batches, stage-1a completion marker | each exercised on real data in scratch dirs: changed config refused, same config skipped, `TRAIN_ARGS` refused under `EXP=main`, `balanced_exact` selection runs, 100 training events → 1 step/epoch, 1 event refused, interrupted reader rebuilt and blocked downstream |
| Second review fixes (2026-10-30): data + code digests in run configs, report/probe/injection validation, grid check, fold-scoped results, injection ridge and per-reader RNG | exercised on real data in scratch dirs (1-epoch smoke runs): rerun skips without touching logs; changed arguments, regenerated galleries and a truncated `metrics.json` refused with the reason on the console, then retrained with `OVERWRITE=1`; report refuses stale runs, mixed protocols and an empty `EXP` without writing; partial reader-fold coverage gives shared-event contrasts whose gates stay `None`; retrain drops old probes; 2-reader injection gives G0 `None` ("covers 2 of 18"); training failure prints the log tail; stage-1a build records the npz digest |
| Frequency prior with no EEG (reader fold 0) | MRR random 0.240, matched 0.235, balanced 0.164, balanced_exact 0.045 (chance 0.052) |
| Injection recovery (smoke, 2 readers, ρ = 0) | amplitude slope 0.99 for raw/res/res_ovl |

Not run: full Stage 1c training, Stage 2 (text folds 2–7), Stage 3 (published model, DERCo, rERP diagnostic).

## Next action

Stage-1a npz digests have been backfilled into all 18 `<SUBJ>_info.json` files. Run Stage 1c–1d on text fold 1 (`STAGES="train probes inject diag report" PY=~/venvs/e2t/bin/python bash
scripts/run_mvp.sh`, ~1 h GPU estimated). Then read G1 in `results/TABLES_fold1.md` before anything else.
Researcher decisions D1–D3 are listed in PLAN.md section 8.
