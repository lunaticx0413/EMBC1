# Plan: deconfounded word-level semantic decoding (IEEE EMBC 2027, revised Idea 1)

Source proposal: [IEEE EMBC 2027 研究方案说明文档（修订版 Idea 1）.md](IEEE%20EMBC%202027%20研究方案说明文档（修订版%20Idea%201）.md)
(GPT-drafted; treated as the specification). Plan date: 2026-10-29. Status and next action: [STATUS.md](STATUS.md).

## 1. Claim under test

> Does lexical-semantic decodability of natural-reading EEG survive explicit control of oculomotor structure,
> candidate-word attributes and non-neural priors?

The paper is a validity study, not a new decoder. The primary criterion is the proposal's §32:
**residualized real EEG > matched controls > chance**, **AND** a synthetic neural signal survives
residualization, **AND** decodable oculomotor information decreases. All main results come before any language
model; LLM generation is out of scope.

## 2. Verified constraints from the local data audit

| Fact (checked 2026-10-29) | Consequence for the proposal |
|---|---|
| ZuCo 2.0 NR: 18 readers, 243–349 sentences each, 105 ch at 500 Hz; 118,336 word fixations, 0 unmatched `rawEEG` pieces | Event table is complete for word fixations |
| The released `rawData` is Automagic output: 0.5–50 Hz filtering, **EOG regression performed** (EOG ch 1, 8, 14, 17, 21, 25, then dropped), **MARA ICA performed** (checked in `automagic` struct of `gip_YAC_NR4_EEG.mat`) | Proposal's "Raw" and "ICA-cleaned" conditions (§17) collapse: the local `raw` *is* the ICA/EOG-cleaned condition. A truly unprocessed condition needs the ZuCo OSF raw release (decision D1) |
| No EOG channels in the release | HEOG/VEOG regressors replaced by a time-aligned gaze-position trace (corneo-retinal proxy) plus saccade/fixation event regressors; the `ocfit` condition plays the role of "EOG-only" |
| `word.rawET` gives 500 Hz gaze samples per fixation; ET↔EEG offset is constant within a sentence (SD ≤ 0.14 ms across all readers) | Gaze can be placed on the EEG timeline exactly |
| Sentence-level `allFixations` does not line up with word fixations (median 36 px / 110 ms error under either index base) | Not used. Off-word fixations (~5% in a 40-sentence check of one reader) have no onsets; inter-fixation gaps > 100 ms enter as a separate blink/off-word event |
| ZuCo `FFD/GD/TRT` fields are in 500 Hz samples, not ms | FFD/GD/TRT recomputed from fixation durations |
| No blink annotations | Blink probe (§15) dropped; gap events are the proxy |
| DERCo not present locally; `datasets/external/EEG-To-Text_official` (Wang & Ji 2022) present, band-power word features | §16.2 and §21 are Stage 3 |
| Prior work in this repository: P01 (random sentence splits leak topic through presentation order → contiguous-block splits), P16 (sentence retrieval fully explained by duration structure), **P21 (fixation-locked ZuCo word retrieval: real ≈ swapped ≈ zero-EEG model)**, P20 (EOG-only control reaches ~60% of the permutation gap on COFETT) | High prior probability that **raw** EEG itself is at the null on this task. The proposal's outcomes A–C all assume raw > chance, so gate G1 (below) is added |
| Training-text frequency prior (no EEG): MRR 0.240 random gallery, 0.235 matched, 0.164 balanced, **0.045 balanced_exact** (chance 0.052; sf0 test readers) | Corpus repetition (Wikipedia biographies) makes "above chance" meaningless on random/matched galleries; see decision 6 |

## 3. Proposal → implementation

| Proposal | Implementation | State |
|---|---|---|
| §5 oculomotor-residualized continuous EEG | `embc/design.py` (time-expanded FIR ridge), `embc/stream.py`, `scripts/build_continuous.py` | built for 18 readers |
| §5.3 no text covariates in decoder input | whitelist + rejection in `design.check_names`; `tests/test_design.py` (lexical names rejected; design identical under shuffled words) | tested |
| §5.4 lexical rERP diagnostic path | same regression machinery, separate script, results never written to inputs | Stage 3 |
| §6 collinearity (Pearson, Spearman, VIF, condition number) | `scripts/diagnostics.py`; cond(X) per reader in `data/continuous/*_info.json` | smoke-tested |
| §7 light EEG encoder | `embc/model.py` (1×1 spatial conv, 3 residual temporal blocks, SE attention, MLP head) | smoke-tested |
| §8 word-level retrieval, InfoNCE to frozen text embeddings | frozen all-mpnet-base-v2 on isolated types; in-batch InfoNCE over unique types | smoke-tested |
| §9 content-word filter | UD NOUN/VERB/ADJ/ADV (PTB perceptron tagger in context), sklearn stop list, alphabetic, ≥ 2 chars, training-text count ≥ 1 | built |
| §10–11 fixed galleries N = 100 (N = 250 sensitivity), random and attribute-matched | `embc/lexicon.py`; plus `balanced` / `balanced_exact` (decision 6) | built |
| §12, §25 metrics | tie-aware R@1/5/10, MRR, nDCG; analytic chance | tested |
| §13 unseen subject + unseen text | lobo7 contiguous text blocks × 3 fixed reader folds | implemented |
| §14–15, §22 post-hoc probes | `scripts/probes.py`: ridge + MLP on frozen z → log FFD, log GD, saccade amplitude in/out, saccade dx, fixation x/y | smoke-tested |
| §17 Exp 1 input construction | `raw`, `res`, `res_ovl` | smoke-tested |
| §18 Exp 2 validity controls | `noise`, `raw_shift`/`res_shift` (circular shift ≥ 10 s), `gaze`, `ocfit`, `prior`, within-subject permutation null | smoke-tested |
| §19 Exp 3 synthetic injection | `scripts/inject_recovery.py` (exact via linearity; SNR −10…+5 dB; ρ = 0, 0.5 collinearity with FFD) | smoke-tested |
| §20 Exp 4 real-data positive control | `scripts/diagnostics.py`: variance retention, FRP shape correlation, GFP ratio, split-half reliability | smoke-tested |
| §21 Exp 5 published model | EEG-To-Text official with residualized band features | Stage 3 |
| §25.4 statistics | subject-bootstrap CIs, paired sign-flip test, permutation null | tested |
| §30 MVP | `scripts/run_mvp.sh` | ready to run |

## 4. Design decisions and deviations (with reasons)

1. **Two residualized conditions.** The FIR design has fixation-onset events (intercept, incoming saccade
   amplitude/dx/dy, fixation x/y), saccade-onset events (fixation offsets; intercept, outgoing
   amplitude/dx/dy), gap events (> 100 ms) and the continuous gaze x/y trace. Lags: −200…+800 ms, −100…+400 ms
   and −100…+500 ms. `res` removes everything **except** the fixation-onset intercept. Removing that term would
   subtract the average fixation-related potential, including its average lexical part, and Exp 4 would then
   fail by construction. `res_ovl` additionally removes the neighbouring fixations' average responses, keeping
   the event's own (RQ1, overlap). Both are linear and label-free.
2. **Residualization is fitted per reader on all of that reader's EEG.** The design has no text labels, so this
   is the §24 "within-subject ocular fit" that the proposal allows at test time. Ridge 1e-3 relative to the
   mean diagonal of X'X.
3. **Event and window.** The event is the first fixation on a word. The epoch is −100…+600 ms at 100 Hz with a
   −100…0 ms baseline, identical for every condition. Events need ≥ 90% valid samples inside their own sentence.
4. **PROPN excluded.** The proposal names only NOUN/VERB/ADJ/ADV, and names are poorly embedded as isolated
   words (decision D2).
5. **Pool.** Targets and negatives both come from content types seen in training texts (count ≥ 1, set from
   training texts only). This avoids a seen-versus-unseen bias between target and negatives. Test target
   *tokens* are still in unseen sentences, read by unseen readers.
6. **Frequency-prior control (new).** A training-frequency ranking reaches MRR 0.24 on random and the
   proposal's matched gallery (POS + length + zipf + log training count). With 1,021 pool types,
   nearest-neighbour matching cannot supply 99 equally frequent negatives for frequent targets. So:
   - `balanced`: same POS; negatives split evenly between more and less frequent types, with tied types
     treated as neutral.
   - `balanced_exact`: the 48% of val/test events (N = 100) where that balance is exact. The prior scores 0.045
     there (chance 0.052).

   "Above chance" is evaluated **only** on `balanced_exact`. On the other galleries every EEG claim is a
   contrast against retrained `noise`, `prior`, `*_shift` and `gaze` models, which can learn the same prior.
7. **Text encoder.** all-mpnet-base-v2 (local snapshot) on isolated lowercase types, frozen. The task is
   retrieval of the word type's context-free meaning. False negatives (cosine ≥ 0.85 to the target) are
   excluded from galleries.
8. **Splits.** Text fold 1 of lobo7 (test = block 1, model selection = block 2, train = blocks 3–7). Readers are
   in 3 seeded folds of 6. Train = 12 readers × training blocks; selection = the same readers × the selection
   block; test = 6 held-out readers × the test block. Pooling the 3 reader folds tests every reader once.
   Folds 2–7 are Stage 2.
9. **Controls are retrained.** Each input condition trains its own encoder with identical architecture,
   budget and galleries (§18). The permutation null (z permuted within reader) is the evaluation-time swap
   control.
10. **Shift control.** A per-reader circular shift (≥ 10 s) of the event times on the concatenated stream. It
    keeps the signal distribution and spectrum and breaks fixation locking.

## 5. Run order and budget

| Stage | Command | Estimated cost |
|---|---|---|
| 0 audit, tests | `python -m pytest -q` | done, < 1 s |
| 1a continuous + residualization | `python scripts/build_continuous.py` | done, ~5 min CPU, 2.2 GB |
| 1b events, galleries | `python scripts/build_lexicon.py --fold 1 --N 100 250` | done, ~1 min |
| 1c MVP training | `STAGES=train bash scripts/run_mvp.sh` (9 conditions × 3 reader folds × 3 seeds; prior once per fold) | ~1 h GPU (estimate) |
| 1d probes, injection, diagnostics, report | `STAGES="probes inject diag report" bash scripts/run_mvp.sh` | ~10 min |
| 2 remaining text folds 2–7 | `FOLD=k` for build_lexicon + run_mvp | only if G1 passes; ~6 h GPU |
| 3 published model, DERCo, rERP diagnostic, unprocessed-EEG condition | separate scripts | after Stage 2 and decision D1 |

Use `PY=~/venvs/e2t/bin/python` on this machine. Main runs live in `runs/main/`; smoke or exploratory runs
need their own `EXP` name. A run directory is only reused when its recorded configuration (arguments, input
data digests including the stage-1a EEG, code digests) matches; `OVERWRITE=1` retrains runs that do not, and
the report refuses runs whose inputs or code changed after training. The scripts read data paths from `embc/paths.py`, which can
be overridden with `EMBC_*` environment variables.

## 6. Gates (fixed before any full run; computed by `scripts/report.py`)

| Gate | Criterion (MRR; subject-level, 95% bootstrap CI over readers) | If it fails |
|---|---|---|
| G0 engineering | tests pass; 0 unmatched fixations; injection amplitude slope ≥ 0.8 for `res` and `res_ovl` at ρ = 0 (injection run on every reader of the current stage-1a build) | fix the pipeline before reading results |
| G1 raw signal exists | random_N100: raw − noise and raw − raw_shift CI lower bound > 0, and permutation p < 0.05 | **Outcome D**: no decodable word information even before deconfounding with this decoder. Stop expansion; before reporting the null, add an injected-signal *decoding* control (not yet implemented) to bound what this decoder can detect |
| G2 primary (§32) | balanced_exact_N100: res CI lower bound > chance and res − noise lower bound > 0; matched_N100: res − {res_shift, gaze, prior, noise} lower bounds > 0 | the claim is not supported |
| G3 probes | linear probe R²(z_res) < R²(z_raw) on ≥ 4 of 7 oculomotor targets | "decodable oculomotor information decreases" is not supported |
| Primary criterion | G2 ∧ G0-injection ∧ G3 | |

Interpretation follows proposal §27. **A**: G2 ∧ G3 ∧ injection pass. **B**: raw − res > 0 while res ≈ shift
(this needs injection, probe decrease and Stage 3 replication before any wording about earlier work). **C**: raw
− res > 0 and res > shift. **D**: G1 fails. Exploratory analyses (N = 250, seen-reader splits, other
hyperparameters) are labelled as such and never move a gate.

## 7. Leakage controls

| Rule (proposal §24) | Enforcement | Check |
|---|---|---|
| No text attributes in residualization | whitelist of ocular modulators; lexical names raise `ValueError` | `test_lexical_covariates_are_rejected`, `test_design_ignores_word_identity` |
| Normalization from training data only | channel mean/SD from training events only (`train_eval.py`) | code review |
| Vocabulary threshold from training texts only | `train_count` counts only training-block sentences | `summary.json` |
| Model selection without test readers or test texts | early stopping on training readers × selection block | `metrics.json` records the partitions |
| Same galleries for every condition | one gallery file per fold, keyed by event id; runs record gallery digests | contrasts are paired on shared events; a gate uses a contrast only when both conditions cover identical events |
| No presentation-order leak (P01) | contiguous-block text folds | lobo7 copied in `assets/splits` |

## 8. Decisions for the researcher

- **D1** Download the ZuCo 2.0 raw release (OSF) to restore a truly unprocessed "Raw" condition and real EOG
  channels? That requires a large download, and the data drive has 8.8 GB free after Stage 1a.
- **D2** Include proper nouns (PROPN) as targets? This roughly doubles the target count, but names are poorly
  represented by isolated-word embeddings.
- **D3** If G1 fails (Outcome D), should the EMBC paper become a controlled-null / validity-framework paper,
  with the controls, frequency prior and injection bound as the contribution? The alternative is to stop.

## 9. Known limitations

- The test block of fold 1 yields ~100 eligible events per reader. Single-fold CIs are wide, and Stage 2 adds 6×
  the events.
- Matched-gallery quality is limited by vocabulary size: mean |Δchars| 2.16 vs 2.61 random, |Δzipf| 0.74 vs
  0.91, same POS 97%.
- The gaze trace is interpolated across saccades and blinks. Off-word fixations have no onset events.
- Residualization cannot separate oculomotor variance that is collinear with lexical processing (§6). The
  ρ = 0.5 injection quantifies this: the amplitude slope drops from ~0.99 to ~0.92 in the smoke test on two
  readers.
