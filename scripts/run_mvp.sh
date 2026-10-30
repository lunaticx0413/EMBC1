#!/usr/bin/env bash
# MVP pipeline (PLAN.md section 5). Run from the repository root.
#   STAGES="lexicon train" FOLD=1 PY=~/venvs/e2t/bin/python bash scripts/run_mvp.sh
#   OVERWRITE=1 STAGES=train bash scripts/run_mvp.sh     # retrain runs whose configuration changed
# Runs go to $RUNS/$EXP/fold<k>/... . run_status.py checks the whole grid in one process first: completed runs
# with the same configuration (arguments, input data and code digests) are skipped without starting
# train_eval.py, so their logs are kept; runs produced by another configuration stop the pipeline with the
# reason on stderr, unless OVERWRITE=1 retrains them.
# Smoke tests: EXP=smoke TRAIN_ARGS="--epochs 2 --max_train_events 3000" (TRAIN_ARGS is refused for EXP=main).
set -euo pipefail
PY=${PY:-python}
FOLD=${FOLD:-1}
EXP=${EXP:-main}
SEEDS=${SEEDS:-"0 1 2"}
SFOLDS=${SFOLDS:-"0 1 2"}
CONDS=${CONDS:-"prior raw res res_ovl ocfit raw_shift res_shift noise gaze"}
STAGES=${STAGES:-"tests build lexicon train probes inject diag report"}
TRAIN_ARGS=${TRAIN_ARGS:-}
OVERWRITE=${OVERWRITE:-}
RUNS=${EMBC_RUNS:-runs}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
if [[ -n $TRAIN_ARGS && $EXP == main ]]; then
  echo "TRAIN_ARGS changes the protocol; use a separate EXP name (e.g. EXP=smoke)" >&2
  exit 2
fi
OW=()
[[ -n $OVERWRITE ]] && OW=(--overwrite)
mkdir -p logs
has() { [[ " $STAGES " == *" $1 "* ]]; }

has tests && $PY -m pytest -q
has build && $PY scripts/build_continuous.py 2>&1 | tee logs/build_continuous.log
has lexicon && $PY scripts/build_lexicon.py --fold "$FOLD" --N 100 250 2>&1 | tee "logs/lexicon_fold$FOLD.log"
if has train; then
  todo=$($PY scripts/run_status.py --exp "$EXP" --fold "$FOLD" --sfolds $SFOLDS --conds $CONDS --seeds $SEEDS \
    $TRAIN_ARGS "${OW[@]}")
  while read -r sf c s; do
    [[ -z $sf ]] && continue
    log="logs/train_${EXP}_f${FOLD}_sf${sf}_${c}_s${s}.log"
    echo "train $EXP fold$FOLD sf$sf $c seed$s"
    if ! $PY scripts/train_eval.py --exp "$EXP" --fold "$FOLD" --subject_fold "$sf" --cond "$c" --seed "$s" \
        $TRAIN_ARGS "${OW[@]}" < /dev/null > "$log" 2>&1; then
      echo "training failed: $EXP fold$FOLD sf$sf $c seed$s; last lines of $log:" >&2
      tail -n 5 "$log" >&2
      exit 1
    fi
  done <<< "$todo"
fi
if has probes; then
  runs=()
  for d in "$RUNS/$EXP/fold$FOLD"/sf*/{raw,res,res_ovl}/seed*; do
    [[ -f $d/metrics.json ]] && runs+=("$d")             # completed runs; probes.py skips stale ones
  done
  (( ${#runs[@]} )) && $PY scripts/probes.py --runs "${runs[@]}" 2>&1 | tee "logs/probes_${EXP}_fold$FOLD.log"
fi
has inject && $PY scripts/inject_recovery.py --fold "$FOLD" 2>&1 | tee "logs/inject_fold$FOLD.log"
has diag && $PY scripts/diagnostics.py --fold "$FOLD" 2>&1 | tee "logs/diag_fold$FOLD.log"
has report && $PY scripts/report.py --exp "$EXP" --fold "$FOLD" 2>&1 | tee "logs/report_${EXP}_fold$FOLD.log"
exit 0
