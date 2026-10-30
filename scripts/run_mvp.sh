#!/usr/bin/env bash
# MVP pipeline (PLAN.md section 5). Run from the repository root.
#   STAGES="lexicon train" FOLD=1 PY=~/venvs/e2t/bin/python bash scripts/run_mvp.sh
# Runs go to $RUNS/$EXP/fold<k>/...; train_eval.py skips a completed run only if its recorded configuration
# matches, and refuses to reuse one produced by a different configuration.
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
RUNS=${EMBC_RUNS:-runs}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
if [[ -n $TRAIN_ARGS && $EXP == main ]]; then
  echo "TRAIN_ARGS changes the protocol; use a separate EXP name (e.g. EXP=smoke)" >&2
  exit 2
fi
mkdir -p logs
has() { [[ " $STAGES " == *" $1 "* ]]; }

has tests && $PY -m pytest -q
has build && $PY scripts/build_continuous.py 2>&1 | tee logs/build_continuous.log
has lexicon && $PY scripts/build_lexicon.py --fold "$FOLD" --N 100 250 2>&1 | tee "logs/lexicon_fold$FOLD.log"
if has train; then
  for sf in $SFOLDS; do for c in $CONDS; do for s in $SEEDS; do
    [[ $c == prior && $s != 0 ]] && continue            # deterministic, no training
    echo "train $EXP fold$FOLD sf$sf $c seed$s"
    $PY scripts/train_eval.py --exp "$EXP" --fold "$FOLD" --subject_fold "$sf" --cond "$c" --seed "$s" $TRAIN_ARGS \
      > "logs/train_${EXP}_f${FOLD}_sf${sf}_${c}_s${s}.log" 2>&1
  done; done; done
fi
if has probes; then
  runs=()
  for d in "$RUNS/$EXP/fold$FOLD"/sf*/{raw,res,res_ovl}/seed*; do
    [[ -f $d/metrics.json ]] && runs+=("$d")             # completed runs only
  done
  (( ${#runs[@]} )) && $PY scripts/probes.py --fold "$FOLD" --runs "${runs[@]}" 2>&1 | tee "logs/probes_${EXP}_fold$FOLD.log"
fi
has inject && $PY scripts/inject_recovery.py --fold "$FOLD" 2>&1 | tee "logs/inject_fold$FOLD.log"
has diag && $PY scripts/diagnostics.py --fold "$FOLD" 2>&1 | tee "logs/diag_fold$FOLD.log"
has report && $PY scripts/report.py --exp "$EXP" --fold "$FOLD" 2>&1 | tee "logs/report_${EXP}_fold$FOLD.log"
exit 0
