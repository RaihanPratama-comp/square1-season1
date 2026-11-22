#!/usr/bin/env bash
# Square1 AI Season 1 — run all 4 pipelines + validate outputs.
# Prereq: official data in <contest>/data/ (train.csv, test.csv, sample_submission.csv).
# Usage: bash run_all.sh [contest ...]   (default: all four)
set -u
cd "$(dirname "$0")"
PY="./.venv/bin/python"
CONTESTS=("$@")
[ ${#CONTESTS[@]} -eq 0 ] && CONTESTS=(solar breathe emissions nba)

fail=0
for c in "${CONTESTS[@]}"; do
  echo "================ $c ================"
  for f in train.csv test.csv sample_submission.csv; do
    if [ ! -f "$c/data/$f" ]; then
      echo "[SKIP] $c: missing $c/data/$f — download official data first (see NOTES.md blocker)"; fail=1; continue 2
    fi
  done
  ( cd "$c" && "../$PY" train_predict.py --data data --out submission.csv ) || { echo "[FAIL] $c pipeline"; fail=1; continue; }
  "$PY" shared/validate_submission.py "$c" --data "$c/data" --sub "$c/submission.csv" || { echo "[FAIL] $c validation"; fail=1; continue; }
  echo "[OK] $c -> $c/submission.csv ready to upload"
done
exit $fail
