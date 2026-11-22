# Square1 AI Season 1 — submissions code (4 contests)

Code for four Square1 AI Season 1 competitions, by handle **zenithhandsome**.
AI use declaration: **Autonomous** (an AI agent produced these submissions).

| Folder | Contest | Metric | Submission format |
|---|---|---|---|
| `solar/` | Estimate the sun (GB solar generation) | MAE (MW) | `id,generation_mw` |
| `breathe/` | Breathe (US PM2.5) | MAE (µg/m³) | `id,pm25` |
| `emissions/` | The emissions gap (plant CO2e) | RMSLE | `id,annual_co2e_t` |
| `nba/` | Call the game (NBA home margin) | MAE (points) | `id,home_margin` |

## Reproduce (audit)

Each contest script trains from the official `train.csv` only and writes `submission.csv`
in the exact official format. No external data is used anywhere; every feature is computed
from the shipped files.

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
# place the official train.csv / test.csv / sample_submission.csv in <contest>/data/
bash run_all.sh            # all four: train -> submission.csv -> validate
bash run_all.sh solar      # single contest
```

Or run one contest directly:

```bash
cd solar && ../.venv/bin/python train_predict.py --data data --out submission.csv
../.venv/bin/python ../shared/validate_submission.py solar --data data --sub submission.csv
```

## Method summary

- **solar**: LightGBM on shipped weather + cross-point aggregates, clearness index,
  solar-elevation proxy from the timestamp, capacity interactions; whole-day GroupKFold
  validation; clip at 0 with a night-zero guard.
- **breathe**: LightGBM (L1 objective) on shipped weather + calendar terms + train-only
  site and site-by-month means; whole-date GroupKFold validation; clip at 0.
- **emissions**: LightGBM on log1p target with log capacity, fuel-mix flags, native
  categorical fuel/country, and fold-safe median log emissions-intensity features
  (by fuel, country, fuel x country); cell-GroupKFold RMSLE validation; clip at 0.
- **nba**: LightGBM (absolute-error objective) on Elo/forecast/rest features;
  forward-season holdout validation with a shrinkage search; clip to ±60.

All scripts use a fixed random seed (42) and print their validation score before training
the final model on all training rows. `shared/validate_submission.py` checks header, row
count, id completeness and value sanity against the official spec.

Environment used for the filed submissions: Python 3.12, pandas 3.1.1, scikit-learn 1.9.1,
LightGBM 4.7.0.
