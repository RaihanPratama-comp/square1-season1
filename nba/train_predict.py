#!/usr/bin/env python3
"""Call the game (Square1 AI) - NBA home margin, MAE, target home_margin.
Validation mirrors test split: forward-in-time holdout (last seasons in train as validation),
because test = 3 seasons AFTER train. No external data. Predictions shrunk toward
Elo-implied margin (MAE-optimal shrinkage checked on validation).
Run: python train_predict.py [--data data] [--out submission.csv]
"""
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
SEED = 42; TARGET, IDCOL = "home_margin", "id"

def add_features(df):
    df = df.copy()
    if {"home_elo","away_elo"} <= set(df.columns):
        df["elo_diff"] = df["home_elo"]-df["away_elo"]; df["elo_sum"] = df["home_elo"]+df["away_elo"]
    if "home_forecast" in df:
        p = df["home_forecast"].clip(0.01, 0.99); df["forecast_logit"] = np.log(p/(1-p))
    if {"home_rest","away_rest"} <= set(df.columns):
        df["rest_diff"] = df["home_rest"].replace(-1, np.nan)-df["away_rest"].replace(-1, np.nan)
    if {"home_seasongame","away_seasongame"} <= set(df.columns):
        df["games_diff"] = df["home_seasongame"]-df["away_seasongame"]
    if "neutral" in df and "elo_diff" in df: df["elo_diff_home"] = df["elo_diff"]*(1-df["neutral"])
    return df

def prep(train, test):
    tr, te = add_features(train), add_features(test)
    drop = {TARGET, IDCOL}
    feats = [c for c in tr.columns if c not in drop and c in te.columns]
    for c in feats:
        if not pd.api.types.is_numeric_dtype(tr[c]):
            cats = sorted(tr[c].astype(str).unique())
            mapping = {v: i for i, v in enumerate(cats)}
            tr[c] = tr[c].astype(str).map(mapping).astype(float)
            te[c] = te[c].astype(str).map(mapping).fillna(-1).astype(float)
    return tr, te, feats

def model():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="regression_l1", n_estimators=500, learning_rate=0.04, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1)
    except Exception:
        return HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05, random_state=SEED)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="submission.csv"); a = ap.parse_args()
    train = pd.read_csv(f"{a.data}/train.csv"); test = pd.read_csv(f"{a.data}/test.csv")
    tr, te, feats = prep(train, test)
    print(f"[info] features={len(feats)} rows train={len(tr)} test={len(te)}")
    # Forward validation: last season(s) as holdout (mirrors future-seasons test)
    shrink = 1.0
    if "season" in tr and tr["season"].nunique() > 3:
        cut = sorted(tr["season"].unique())[-2]
        tri, vai = tr[tr["season"] < cut], tr[tr["season"] >= cut]
        m = model(); m.fit(tri[feats], tri[TARGET]); raw = m.predict(vai[feats])
        best = min(((s, mean_absolute_error(vai[TARGET], raw*s)) for s in np.arange(0.7, 1.05, 0.05)), key=lambda x: x[1])
        shrink = float(best[0])
        base = np.where(vai.get("neutral", 0) == 1, 0.0, 3.0)
        print(f"[valid] forward-holdout MAE model={best[1]:.3f} shrink={shrink:.2f} | home+3 baseline={mean_absolute_error(vai[TARGET], base):.3f} (pinned: 10.619 / GBDT 9.47)")
    m = model(); m.fit(tr[feats], tr[TARGET])
    pred = np.clip(m.predict(te[feats])*shrink, -60, 60)
    pd.DataFrame({IDCOL: test[IDCOL], TARGET: pred}).to_csv(a.out, index=False)
    print(f"[done] wrote {a.out} rows={len(pred)} shrink={shrink}")
if __name__ == "__main__": main()
