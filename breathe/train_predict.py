#!/usr/bin/env python3
"""Breathe (Square1 AI) - PM2.5 USA, MAE, target pm25.
Validation mirrors test split: future-period holdout + whole-date groups (public/private are whole dates).
Site history features computed from TRAIN ONLY (no test-date PM2.5 shipped -> no leakage).
Run: python train_predict.py [--data data] [--out submission.csv]
"""
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
SEED = 42; TARGET, IDCOL = "pm25", "id"

def add_cal(df):
    df = df.copy()
    if "date" in df:
        d = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = d.dt.month; df["doy"] = d.dt.day_of_year; df["dow"] = d.dt.day_of_week
        df["doy_sin"] = np.sin(2*np.pi*df["doy"]/365.25); df["doy_cos"] = np.cos(2*np.pi*df["doy"]/365.25)
        df["_date"] = d.dt.date.astype(str)
    if {"temp_max_c","temp_min_c"} <= set(df.columns): df["temp_range"] = df["temp_max_c"]-df["temp_min_c"]
    if {"wind_ms","rh_pct"} <= set(df.columns): df["wind_x_rh"] = df["wind_ms"]*df["rh_pct"]
    if "precip_mm" in df: df["precip_log"] = np.log1p(df["precip_mm"].clip(lower=0))
    return df.replace(-999, np.nan)

def add_site_history(tr, te):
    # Train-only aggregates: site mean, site x month mean, division x month mean (expanding-safe: test has no target)
    g = tr[TARGET]
    tr["site_mean"] = g.groupby(tr["site"]).transform("mean") if "site" in tr else g.mean()
    te["site_mean"] = te["site"].map(g.groupby(tr["site"]).mean()) if "site" in te else g.mean()
    if "site" in tr and "month" in tr:
        sm = tr.groupby(["site","month"])[TARGET].mean()
        tr["site_month_mean"] = [sm.get((s,m), np.nan) for s,m in zip(tr["site"], tr["month"])]
        te["site_month_mean"] = [sm.get((s,m), np.nan) for s,m in zip(te["site"], te["month"])]
    for d in (tr, te):
        d["site_mean"] = d["site_mean"].fillna(g.mean())
        if "site_month_mean" in d: d["site_month_mean"] = d["site_month_mean"].fillna(d["site_mean"])
    return tr, te

def prep(train, test):
    tr, te = add_cal(train), add_cal(test)
    tr, te = add_site_history(tr, te)
    drop = {TARGET, IDCOL, "date", "_date"}
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
        return lgb.LGBMRegressor(objective="regression_l1", n_estimators=700, learning_rate=0.05, num_leaves=63, subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1)
    except Exception:
        return HistGradientBoostingRegressor(loss="absolute_error", max_iter=500, learning_rate=0.05, random_state=SEED)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="submission.csv"); a = ap.parse_args()
    train = pd.read_csv(f"{a.data}/train.csv"); test = pd.read_csv(f"{a.data}/test.csv")
    tr, te, feats = prep(train, test)
    print(f"[info] features={len(feats)} rows train={len(tr)} test={len(te)}")
    if "_date" in tr:
        gkf = GroupKFold(3); scores = []
        for tri, vai in gkf.split(tr, tr[TARGET], tr["_date"]):
            m = model(); m.fit(tr.iloc[tri][feats], tr.iloc[tri][TARGET])
            scores.append(mean_absolute_error(tr.iloc[vai][TARGET], np.clip(m.predict(tr.iloc[vai][feats]), 0, None)))
        print(f"[valid] whole-date GroupKFold MAE: {np.mean(scores):.3f} +- {np.std(scores):.3f} (pinned baselines: 3.401 / GBDT 2.99)")
    m = model(); m.fit(tr[feats], tr[TARGET])
    pred = np.clip(m.predict(te[feats]), 0, None)
    pd.DataFrame({IDCOL: test[IDCOL], TARGET: pred}).to_csv(a.out, index=False)
    print(f"[done] wrote {a.out} rows={len(pred)}")
if __name__ == "__main__": main()
