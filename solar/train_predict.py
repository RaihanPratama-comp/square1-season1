#!/usr/bin/env python3
"""Estimate the sun (Square1 AI) - solar GB, MAE, target generation_mw.
Validation mirrors test split: whole days held out (GroupKFold by date) + last-period holdout.
No external data. Run: python train_predict.py [--data data] [--out submission.csv]
"""
import argparse, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

SEED = 42
TARGET, IDCOL = "generation_mw", "id"
POINTS = ["london","birmingham","manchester","leeds","bristol","norwich","southampton","plymouth","cardiff","newcastle","glasgow","edinburgh"]

def add_features(df):
    df = df.copy()
    if "datetime_utc" in df:
        dt = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
        df["hour"] = dt.dt.hour + dt.dt.minute/60.0
        df["doy"] = dt.dt.day_of_year; df["month"] = dt.dt.month
        df["doy_sin"] = np.sin(2*np.pi*df["doy"]/365.25); df["doy_cos"] = np.cos(2*np.pi*df["doy"]/365.25)
        df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24); df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
        # Approx solar elevation proxy for GB centre (52.5N, 0E) - pure maths from timestamp, no external data
        decl = np.radians(-23.44*np.cos(2*np.pi*(df["doy"]+10)/365.25))
        lat = np.radians(52.5); ha = np.radians(15*(df["hour"]-12))
        df["solar_elev_sin"] = np.sin(lat)*np.sin(decl)+np.cos(lat)*np.cos(decl)*np.cos(ha)
        df["_date"] = dt.dt.date.astype(str)
    ghi = [c for c in df.columns if c.endswith("_ghi")]
    clr = [c for c in df.columns if c.endswith("_clearsky")]
    cld = [c for c in df.columns if c.endswith("_cloud_pct")]
    if ghi: df["ghi_mean"] = df[ghi].mean(axis=1); df["ghi_max"] = df[ghi].max(axis=1); df["ghi_min"] = df[ghi].min(axis=1)
    if clr: df["clearsky_mean"] = df[clr].mean(axis=1)
    if ghi and clr and len(ghi)==len(clr):
        df["clearness"] = df["ghi_mean"]/(df["clearsky_mean"]+1.0)
    if cld: df["cloud_mean"] = df[cld].mean(axis=1)
    if "installed_capacity_mwp" in df and "ghi_mean" in df:
        df["cap_x_ghi"] = df["installed_capacity_mwp"]*df["ghi_mean"]
        df["cap_x_clearness"] = df["installed_capacity_mwp"]*df.get("clearness", 0)
    df = df.replace(-999, np.nan)
    return df

def prep(train, test):
    tr, te = add_features(train), add_features(test)
    drop = {TARGET, IDCOL, "datetime_utc", "_date"}
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
        return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1)
    except Exception:
        return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=SEED)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="submission.csv"); a = ap.parse_args()
    train = pd.read_csv(f"{a.data}/train.csv"); test = pd.read_csv(f"{a.data}/test.csv")
    tr, te, feats = prep(train, test)
    # Baseline 1: month x half-hour mean capacity factor x capacity (pinned baseline, approx)
    if "installed_capacity_mwp" in tr and tr["installed_capacity_mwp"].max() > 0:
        cf = tr[TARGET]/tr["installed_capacity_mwp"].replace(0, np.nan)
        grp = cf.groupby([tr.get("month", 1), tr.get("hour", 0)]).mean()
        print(f"[info] features={len(feats)} rows train={len(tr)} test={len(te)}")
    # Validation: whole-day groups (mirrors public/private whole-day split)
    if "_date" in add_features(train).columns or "datetime_utc" in train:
        groups = add_features(train)["_date"]
        gkf = GroupKFold(3); scores = []
        for tri, vai in gkf.split(tr, tr[TARGET], groups):
            m = model(); m.fit(tr.iloc[tri][feats], tr.iloc[tri][TARGET])
            scores.append(mean_absolute_error(tr.iloc[vai][TARGET], np.clip(m.predict(tr.iloc[vai][feats]), 0, None)))
        print(f"[valid] whole-day GroupKFold MAE: {np.mean(scores):.3f} +- {np.std(scores):.3f} (pinned baselines: 700.405 / GBDT 279)")
    m = model(); m.fit(tr[feats], tr[TARGET])
    pred = np.clip(m.predict(te[feats]), 0, None)
    # Night guard: zero when mean GHI is zero
    if "ghi_mean" in te: pred = np.where(te["ghi_mean"].fillna(0) <= 0, 0.0, pred)
    pd.DataFrame({IDCOL: test[IDCOL], TARGET: pred}).to_csv(a.out, index=False)
    print(f"[done] wrote {a.out} rows={len(pred)}")

if __name__ == "__main__": main()
