#!/usr/bin/env python3
"""The emissions gap (Square1 AI) - CO2e power plant, RMSLE, target annual_co2e_t.
Target transform log1p (matches RMSLE). Validation mirrors test split: GroupKFold by
country x 0.5deg cell (plants/cells never straddle train/test). No external data.
Emissions-intensity features (median log tonnes-per-MW by fuel / country / fuel x country)
are computed on training rows only - inside each CV fold for validation, on the full
training file for the final model - so no test target is ever touched.
Run: python train_predict.py [--data data] [--out submission.csv]
"""
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
SEED = 42; TARGET, IDCOL = "annual_co2e_t", "id"

def add_features(df):
    df = df.copy()
    if "capacity_mw" in df: df["log_capacity"] = np.log1p(df["capacity_mw"].clip(lower=0))
    if {"lat_r","lon_r"} <= set(df.columns):
        df["cell"] = df["country"].astype(str) + "_" + df["lat_r"].astype(str) + "_" + df["lon_r"].astype(str) if "country" in df else df["lat_r"].astype(str)+"_"+df["lon_r"].astype(str)
    if "fuels" in df:
        for f in ["coal","gas","oil","biomass","waste"]:
            df[f"fuel_{f}"] = df["fuels"].astype(str).str.lower().str.contains(f).astype(int)
    return df

def prep(train, test):
    tr, te = add_features(train), add_features(test)
    drop = {TARGET, IDCOL, "plant", "fuels", "cell"}
    feats = [c for c in tr.columns if c not in drop and c in te.columns]
    for c in feats:
        if not pd.api.types.is_numeric_dtype(tr[c]):
            cats = sorted(tr[c].astype(str).unique())
            mapping = {v: i for i, v in enumerate(cats)}
            tr[c] = tr[c].astype(str).map(mapping).astype(float)
            te[c] = te[c].astype(str).map(mapping).fillna(-1).astype(float)
    return tr, te, feats

INTENSITY_FEATS = ["fuel_log_intensity", "country_log_intensity", "fuel_country_log_intensity", "est_log_emissions"]

def add_intensity(fit_df, apply_df):
    """Median log1p(tonnes per MW) by fuel / country / fuel x country, fit on fit_df only."""
    out = apply_df.copy()
    cap = fit_df["capacity_mw"].clip(lower=1e-9)
    log_int = np.log1p(fit_df[TARGET].clip(lower=0) / cap)
    tmp = pd.DataFrame({"fuel": fit_df["primary_fuel"], "country": fit_df["country"], "li": log_int})
    glob = float(log_int.median())
    fuel_med = tmp.groupby("fuel")["li"].median()
    country_med = tmp.groupby("country")["li"].median()
    fc = tmp.groupby(["fuel", "country"])["li"].agg(["median", "count"])
    out["fuel_log_intensity"] = out["primary_fuel"].map(fuel_med).astype(float).fillna(glob)
    out["country_log_intensity"] = out["country"].map(country_med).astype(float).fillna(glob)
    key = list(zip(out["primary_fuel"], out["country"]))
    fc_med = pd.Series(key).map(fc["median"]).to_numpy()
    fc_cnt = pd.Series(key).map(fc["count"]).fillna(0).to_numpy()
    # shrink sparse fuel x country cells toward the fuel median (weight n/(n+5))
    w = fc_cnt / (fc_cnt + 5.0)
    out["fuel_country_log_intensity"] = np.where(np.isnan(fc_med), out["fuel_log_intensity"],
                                                 w * np.nan_to_num(fc_med) + (1 - w) * out["fuel_log_intensity"])
    out["est_log_emissions"] = out["log_capacity"] + out["fuel_country_log_intensity"]
    return out

def rmsle(y, p): return float(np.sqrt(np.mean((np.log1p(np.clip(p,0,None))-np.log1p(np.clip(y,0,None)))**2)))
CAT_COLS = ["primary_fuel", "country"]

def as_categorical(df, categories):
    out = df.copy()
    for c in CAT_COLS:
        out[c] = pd.Categorical(out[c].astype(int), categories=categories[c])
    return out

def model():
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=900, learning_rate=0.04, num_leaves=127, subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbose=-1)
    except Exception:
        return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=SEED)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="submission.csv"); a = ap.parse_args()
    train = pd.read_csv(f"{a.data}/train.csv"); test = pd.read_csv(f"{a.data}/test.csv")
    tr, te, feats = prep(train, test)
    categories = {c: sorted(tr[c].astype(int).unique()) for c in CAT_COLS}
    tr = as_categorical(tr, categories); te = as_categorical(te, categories)
    ylog = np.log1p(tr[TARGET].clip(lower=0))
    all_feats = feats + INTENSITY_FEATS
    print(f"[info] features={len(all_feats)} rows train={len(tr)} test={len(te)} zeros_in_target={(tr[TARGET]==0).mean():.3f}")
    groups = add_features(train).get("cell")
    if groups is not None:
        gkf = GroupKFold(3); scores = []
        for tri, vai in gkf.split(tr, ylog, groups):
            tr_f = add_intensity(tr.iloc[tri], tr.iloc[tri])
            va_f = add_intensity(tr.iloc[tri], tr.iloc[vai])
            m = model(); m.fit(tr_f[all_feats], ylog.iloc[tri])
            scores.append(rmsle(tr.iloc[vai][TARGET], np.expm1(m.predict(va_f[all_feats]))))
        print(f"[valid] cell-GroupKFold RMSLE: {np.mean(scores):.3f} +- {np.std(scores):.3f} (pinned baselines: 1.173 / GBDT-log 0.946)")
    tr_full = add_intensity(tr, tr); te_full = add_intensity(tr, te)
    m = model(); m.fit(tr_full[all_feats], ylog)
    pred = np.clip(np.expm1(m.predict(te_full[all_feats])), 0, None)
    pd.DataFrame({IDCOL: test[IDCOL], TARGET: pred}).to_csv(a.out, index=False)
    print(f"[done] wrote {a.out} rows={len(pred)}")

if __name__ == "__main__": main()
