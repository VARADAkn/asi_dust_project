"""Step 4 - Stage 1: XGBoost AOD regression (one model per wavelength) + baselines + plots."""
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import config
from common import reg_metrics, time_split

config.RESULTS_DIR.mkdir(exist_ok=True)
df = time_split(pd.read_csv(config.DATASET_CSV, parse_dates=["time"]))
print("Split sizes:", df["split"].value_counts().to_dict())

baselines = {
    "LinearRegression": lambda: make_pipeline(SimpleImputer(strategy="median"), LinearRegression()),
    "DecisionTree": lambda: make_pipeline(SimpleImputer(strategy="median"), DecisionTreeRegressor(max_depth=8, random_state=0)),
    "RandomForest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(200, random_state=0, n_jobs=-1)),
}

rows, preds = [], []
fig, axes = plt.subplots(1, len(config.WAVELENGTHS), figsize=(4.2 * len(config.WAVELENGTHS), 4))
for ax, wl in zip(np.atleast_1d(axes), config.WAVELENGTHS):
    tgt = f"AOD_{wl}nm"
    d = df.dropna(subset=[tgt])
    tr, te = d[d.split == "train"], d[d.split == "test"]
    if len(tr) < 20 or len(te) < 5:
        print(f"{tgt}: not enough samples (train={len(tr)}, test={len(te)}), skipped")
        continue
    Xtr, ytr, Xte, yte = tr[config.FEATURE_NAMES], tr[tgt], te[config.FEATURE_NAMES], te[tgt]

    params = dict(config.XGB_PARAMS, objective="reg:squarederror", eval_metric="rmse")
    dtr = xgb.DMatrix(Xtr, label=ytr)
    cv = xgb.cv(params, dtr, num_boost_round=config.MAX_ROUNDS,
                nfold=min(config.CV_FOLDS, max(2, len(tr) // 5)),
                early_stopping_rounds=config.EARLY_STOP, seed=config.XGB_PARAMS["seed"])
    n_rounds = len(cv)
    model = xgb.train(params, dtr, num_boost_round=n_rounds)
    p = model.predict(xgb.DMatrix(Xte))
    model.save_model(str(config.RESULTS_DIR / f"xgb_aod_{wl}nm.json"))

    m = reg_metrics(yte, p)
    rows.append(dict(wavelength=wl, model="XGBoost", rounds=n_rounds, **m))
    preds.append(pd.DataFrame(dict(time=te["time"].values, wavelength=wl, aeronet=yte.values, predicted=p)))
    for name, mk in baselines.items():
        b = mk().fit(Xtr, ytr)
        rows.append(dict(wavelength=wl, model=name, rounds=np.nan, **reg_metrics(yte, b.predict(Xte))))

    ax.scatter(yte, p, s=8, alpha=0.5)
    lim = [0, max(yte.max(), p.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    a, b0 = np.polyfit(yte, p, 1)
    ax.plot(lim, [a * x + b0 for x in lim], "r", lw=1, label=f"fit y={a:.2f}x+{b0:.2f}")
    ax.set_title(f"AOD {wl} nm  r={m['r']:.2f} RMSE={m['RMSE']:.3f}")
    ax.set_xlabel("AERONET AOD"); ax.set_ylabel("Predicted AOD"); ax.legend(fontsize=7)

    if wl == config.STAGE2_WL:
        imp = pd.Series(model.get_score(importance_type="gain")).sort_values().tail(15)
        plt.figure(figsize=(6, 5)); imp.plot.barh(); plt.title(f"XGBoost feature importance (gain), {wl} nm")
        plt.tight_layout(); plt.savefig(config.RESULTS_DIR / "feature_importance.png", dpi=150); plt.close()

fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "aod_scatter.png", dpi=150)
res = pd.DataFrame(rows).round(4)
res.to_csv(config.RESULTS_DIR / "aod_metrics.csv", index=False)
if preds:
    pd.concat(preds).to_csv(config.RESULTS_DIR / "aod_test_predictions.csv", index=False)
print(res.to_string(index=False))
