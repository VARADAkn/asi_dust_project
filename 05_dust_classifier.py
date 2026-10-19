"""Step 5 - Stage 2: dust / non-dust classifier.
Labels come from AERONET (high AOD + low Angstrom exponent). Two variants are compared:
  image_only        : image features
  image+pred_AOD    : image features + AOD predicted from the image (out-of-fold on the training set)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import config
from common import time_split

config.RESULTS_DIR.mkdir(exist_ok=True)
df = pd.read_csv(config.DATASET_CSV, parse_dates=["time"])
tgt = f"AOD_{config.STAGE2_WL}nm"
df = df.dropna(subset=[config.DUST_AOD_COL, config.AE_COL, tgt])
df["dust"] = ((df[config.DUST_AOD_COL] >= config.DUST_AOD_MIN) & (df[config.AE_COL] <= config.DUST_AE_MAX)).astype(int)
df = time_split(df)
tr, te = df[df.split == "train"].copy(), df[df.split == "test"].copy()
print(f"Dust samples: train {tr.dust.sum()}/{len(tr)}, test {te.dust.sum()}/{len(te)}")
if tr.dust.sum() < 5 or te.dust.sum() < 3:
    raise SystemExit("Too few dust samples. Relax DUST_AOD_MIN / DUST_AE_MAX in config.py, use more data, "
                     "or report this as a limitation.")

# ---- out-of-fold predicted AOD (so the classifier never sees an AOD predicted from its own training image)
reg = xgb.XGBRegressor(n_estimators=300, max_depth=config.XGB_PARAMS["max_depth"],
                       learning_rate=config.XGB_PARAMS["learning_rate"],
                       subsample=config.XGB_PARAMS["subsample"], random_state=42)
Xtr, Xte = tr[config.FEATURE_NAMES], te[config.FEATURE_NAMES]
tr["AOD_pred"] = cross_val_predict(reg, Xtr, tr[tgt], cv=KFold(5, shuffle=False))
te["AOD_pred"] = reg.fit(Xtr, tr[tgt]).predict(Xte)

variants = {"image_only": config.FEATURE_NAMES,
            "image+pred_AOD": config.FEATURE_NAMES + ["AOD_pred"]}
rows = []
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, cols) in zip(axes, variants.items()):
    spw = (tr.dust == 0).sum() / max((tr.dust == 1).sum(), 1)
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=config.XGB_PARAMS["learning_rate"],
                            subsample=config.XGB_PARAMS["subsample"], scale_pos_weight=spw,
                            eval_metric="logloss", random_state=42)
    clf.fit(tr[cols], tr.dust)
    pred = clf.predict(te[cols])
    p, r, f, _ = precision_recall_fscore_support(te.dust, pred, average="binary", zero_division=0)
    rows.append(dict(variant=name, accuracy=accuracy_score(te.dust, pred), precision=p, recall=r, F1=f))
    ConfusionMatrixDisplay(confusion_matrix(te.dust, pred, labels=[0, 1]),
                           display_labels=["non-dust", "dust"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name}  F1={f:.2f}")
    clf.save_model(str(config.RESULTS_DIR / f"xgb_dust_{name.replace('+', '_')}.json"))

fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "dust_confusion.png", dpi=150)
res = pd.DataFrame(rows).round(3)
res.to_csv(config.RESULTS_DIR / "dust_metrics.csv", index=False)
print(res.to_string(index=False))
