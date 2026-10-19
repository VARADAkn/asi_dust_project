"""Step 3 - match each image with the nearest AERONET reading (within TIME_TOL_MIN) -> data/dataset.csv"""
import pandas as pd
import config
from common import read_aeronet

feat = pd.read_csv(config.FEATURES_CSV, parse_dates=["time"]).sort_values("time")
aod = read_aeronet(config.AERONET_CSV).sort_values("time")
cols = ["time", config.AE_COL] + [f"AOD_{w}nm" for w in config.WAVELENGTHS]
missing = [c for c in cols if c not in aod.columns]
if missing:
    raise SystemExit(f"Columns not found in AERONET file: {missing}\nAvailable: {list(aod.columns)}")
aod = aod[cols].copy()
aod["aeronet_time"] = aod["time"]

m = pd.merge_asof(feat, aod, on="time", direction="nearest",
                  tolerance=pd.Timedelta(minutes=config.TIME_TOL_MIN))
m = m.dropna(subset=["aeronet_time"])
m["dt_min"] = (m["time"] - m["aeronet_time"]).dt.total_seconds().abs() / 60
m.to_csv(config.DATASET_CSV, index=False)
print(f"{len(feat)} images -> {len(m)} matched samples saved to {config.DATASET_CSV}")
print(m[[f"AOD_{w}nm" for w in config.WAVELENGTHS] + [config.AE_COL]].describe().round(3))
