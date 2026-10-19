"""Step 2 - preprocess every sky image and extract features -> data/features.csv"""
import re
from datetime import datetime, timedelta
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import config
from preprocessing import load_rgb, preprocess
from features import extract_features, add_solar_zenith

pat = re.compile(config.TIME_REGEX)


def parse_time(name):
    m = pat.search(name)
    return datetime.strptime("".join(m.groups()), config.TIME_FORMAT) if m else None


files = sorted(p for p in config.IMAGE_DIR.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
files = files[::config.SUBSAMPLE_EVERY]
print(f"{len(files)} images selected")

rows, skipped = [], 0
for p in tqdm(files):
    t = parse_time(p.name)
    if t is not None:
        t = t - timedelta(hours=config.TIME_OFFSET_HOURS)
    rgb = load_rgb(p)
    if t is None or rgb is None:
        skipped += 1
        continue
    sky, geom, info = preprocess(rgb)
    if info["valid_frac"] < config.MIN_VALID_FRAC:
        skipped += 1
        continue
    row = dict(time=t, file=str(p), valid_frac=info["valid_frac"], threshold=info["threshold"])
    row.update(extract_features(rgb, sky, geom))
    rows.append(row)

if not rows:
    raise SystemExit("No usable images. Check TIME_REGEX / TIME_FORMAT in config.py against a real file name.")
df = add_solar_zenith(pd.DataFrame(rows))
df = df[df["SZA"] < 85]            # daytime only
if df.empty:
    raise SystemExit("All images are at night (SZA >= 85). Check SITE_LAT / SITE_LON and TIME_OFFSET_HOURS in config.py "
                     "- file-name times must be UTC.")
df.to_csv(config.FEATURES_CSV, index=False)
print(f"Saved {len(df)} rows to {config.FEATURES_CSV} (skipped {skipped})")
