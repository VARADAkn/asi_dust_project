"""Image features: global RGB means, RBR, brightness spread, ring (almucantar-like) samples."""
import numpy as np
import pandas as pd
import config


def extract_features(rgb, sky, geom):
    cx, cy, r = geom
    px = rgb[sky].astype(np.float64)
    f = {}
    f["R_mean"], f["G_mean"], f["B_mean"] = px.mean(axis=0)
    f["RBR"] = f["R_mean"] / max(f["B_mean"], 1e-6)
    gray = px @ np.array([0.299, 0.587, 0.114])
    f["gray_std"] = gray.std()

    h, w = sky.shape
    half = config.PATCH // 2
    ang = np.linspace(0, 2 * np.pi, config.N_ANNULUS, endpoint=False)
    rad = config.ANNULUS_RADIUS_FRAC * r
    for i, a in enumerate(ang):
        x, y = int(round(cx + rad * np.cos(a))), int(round(cy + rad * np.sin(a)))
        y0, y1, x0, x1 = max(y - half, 0), min(y + half + 1, h), max(x - half, 0), min(x + half + 1, w)
        m = sky[y0:y1, x0:x1]
        if m.sum() == 0:
            vals = (np.nan, np.nan, np.nan)      # ring point is masked -> missing (XGBoost handles NaN)
        else:
            vals = rgb[y0:y1, x0:x1][m].mean(axis=0)
        for c, v in zip("RGB", vals):
            f[f"ann{i}_{c}"] = v
    return f


def add_solar_zenith(df):
    """Solar zenith angle from time stamp (UTC) and site location."""
    from solar import solar_position
    df = df.copy()
    df["SZA"] = solar_position(df["time"], config.SITE_LAT, config.SITE_LON, config.SITE_ALT)["zenith"].values
    return df
