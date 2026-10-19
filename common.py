import numpy as np
import pandas as pd
import config


def reg_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    return dict(n=len(y),
                r=float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else np.nan,
                RMSE=float(np.sqrt(np.mean((p - y) ** 2))),
                MAE=float(np.mean(np.abs(p - y))),
                bias=float(np.mean(p - y)))


def time_split(df, frac=None):
    """Chronological split: first `frac` of every month -> train, rest -> test (no shuffling)."""
    frac = config.SPLIT_TRAIN_FRACTION if frac is None else frac
    df = df.sort_values("time").reset_index(drop=True)
    df["split"] = "test"
    for _, g in df.groupby(df["time"].dt.to_period("M")):
        k = int(round(len(g) * frac))
        df.loc[g.index[:k], "split"] = "train"
    return df


def read_aeronet(path):
    """Read an AERONET v3 'all points' AOD file (skips the header lines automatically)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    hdr = next(i for i, l in enumerate(lines) if "Date(dd:mm:yyyy)" in l)
    df = pd.read_csv(path, skiprows=hdr)
    df = df.replace(-999.0, np.nan).replace(-999, np.nan)
    df["time"] = pd.to_datetime(df["Date(dd:mm:yyyy)"] + " " + df["Time(hh:mm:ss)"],
                                format="%d:%m:%Y %H:%M:%S")
    return df
