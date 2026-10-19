"""Solar position (zenith, azimuth). Uses pvlib if installed, otherwise a NOAA-style approximation
(accurate to a fraction of a degree, enough for this project)."""
import numpy as np
import pandas as pd


def solar_position(times_utc, lat, lon, alt=0.0):
    """times_utc: tz-naive UTC timestamps. Returns DataFrame with columns zenith, azimuth (degrees)."""
    t = pd.DatetimeIndex(times_utc)
    try:
        from pvlib import solarposition
        sp = solarposition.get_solarposition(t.tz_localize("UTC"), lat, lon, altitude=alt)
        return pd.DataFrame({"zenith": sp["apparent_zenith"].values, "azimuth": sp["azimuth"].values})
    except ImportError:
        pass
    doy = t.dayofyear.values
    hour = t.hour.values + t.minute.values / 60 + t.second.values / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = np.deg2rad(tst / 4 - 180)
    la = np.deg2rad(lat)
    cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    zen = np.degrees(np.arccos(np.clip(cz, -1, 1)))
    az = (np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(dec) * np.cos(la))) + 180) % 360
    return pd.DataFrame({"zenith": zen, "azimuth": az})
