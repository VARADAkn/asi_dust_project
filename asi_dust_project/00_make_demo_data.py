"""OPTIONAL - create synthetic sky images + a fake AERONET file so you can test the whole pipeline
before you have real data. The images are NOT real; results from them mean nothing scientifically."""
import numpy as np, pandas as pd, cv2
from solar import solar_position
import config

rng = np.random.default_rng(0)
config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
S, R = 400, 190
yy, xx = np.mgrid[:S, :S]
rr = np.hypot(xx - S / 2, yy - S / 2)
inside = rr <= R

days = list(pd.date_range("2024-01-02", periods=18, freq="D")) + list(pd.date_range("2024-02-02", periods=18, freq="D"))
aer_rows = []
for day in days:
    aod440 = float(np.clip(rng.lognormal(-1.8, 0.7), 0.03, 1.2))
    ae = float(rng.uniform(0.2, 1.8))
    if rng.random() < 0.25:                      # a "dust-like" day: more aerosol, coarse particles
        aod440, ae = aod440 + 0.3, rng.uniform(0.1, 0.5)
    noon = round((12 - config.SITE_LON / 15) * 4) / 4        # local solar noon in UTC hours (daytime at ANY site)
    times = pd.date_range(day + pd.Timedelta(hours=noon - 2.5), day + pd.Timedelta(hours=noon + 2.5), freq="15min")
    sp = solar_position(times, config.SITE_LAT, config.SITE_LON)
    for t, zen, azi in zip(times, sp["zenith"], sp["azimuth"]):
        a440 = aod440 * (1 + 0.05 * rng.normal())
        zn = rr / R
        base = np.stack([60 + 90 * zn, 110 + 70 * zn, 210 - 20 * zn], -1)
        haze = a440 * 120 * (0.4 + zn)[..., None] * np.array([0.9, 0.6, -0.5])
        img = base + haze
        ang = np.deg2rad(azi - 90)
        sx, sy = S / 2 + (min(zen, 89) / 90) * R * np.cos(ang), S / 2 + (min(zen, 89) / 90) * R * np.sin(ang)
        sun = np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 18 ** 2)))[..., None]
        img = img + 255 * sun
        if rng.random() < 0.4:                    # a cloud blob
            cx, cy = rng.uniform(100, 300, 2)
            blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 30 ** 2)))[..., None]
            img = img * (1 - blob) + 240 * blob
        img = img + rng.normal(0, 3, img.shape)
        img[~inside] = 0
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(str(config.IMAGE_DIR / f"IMG_{t:%Y%m%d_%H%M%S}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if t.minute % 15 == 0:
            row = {"Date(dd:mm:yyyy)": t.strftime("%d:%m:%Y"), "Time(hh:mm:ss)": t.strftime("%H:%M:%S"),
                   config.AE_COL: ae}
            for w in config.WAVELENGTHS:
                row[f"AOD_{w}nm"] = a440 * (w / 440) ** (-ae)
            aer_rows.append(row)
aer = pd.DataFrame(aer_rows)
with open(config.AERONET_CSV, "w") as f:
    f.write("AERONET Version 3 (synthetic demo)\nDemo\nDemo\nDemo\nDemo\nDemo\n")
    aer.to_csv(f, index=False)
print(f"Wrote {len(list(config.IMAGE_DIR.glob('*.jpg')))} images and {len(aer)} fake AERONET rows")
