"""Step 1d - download ONLY the NREL SRRL images that lie within TIME_TOL_MIN of an AERONET record
(at most NREL_MAX_SAMPLES, evenly spread over time) into data/images/YYYY/MM/DD/.
Run AFTER 01_get_aeronet.py.  Needs:  pip install huggingface_hub pyarrow"""
import pandas as pd
import config
from common import read_aeronet


def pick_pairs(aod, imgs, tol_min, max_samples, min_csi=None):
    """aod: AERONET rows (column 'time'); imgs: image list (columns filename, datetime_utc, clear_sky_index)."""
    imgs = imgs.drop_duplicates("filename").sort_values("datetime_utc")
    if min_csi is not None:
        imgs = imgs[imgs["clear_sky_index"] >= min_csi]
    aod = aod.sort_values("time")
    m = pd.merge_asof(aod, imgs, left_on="time", right_on="datetime_utc", direction="nearest",
                      tolerance=pd.Timedelta(minutes=tol_min)).dropna(subset=["filename"])
    m["dt_min"] = (m["time"] - m["datetime_utc"]).abs().dt.total_seconds() / 60
    m = m.sort_values("dt_min").drop_duplicates("filename").sort_values("time").reset_index(drop=True)
    if max_samples and len(m) > max_samples:
        keep = pd.Series(range(len(m))).iloc[[int(round(i)) for i in
                                              pd.Series(range(max_samples)) * (len(m) - 1) / (max_samples - 1)]]
        m = m.iloc[sorted(set(keep))].reset_index(drop=True)
    return m


def load_image_lists(days, repo, meta_dir):
    from huggingface_hub import hf_hub_download
    out, seen = [], set()
    for d in days:
        for k in (-1, 0, 1):                       # a UTC afternoon can sit in the next local-day file
            day = pd.Timestamp(d) + pd.Timedelta(days=k)
            if day in seen:
                continue
            seen.add(day)
            fn = f"parquets/{day:%Y}/{day:%m}/weather_images_{day.year}_{day.dayofyear:03d}.parquet"
            try:
                p = hf_hub_download(repo, fn, repo_type="dataset", local_dir=str(meta_dir))
            except Exception:
                continue
            q = pd.read_parquet(p, columns=["filename", "datetime_utc", "clear_sky_index"])
            q["datetime_utc"] = pd.to_datetime(q["datetime_utc"]).dt.tz_localize(None)
            out.append(q)
    if not out:
        raise SystemExit("No image lists found on Hugging Face for these dates (images start Oct 2025).")
    return pd.concat(out, ignore_index=True)


def main():
    from huggingface_hub import hf_hub_download
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x
    aod = read_aeronet(config.AERONET_CSV).dropna(subset=[f"AOD_{w}nm" for w in config.WAVELENGTHS])
    days = sorted(aod["time"].dt.normalize().unique())
    imgs = load_image_lists(days, config.NREL_REPO, config.DATA_DIR / "nrel_meta")
    pairs = pick_pairs(aod, imgs, config.TIME_TOL_MIN, config.NREL_MAX_SAMPLES, config.NREL_MIN_CLEAR_SKY_INDEX)
    if pairs.empty:
        raise SystemExit("No AERONET record has an image within the time tolerance. Check the AERONET dates.")
    print(f"{len(pairs)} image/AOD pairs selected - downloading ...")
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for fn in tqdm(pairs["filename"]):
        try:
            hf_hub_download(config.NREL_REPO, f"{fn[:4]}/{fn[4:6]}/{fn[6:8]}/{fn}",
                            repo_type="dataset", local_dir=str(config.IMAGE_DIR))
        except Exception as e:
            failed += 1
            print("  failed:", fn, str(e)[:80])
    pairs[["filename", "datetime_utc", "time", "dt_min", "clear_sky_index"]].to_csv(
        config.DATA_DIR / "nrel_selected.csv", index=False)
    print(f"Done. {len(pairs) - failed} images in {config.IMAGE_DIR} (failed: {failed}).")
    print("Next: python show_preprocessing.py, then 02, 03, 04, 05.")


if __name__ == "__main__":
    main()
