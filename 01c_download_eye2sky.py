"""Download Eye2Sky all-sky image zips (one zip = one station, one day) and unpack them SAFELY to data/images/<site>/<date>/.

Why not the official bash script? It increments the date twice (you get the wrong days) and needs GNU `date`.
Why unpack here? Windows' built-in unzip fails when the full path is too long or a name has odd characters
("destination file could not be created"). This script flattens the folders, cleans the names and keeps only every n-th image.

Settings: see the EYE2SKY block at the end of config.py.  Run:  python 01c_download_eye2sky.py
"""
import re
import shutil
import sys
import time
import zipfile
from datetime import date, datetime, timedelta

import requests
import config

IMG_EXT = (".jpg", ".jpeg", ".png")


def days(start, end):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    return [d0 + timedelta(n) for n in range((d1 - d0).days + 1)]


def download(url, dest, tries=3):
    for k in range(tries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    return "missing"
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dest)
                return "ok"
        except requests.RequestException as e:
            print(f"   attempt {k + 1} failed: {e}")
            time.sleep(3)
    return "failed"


def unpack(zpath, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            raise zipfile.BadZipFile(f"corrupt entry {bad} in {zpath} - delete the zip and download again")
        names = sorted(n for n in z.namelist() if n.lower().endswith(IMG_EXT) and not n.endswith("/"))
        keep = names[::config.E2S_EXTRACT_EVERY]
        used = set()
        for n in keep:
            base = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", n.replace("\\", "/").split("/")[-1])
            if base in used:                                  # same file name in two sub-folders
                base = f"{len(used)}_{base}"
            used.add(base)
            with z.open(n) as src, open(out_dir / base, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return len(names), len(keep)


config.E2S_ZIP_DIR.mkdir(parents=True, exist_ok=True)
summary = []
for site in config.E2S_SITES:
    for d in days(config.E2S_START, config.E2S_END):
        name = f"ASI_{d:%Y%m%d}_{site}.zip"
        url = f"{config.E2S_URL}/{d:%Y/%m/%d}/{name}"
        zpath = config.E2S_ZIP_DIR / name
        if not zpath.exists():
            print("Downloading", url)
            status = download(url, zpath)
            if status != "ok":
                print(f"   -> {status} (check the station code and that the date is inside its coverage period)")
                summary.append((site, d, status))
                continue
        out = config.IMAGE_DIR / site / f"{d:%Y%m%d}"
        try:
            total, kept = unpack(zpath, out)
        except zipfile.BadZipFile as e:
            print("   ->", e)
            summary.append((site, d, "bad zip"))
            continue
        print(f"   unpacked {kept} of {total} images -> {out}")
        summary.append((site, d, f"{kept} images"))
        if config.E2S_DELETE_ZIP:
            zpath.unlink()

print("\nSummary:")
for s, d, st in summary:
    print(f"  {s} {d}: {st}")
first = next(iter(sorted(config.IMAGE_DIR.rglob("*.jp*g"))), None)
if first:
    print("\nFirst image file name:", first.name)
    print("Compare it with TIME_REGEX in config.py - the date (8 digits) and time (6 digits) must be found.")
