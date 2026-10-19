"""Save before / sky-mask / after pictures for a few images -> results/preprocessing_check.png
Run this first on your real images and tune config.py (SUN_MASK_FRAC, MASK_PATH, CIRCLE ...) until it looks right."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from preprocessing import load_rgb, preprocess

files = sorted(p for p in config.IMAGE_DIR.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
pick = files[:: max(1, len(files) // n)][:n]
fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
ax = np.atleast_2d(ax)
for row, p in zip(ax, pick):
    rgb = load_rgb(p)
    sky, (cx, cy, r), info = preprocess(rgb)
    out = rgb.copy(); out[~sky] = 0
    row[0].imshow(rgb); row[0].set_title(p.name, fontsize=7)
    row[1].imshow(sky, cmap="gray"); row[1].set_title(f"sky mask ({info['valid_frac']:.0%} of disc)", fontsize=7)
    row[2].imshow(out); row[2].set_title(f"after masking, thr={info['threshold']:.0f}", fontsize=7)
    for a in row: a.axis("off")
fig.tight_layout(); config.RESULTS_DIR.mkdir(exist_ok=True)
fig.savefig(config.RESULTS_DIR / "preprocessing_check.png", dpi=110)
print("saved", config.RESULTS_DIR / "preprocessing_check.png")
