"""Three preprocessing steps: fixed mask, sun mask, cloud / glare mask."""
import cv2
import numpy as np
import config


def load_rgb(path):
    bgr = cv2.imread(str(path))
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


_mask_cache = {}


def _fixed_mask(rgb):
    """Step 1: pixels that can be sky (inside the lens, not buildings / sun-blocking arm)."""
    h, w = rgb.shape[:2]
    if config.MASK_PATH is not None:
        key = (h, w)
        if key not in _mask_cache:
            m = cv2.imread(str(config.MASK_PATH), cv2.IMREAD_GRAYSCALE)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            _mask_cache[key] = m > 0
        valid = _mask_cache[key].copy()
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        valid = gray > config.DARK_THRESHOLD
        valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((5, 5), np.uint8)).astype(bool)

    if config.CIRCLE is not None:
        cx, cy, r = config.CIRCLE
    else:
        ys, xs = np.where(valid)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        r = min(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    yy, xx = np.ogrid[:h, :w]
    valid &= ((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.98 * r) ** 2
    return valid, (cx, cy, r)


def _find_sun(gray, valid, r):
    """Step 2: brightest window (box filter = integral-image search)."""
    win = max(3, int(config.SUN_WIN_FRAC * r))
    g = np.where(valid, gray, 0).astype(np.float32)
    s = cv2.boxFilter(g, -1, (2 * win + 1, 2 * win + 1), normalize=False,
                      borderType=cv2.BORDER_CONSTANT)
    s[~valid] = -1
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return x, y


def ridler_calvard(values, init=None, tol=None):
    """Step 3: iterative threshold (Ridler & Calvard, 1978)."""
    init = config.CLOUD_INIT_THRESHOLD if init is None else init
    tol = config.CLOUD_TOL if tol is None else tol
    v = values[values > 0]
    t = float(init)
    if v.size == 0:
        return t
    for _ in range(100):
        lo, hi = v[v <= t], v[v > t]
        if lo.size == 0 or hi.size == 0:
            break
        t_new = (lo.mean() + hi.mean()) / 2.0
        done = abs(t_new - t) <= tol
        t = t_new
        if done:
            break
    return t


def preprocess(rgb):
    """Return (sky_mask, (cx, cy, r), info). sky_mask is True for clean, cloud-free sky pixels."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    valid, (cx, cy, r) = _fixed_mask(rgb)                       # step 1
    sx, sy = _find_sun(gray, valid, r)                          # step 2
    yy, xx = np.ogrid[:h, :w]
    sun = ((xx - sx) ** 2 + (yy - sy) ** 2) <= (config.SUN_MASK_FRAC * r) ** 2
    valid_ns = valid & ~sun
    thr = ridler_calvard(gray[valid_ns])                        # step 3
    if config.CLOUD_THRESHOLD_FLOOR is not None:
        thr = max(thr, config.CLOUD_THRESHOLD_FLOOR)
    sky = valid_ns & (gray <= thr)

    info = dict(sun_xy=(sx, sy), threshold=thr,
                valid_frac=float(sky.sum() / (np.pi * r * r)))
    return sky, (cx, cy, r), info
