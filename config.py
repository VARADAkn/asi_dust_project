"""
Central settings for the ASI dust / AOD project.
Edit this file first, then run the numbered scripts in order.
"""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
IMAGE_DIR = DATA_DIR / "images"            # put all sky images (jpg/png) here, sub-folders are fine
AERONET_CSV = DATA_DIR / "aeronet_aod.csv" # raw AERONET AOD "all points" file
FEATURES_CSV = DATA_DIR / "features.csv"   # output of 02_extract_features.py
DATASET_CSV = DATA_DIR / "dataset.csv"     # output of 03_match_aeronet.py

# ---------------------------------------------------------------- site / camera
SITE_LAT = 53.15      # camera latitude  (SGP ARM: 36.605 ; Eye2Sky: see Eye2Sky_Station_List.xlsx)
SITE_LON = 8.17       # camera longitude (SGP ARM: -97.485)
SITE_ALT = 0.0        # metres

# Optional PNG mask (non-zero = usable sky). Use the Eye2Sky mask or your own fixed
# mask for buildings / the TSI sun-blocking arm. None = estimate the fisheye circle automatically.
MASK_PATH = None
# Optional manual fisheye circle (cx, cy, radius) in pixels. None = estimate from the mask / image.
CIRCLE = None

# Time stamp inside the image file name (all times are treated as UTC).
# CHECK a real file name and adjust! Default matches e.g. IMG_20220601_093000.jpg and ARM-style
# names such as sgptsiskyimageC1.a1.20200615.160000.jpg (date and time separated by '.', '_' or '-')
TIME_REGEX = r"(\d{8})[._-]?(\d{6})"
TIME_FORMAT = "%Y%m%d%H%M%S"
SUBSAMPLE_EVERY = 1   # use every n-th image (e.g. 10 if you have an image every 30 s)

# ---------------------------------------------------------------- preprocessing (fractions of sky radius)
DARK_THRESHOLD = 10        # pixels darker than this are "outside the lens"
SUN_WIN_FRAC = 0.05        # half-size of the window used to find the brightest region
SUN_MASK_FRAC = 0.25       # radius of the circle masked around the sun
CLOUD_INIT_THRESHOLD = 150 # Ridler-Calvard start threshold (as in the reference paper)
CLOUD_TOL = 1.0            # stop when threshold changes by <= this
CLOUD_THRESHOLD_FLOOR = None  # None = exactly as in the paper. A number (e.g. 200) stops the method from
                              # throwing away bright but cloud-free sky when the sky is clear.
MIN_VALID_FRAC = 0.05      # discard images with less than 5 % usable sky

# ---------------------------------------------------------------- features
N_ANNULUS = 12             # sample points on the ring (paper: at least 10)
ANNULUS_RADIUS_FRAC = 0.7  # ring radius (paper: 0.6-0.8 of sky radius)
PATCH = 5                  # each ring point = mean of a PATCH x PATCH window

BASE_FEATURES = ["R_mean", "G_mean", "B_mean", "RBR", "gray_std", "SZA"]
ANNULUS_FEATURES = [f"ann{i}_{c}" for i in range(N_ANNULUS) for c in "RGB"]
FEATURE_NAMES = BASE_FEATURES + ANNULUS_FEATURES

# ---------------------------------------------------------------- AERONET
AERONET_SITE = "ARM_SGP"   # exact AERONET site name
# ARM account for 01b_download_arm.py (get username + token at https://adc.arm.gov/armlive/register)
ARM_USER = "your_arm_username"
ARM_TOKEN = "your_arm_token"
ARM_DATASTREAM = "sgptsiskyimageC1.a1"
ARM_START = "2020-06-01"   # NOTE: ARM metadata lists this datastream as 2000-07-02 to 2020-11-30 - check the range!
ARM_END = "2020-06-07"
AERONET_START = (2024, 1, 1)
AERONET_END = (2024, 11, 30)
AERONET_PRODUCT = "AOD20"  # AOD20 = Level 2.0 (best), AOD15 = Level 1.5
WAVELENGTHS = [440, 675, 870, 1020]
AE_COL = "440-870_Angstrom_Exponent"
TIME_TOL_MIN = 5           # max image / AERONET time difference

# ---------------------------------------------------------------- model
SPLIT_TRAIN_FRACTION = 2 / 3     # first 2/3 of every month = train, last 1/3 = test
XGB_PARAMS = dict(max_depth=7, learning_rate=0.08, subsample=0.75,
                  colsample_bytree=1.0, gamma=0, seed=42)
CV_FOLDS = 10
EARLY_STOP = 30
MAX_ROUNDS = 1000

# ---------------------------------------------------------------- dust label (YOUR CHOICE - justify it!)
DUST_AOD_COL = "AOD_440nm"
DUST_AOD_MIN = 0.2
DUST_AE_MAX = 0.6
USE_PRED_AOD = True        # stage 2 also gets the predicted AOD (out-of-fold on the training set)
STAGE2_WL = 675


# ================================================================= EYE2SKY SETTINGS (01c_download_eye2sky.py)
E2S_URL = "https://eye2sky.de/data/asi/imgs"
E2S_SITES = ["OLUOL"]              # Eye2Sky station code(s); OLUOL = Oldenburg (covers Apr 2022 - Mar 2023)
E2S_START = "2022-06-01"           # first day, INCLUSIVE (the official script treats the end as exclusive; this one does not)
E2S_END = "2022-06-03"             # last day, inclusive. Start with 2-3 clear days!
E2S_EXTRACT_EVERY = 10             # keep every n-th image from each zip (images are 30 s apart -> 10 = every 5 min)
E2S_ZIP_DIR = DATA_DIR / "eye2sky_zips"
E2S_DELETE_ZIP = False             # True = delete each zip after unpacking (saves disk)

# Hours to SUBTRACT from the time in the image file name to get UTC (0 if the names are already UTC).
# Check this on a clear day - see README section "Time zone check".
TIME_OFFSET_HOURS = 0

# ================================================================= NREL SRRL SETTINGS (01d_download_nrel.py)
# Images: Hugging Face dataset knl2366/NREL_Sky_Imagery (EKO ASI-16, Golden CO, 1-min, from Oct 2025, CC-BY 4.0)
NREL_REPO = "knl2366/NREL_Sky_Imagery"
NREL_MAX_SAMPLES = 1500            # download at most this many image/AOD pairs (evenly spread over time)
NREL_MIN_CLEAR_SKY_INDEX = None    # e.g. 0.8 keeps only near-clear-sky images; None = no filter

# ================================================================= PRESET
# "nrel"    -> NREL SRRL (Golden CO) images from Hugging Face + nearest AERONET site (Level 1.5!)
# "arm"     -> ARM SGP images + AERONET ARM_SGP
# "eye2sky" -> Eye2Sky OLUOL images + nearest AERONET site (Helgoland, ~116 km away; Buesum ~119 km, Hamburg ~128 km)
PRESET = "nrel"
if PRESET == "nrel":
    SITE_LAT, SITE_LON, SITE_ALT = 39.742, -105.180, 1829.0
    # Nearest AERONET site I found is BSRN_BAO_Boulder (~35-40 km away). I could NOT verify it has data for
    # these dates: run `python 01_get_aeronet.py` and read the message. If empty, try another nearby site.
    AERONET_SITE = "BSRN_BAO_Boulder"
    AERONET_START, AERONET_END = (2025, 10, 7), (2026, 4, 30)
    AERONET_PRODUCT = "AOD15"      # Level 2.0 is not released for such recent dates -> use Level 1.5
    TIME_REGEX = r"(\d{8})[._-]?(\d{6})"   # NREL names: 20251007-163033.jpg (UTC)
    TIME_OFFSET_HOURS = 0
    SUBSAMPLE_EVERY = 1             # 01d already limits the number of images
    # Colorado rarely has dust, so the paper-style thresholds would give no positives. These looser values
    # define a "coarse-dominated, elevated aerosol" proxy - NOT confirmed dust. Say so in your report.
    DUST_AOD_MIN, DUST_AE_MAX = 0.10, 1.0
elif PRESET == "eye2sky":
    SITE_LAT, SITE_LON = 53.15, 8.17       # approx. Oldenburg; use the exact values from Eye2Sky_Station_List.xlsx
    AERONET_SITE = "Helgoland"             # check it has Level 2.0 data for your dates (AERONET download tool)
    AERONET_START, AERONET_END = (2022, 6, 1), (2022, 6, 30)
    TIME_REGEX = r"(\d{8})[._T-]?(\d{6})"   # adjust after you SEE a real file name
elif PRESET == "arm":
    SITE_LAT, SITE_LON = 36.605, -97.485
    AERONET_SITE = "ARM_SGP"
