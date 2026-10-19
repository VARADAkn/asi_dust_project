# Dust prediction from All-Sky Images + AERONET AOD  (NREL SRRL images)

Pipeline: sky image -> 3 preprocessing masks -> features -> match with AERONET AOD ->
Stage 1 XGBoost AOD regression (440/675/870/1020 nm) -> Stage 2 dust / non-dust classifier.

Images : NREL SRRL EKO ASI-16, Golden CO (Hugging Face: knl2366/NREL_Sky_Imagery, CC-BY 4.0)
         https://huggingface.co/datasets/knl2366/NREL_Sky_Imagery
AOD    : AERONET Version 3, direct sun, Level 1.5 (Level 2.0 is not released for recent dates)
         https://aeronet.gsfc.nasa.gov/cgi-bin/webtool_aod_v3

## 0. Setup (once)
Python 3.9+ . Use a SHORT folder path on Windows (e.g. C:\asi_project).

    python -m venv venv
    venv\Scripts\activate          (Windows)      |   source venv/bin/activate   (Linux/Mac)
    pip install -r requirements.txt

## 1. (Optional, 1 minute) test the code with FAKE data first
The images are synthetic, so the scores mean nothing - this only checks that your installation works.

    python 00_make_demo_data.py
    python show_preprocessing.py
    python 02_extract_features.py
    python 03_match_aeronet.py
    python 04_train_aod.py
    python 05_dust_classifier.py

Then DELETE the fake files before real work:
    delete everything inside data/ (images, csv files) and results/

## 2. Real run (NREL images + AERONET), about 1500 image/AOD pairs
config.py is already set to PRESET = "nrel". Check these values at the top of the "nrel" block:
AERONET_SITE, AERONET_START / AERONET_END, NREL_MAX_SAMPLES (1000-2000).

    python 01_get_aeronet.py
    python 01d_download_nrel.py
    python show_preprocessing.py
    python 02_extract_features.py
    python 03_match_aeronet.py
    python 04_train_aod.py
    python 05_dust_classifier.py

What each step does
- 01_get_aeronet.py     downloads AOD for the site/dates into data/aeronet_aod.csv.
                        If it says "no data": the site has nothing for those dates -> choose another nearby
                        AERONET site (AERONET_SITE) or other dates. The default BSRN_BAO_Boulder is ~35-40 km
                        from the camera and I could NOT verify it has data for Oct 2025-Apr 2026.
- 01d_download_nrel.py  downloads ONLY images within 5 min of an AERONET record (max NREL_MAX_SAMPLES) into
                        data/images/. List of chosen pairs: data/nrel_selected.csv.
- show_preprocessing.py saves results/preprocessing_check.png (original / sky mask / after masking). LOOK at it.
- 02_extract_features.py -> data/features.csv   (RGB means, RBR, brightness std, SZA, 12 ring points)
- 03_match_aeronet.py    -> data/dataset.csv    (image features + AOD at 4 wavelengths + Angstrom exponent)
- 04_train_aod.py        -> results/aod_metrics.csv, aod_scatter.png, feature_importance.png, saved models
- 05_dust_classifier.py  -> results/dust_metrics.csv, dust_confusion.png

## 3. Things to check / tune (in config.py)
1. Fisheye circle wrong or sun mask wrong in preprocessing_check.png:
   set CIRCLE = (cx, cy, radius) in pixels and/or SUN_MASK_FRAC. Re-run show_preprocessing.py.
2. Most ring features empty (NaN) / almost no sky left on clear days: set CLOUD_THRESHOLD_FLOOR = 200.
3. "Too few dust samples" in step 05: Colorado rarely has dust. The nrel preset already uses loose thresholds
   (AOD440 >= 0.10 and Angstrom exponent <= 1.0). This is a proxy for elevated coarse-dominated aerosol,
   NOT confirmed dust - say so in your report. You can loosen it further or report Stage 1 as the main result.
4. Fewer pairs than expected: widen AERONET_START / AERONET_END (AERONET only reports when the sun is clear).
5. Split is chronological (first 2/3 of each month = train, last 1/3 = test). Never shuffle.

## Limitations to state in the report
- Camera and AERONET photometer are different sites (distance), unlike the paper's ARM SGP set-up.
- Level 1.5 AOD (cloud-screened, not final quality-assured).
- ASI-16 is a different, uncalibrated-colour camera from the TSI-660; images cover only Oct 2025 onwards,
  so seasonal coverage is short.
- No sun-blocking arm on this camera, so no arm mask is needed.

## Other (older) options still included
01b_download_arm.py (ARM SGP via ACT) and 01c_download_eye2sky.py (Eye2Sky images). To use them, change PRESET
at the bottom of config.py to "arm" or "eye2sky".
