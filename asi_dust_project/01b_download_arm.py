"""OPTIONAL - download ARM SGP Total Sky Imager images with the ACT package (ARM's own Python tool).
    pip install act-atmos
Set ARM_USER / ARM_TOKEN / dates in config.py first. Files go to data/images/arm_raw/.
If you prefer, order the files by hand at https://adc.arm.gov/discovery and unzip them into data/images/.
Downloaded files may be .tar / .zip archives or .jpg - unpack archives into data/images/ before step 02."""
import act
import config

out = config.IMAGE_DIR / "arm_raw"
out.mkdir(parents=True, exist_ok=True)
files = act.discovery.download_arm_data(config.ARM_USER, config.ARM_TOKEN, config.ARM_DATASTREAM,
                                        config.ARM_START, config.ARM_END, output=str(out))
print(f"Downloaded {len(files)} files to {out}")
