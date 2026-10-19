"""Step 1 - download AERONET AOD ('all points') for the site and period set in config.py."""
import requests
import config

(y1, m1, d1), (y2, m2, d2) = config.AERONET_START, config.AERONET_END
url = ("https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
       f"?site={config.AERONET_SITE}&year={y1}&month={m1}&day={d1}&hour=0"
       f"&year2={y2}&month2={m2}&day2={d2}&hour2=23&{config.AERONET_PRODUCT}=1&AVG=10&if_no_html=1")
print("Requesting:", url)
r = requests.get(url, timeout=120)
r.raise_for_status()
if "Date(dd:mm:yyyy)" not in r.text:
    raise SystemExit("AERONET returned no data table for this site / period / level.\n"
                     "Try another nearby site, other dates, or AOD15 instead of AOD20 in config.py.\n"
                     "Server said: " + r.text[:300])
config.DATA_DIR.mkdir(exist_ok=True)
config.AERONET_CSV.write_text(r.text, encoding="utf-8")
print(f"Saved {len(r.text.splitlines())} lines to {config.AERONET_CSV}")
print("Open the file once and check that it contains AOD_440nm / AOD_675nm columns "
      "and the Angstrom exponent column named in config.AE_COL.")
