# import os
# import time
# import logging

# import pandas as pd
# import numpy as np
# import requests
# from tqdm.auto import tqdm
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter

# # ——————————————————————————————————————————————————————————————
# # CONFIGURATION
# INPUT_CSV     = 'inference_input.csv'            # your original data
# OUTPUT_CSV    = 'addresses_corrected_free.csv'   # final output
# FAILED_CSV    = 'failed_geocodes.csv'            # any still‑missing rows
# LOG_FILE      = 'geocode_errors.log'
# PHOTON_URL    = 'https://photon.komoot.io/api/'

# # ——————————————————————————————————————————————————————————————
# # 1) Load your DataFrame
# df = pd.read_csv(INPUT_CSV)   # expects cols: ['orig_index','text','lat','long']

# # ——————————————————————————————————————————————————————————————
# # 2) Set up logging
# logging.basicConfig(
#     filename=LOG_FILE,
#     level=logging.WARNING,
#     format='%(asctime)s %(levelname)s: %(message)s'
# )

# # ——————————————————————————————————————————————————————————————
# # 3) First pass: Nominatim (free, throttled to 1 req/sec)
# nom = Nominatim(user_agent="free_geocoder", timeout=10)
# nom_geocode = RateLimiter(
#     nom.geocode,
#     min_delay_seconds=1.0,
#     max_retries=3,
#     error_wait_seconds=5.0,
#     swallow_exceptions=False
# )

# def lookup_nominatim(address):
#     try:
#         loc = nom_geocode(address)
#         if loc:
#             return loc.latitude, loc.longitude
#         logging.warning(f"Nominatim: no result for {address!r}")
#     except Exception as e:
#         logging.warning(f"Nominatim error for {address!r}: {e}")
#     return np.nan, np.nan

# # apply Nominatim with progress bar
# tqdm.pandas(desc="1️⃣ Nominatim pass")
# df[['lat_new', 'long_new']] = df['text'] \
#     .progress_apply(lambda a: pd.Series(lookup_nominatim(a)))

# # overwrite where blank or differs by >0.0005°
# tol = 0.0005
# mask_nom = (
#     df['lat'].isna() |
#     df['long'].isna() |
#     (df['lat'].sub(df['lat_new']).abs() > tol) |
#     (df['long'].sub(df['long_new']).abs() > tol)
# )
# df.loc[mask_nom, ['lat','long']] = df.loc[mask_nom, ['lat_new','long_new']]

# # ——————————————————————————————————————————————————————————————
# # 4) Second pass: Photon fallback for any still‑missing
# still_missing = df['lat'].isna() | df['long'].isna()
# print(f"{still_missing.sum()} addresses still missing → trying Photon fallback…")

# def lookup_photon(address):
#     params = {'q': address, 'limit': 1}
#     try:
#         resp = requests.get(PHOTON_URL, params=params, timeout=5)
#         resp.raise_for_status()
#         data = resp.json()
#         feats = data.get('features')
#         if feats:
#             lon, lat = feats[0]['geometry']['coordinates']
#             return lat, lon
#     except Exception as e:
#         logging.warning(f"Photon error for {address!r}: {e}")
#     return np.nan, np.nan

# if still_missing.any():
#     tqdm.pandas(desc="2️⃣ Photon pass")
#     df.loc[still_missing, ['lat','long']] = (
#         df.loc[still_missing, 'text']
#           .progress_apply(lambda a: pd.Series(lookup_photon(a)))
#     )

# # ——————————————————————————————————————————————————————————————
# # 5) Final reporting & save
# final_missing = df['lat'].isna() | df['long'].isna()
# if final_missing.any():
#     n = final_missing.sum()
#     print(f"⚠️  Still {n} rows without coords. See {FAILED_CSV}")
#     df.loc[final_missing, ['orig_index','text']] \
#       .to_csv(FAILED_CSV, index=False)
# else:
#     print("✅ All addresses now have lat/long!")

# # drop helper columns and write out
# df.drop(columns=['lat_new','long_new'], inplace=True)
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"Done: wrote {OUTPUT_CSV}")

# import pandas as pd

# # 1. Load your two CSVs
# addr = pd.read_csv("addresses_corrected_free.csv")     # columns: orig_index, text, lat, long
# geo  = pd.read_csv("geocoder_address.csv")             # columns: original_address, lat, long

# # 2. Build lookup dicts from the geocoder file
# lat_lookup  = geo.set_index("original_address")["lat"].to_dict()
# lng_lookup  = geo.set_index("original_address")["long"].to_dict()

# # 3. Fill only the missing coords in-place, row by row
# addr["lat"] = addr.apply(
#     lambda row: lat_lookup.get(row["text"], row["lat"]) 
#                 if pd.isna(row["lat"]) else row["lat"],
#     axis=1
# )
# addr["long"] = addr.apply(
#     lambda row: lng_lookup.get(row["text"], row["long"]) 
#                 if pd.isna(row["long"]) else row["long"],
#     axis=1
# )

# # 4. Write out a new CSV preserving original order & all rows
# addr.to_csv("addresses_corrected_filled.csv", index=False)

# print("Done — all original rows kept, missing lat/long filled.")


import pandas as pd

df = pd.read_csv("../data/hotel_pairs.csv")
print(df['label'].value_counts(normalize=True))
