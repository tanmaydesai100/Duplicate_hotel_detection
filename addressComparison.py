# import os
# import re
# import time
# import requests
# import pandas as pd
# from math import radians, cos, sin, asin, sqrt
# from postal.parser import parse_address
# from rapidfuzz import fuzz
# from multiprocessing import Pool
# from tqdm import tqdm

# # ─────────────── POSTCODE LOOKUP / ADDRESS FORMATTING ───────────────

# postcode_regex = r"(GIR ?0AA|(?:[A-Z]{1,2}\d{1,2}[A-Z]?) ?\d[A-Z]{2})"
# postcode_cache = {}

# def lookup_postcode(postcode: str):
#     """
#     Given a UK postcode (possibly with or without space), query postcodes.io,
#     return the JSON 'result' dict, or None if lookup fails.
#     """
#     pc = postcode.replace(" ", "").upper()
#     if pc in postcode_cache:
#         return postcode_cache[pc]

#     url = f"https://api.postcodes.io/postcodes/{pc}"
#     try:
#         resp = requests.get(url, timeout=5)
#         time.sleep(0.6)  # rate limit courtesy
#         if resp.status_code == 200:
#             data = resp.json().get("result")
#             postcode_cache[pc] = data
#             return data
#     except Exception:
#         pass

#     postcode_cache[pc] = None
#     return None

# def format_address(raw_address: str):
#     """
#     1) Try to extract a postcode via regex from raw_address.
#     2) If found, look it up to get 'admin_district' (city) and 'region'.
#     3) Build a 'modified_address' = [ address_without_postcode ] + [city if missing] + [region if missing] + [postcode].
#     Returns: (modified_address, postcode, admin_district, region)
#     If no postcode found, returns (original raw_address, None, None, None).
#     """
#     if not isinstance(raw_address, str):
#         return raw_address, None, None, None

#     m = re.search(postcode_regex, raw_address, re.IGNORECASE)
#     if not m:
#         # No postcode found → return original string, plus Nones
#         return raw_address, None, None, None

#     found_pc = m.group(0).upper().replace(" ", "")
#     data = lookup_postcode(found_pc)

#     # Strip the matched postcode out of the original text
#     address_wo_pc = re.sub(postcode_regex, "", raw_address, flags=re.IGNORECASE).strip(", ")

#     parts = [address_wo_pc]
#     admin_district = None
#     region = None

#     if data:
#         admin_district = data.get("admin_district", None)
#         region = data.get("region", None)
#         if admin_district and admin_district.lower() not in address_wo_pc.lower():
#             parts.append(admin_district)
#         if region and region.lower() not in address_wo_pc.lower():
#             parts.append(region)

#     parts.append(found_pc)
#     modified_addr = ", ".join(p.strip() for p in parts if p)

#     return modified_addr, found_pc, admin_district, region

# # ─────────────── ADDRESS‐MATCHING HELPERS ───────────────

# def normalize_uk_postcode(pc_raw):
#     txt = str(pc_raw).strip().upper().replace(" ", "")
#     m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})$", txt)
#     if m:
#         return f"{m.group(1)} {m.group(2)}"
#     return txt

# def normalize_address(raw):
#     s = str(raw).lower()
#     s = re.sub(r"[^\w\s]", " ", s)
#     s = re.sub(r"\s+", " ", s.strip())
#     return s

# def parse_addr(addr_str):
#     tokens = parse_address(addr_str)
#     return {field: value for value, field in tokens}

# def haversine(lat1, lon1, lat2, lon2):
#     if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
#         return float("inf")
#     R = 6371000  # meters
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
#     c = 2 * asin(sqrt(a))
#     return R * c

# def latlon_score(lat1, lon1, lat2, lon2):
#     dist = haversine(lat1, lon1, lat2, lon2)
#     if dist <= 50:
#         return 1.0
#     elif dist <= 200:
#         return max(0.5, 1.0 - ((dist - 50) / 150))
#     else:
#         return 0.0

# def match_addresses(addrA, addrB, latA, lonA, latB, lonB, thresh=0.90):
#     """
#     Fuzzy + postcode + geo match between two normalized address strings.
#     Returns (is_match: bool, final_score: float).
#     """
#     normA = normalize_address(addrA)
#     parsedA = parse_addr(normA)
#     pcA = normalize_uk_postcode(parsedA.get("postcode", ""))

#     normB = normalize_address(addrB)
#     parsedB = parse_addr(normB)
#     pcB = normalize_uk_postcode(parsedB.get("postcode", ""))

#     pc_match = 1 if (pcA and pcA == pcB) else 0

#     roadA = parsedA.get("road", "")
#     roadB = parsedB.get("road", "")
#     road_score = fuzz.token_sort_ratio(roadA, roadB) / 100

#     hnA = re.sub(r"[^\d]", "", parsedA.get("house_number", ""))
#     hnB = re.sub(r"[^\d]", "", parsedB.get("house_number", ""))
#     hn_match = 1 if (hnA and hnA == hnB) else 0

#     cityA = parsedA.get("city", "")
#     cityB = parsedB.get("city", "")
#     city_match = 1 if cityA == cityB else 0

#     geo_score = latlon_score(latA, lonA, latB, lonB)

#     final_score = (
#         0.3 * pc_match +
#         0.3 * road_score +
#         0.1 * hn_match +
#         0.1 * city_match +
#         0.2 * geo_score
#     )
#     return final_score >= thresh, final_score

# # ─────────────── GLOBALS & WORKER SETUP ───────────────

# dfA = None
# dfB = None

# def init_worker(a_frame, b_frame,
#                 addrA_col, latA_col, lonA_col,
#                 addrB_col, latB_col, lonB_col):
#     """
#     In each worker, store DataFrame references and column‐names in globals.
#     """
#     global dfA, dfB
#     global address_col_A_global, lat_col_A_global, lon_col_A_global
#     global address_col_B_global, lat_col_B_global, lon_col_B_global

#     dfA = a_frame
#     dfB = b_frame
#     address_col_A_global = addrA_col
#     lat_col_A_global     = latA_col
#     lon_col_A_global     = lonA_col
#     address_col_B_global = addrB_col
#     lat_col_B_global     = latB_col
#     lon_col_B_global     = lonB_col

# def match_row(i):
#     """
#     Compare row i of dfA against every row of dfB. Return a list of match‐dicts.
#     Each dict includes:
#       - orig_indexA
#       - orig_indexB
#       - addressA (modified)
#       - addressB (modified)
#       - score
#     """
#     results = []
#     rowA = dfA.iloc[i]
#     addrA = rowA[address_col_A_global]
#     latA = rowA[lat_col_A_global]
#     lonA = rowA[lon_col_A_global]
#     origA = rowA["orig_indexA"]

#     for j, rowB in dfB.iterrows():
#         addrB = rowB[address_col_B_global]
#         latB = rowB[lat_col_B_global]
#         lonB = rowB[lon_col_B_global]
#         origB = rowB["orig_indexB"]

#         is_match, score = match_addresses(addrA, addrB, latA, lonA, latB, lonB)
#         if is_match:
#             results.append({
#                 "orig_indexA": origA,
#                 "orig_indexB": origB,
#                 "addressA": addrA,
#                 "addressB": addrB,
#                 "score": round(score, 4)
#             })
#     return results

# # ─────────────── MAIN SCRIPT ───────────────

# if __name__ == "__main__":
#     # ───────── USER CONFIG ─────────
#     fileA = "kaggle/hotel_with_id.csv"       # path to the 69k‐row CSV
#     fileB = "ukDataset/dataset2_final.csv"       # path to the 11k‐row CSV

#     address_col_A = "Final_address"    # column name in datasetA (raw address)
#     lat_col_A     = "Latitude"   # column name in datasetA
#     lon_col_A     = "Longitude"  # column name in datasetA

#     address_col_B = "modified_address"    # column name in datasetB (raw address)
#     lat_col_B     = "lat"   # column name in datasetB
#     lon_col_B     = "lon"  # column name in datasetB

#     output_file = "matches_first100.csv"
#     threshold = 0.90

#     # ─── 1) LOAD AND ENRICH DATAFRAME A ───
#     print("Loading datasetA and enriching with postcode data…")
#     dfA_raw = pd.read_csv(fileA, encoding="ISO-8859-1", dtype=str)
#     dfA_raw["orig_indexA"] = dfA_raw.index

#     # Cast lat/lon to float:
#     dfA_raw[lat_col_A] = dfA_raw[lat_col_A].astype(float)
#     dfA_raw[lon_col_A] = dfA_raw[lon_col_A].astype(float)

#     # Add enrichment columns
#     dfA_raw["modified_address"] = ""
#     dfA_raw["pc_postcode"]      = ""
#     dfA_raw["pc_city"]          = ""
#     dfA_raw["pc_region"]        = ""

#     for idx, row in tqdm(
#         dfA_raw.iterrows(),
#         total=len(dfA_raw),
#         desc="Enriching datasetA"
#     ):
#         raw_addr = row[address_col_A]
#         mod_addr, pc, city, region = format_address(raw_addr)
#         dfA_raw.at[idx, "modified_address"] = mod_addr
#         dfA_raw.at[idx, "pc_postcode"]      = pc
#         dfA_raw.at[idx, "pc_city"]          = city
#         dfA_raw.at[idx, "pc_region"]        = region

#     # Take only the first 100 rows (we’ll match these against all of B)
#     dfA = dfA_raw.iloc[:100].copy().reset_index(drop=True)

#     # ─── 2) LOAD AND ENRICH DATAFRAME B ───
#     print("Loading datasetB and enriching with postcode data…")
#     dfB = pd.read_csv(fileB, encoding="ISO-8859-1", dtype=str)
#     dfB["orig_indexB"] = dfB.index

#     # Cast lat/lon to float:
#     dfB[lat_col_B] = dfB[lat_col_B].astype(float)
#     dfB[lon_col_B] = dfB[lon_col_B].astype(float)

#     dfB["modified_address"] = ""
#     dfB["pc_postcode"]      = ""
#     dfB["pc_city"]          = ""
#     dfB["pc_region"]        = ""

#     for idx, row in tqdm(
#         dfB.iterrows(),
#         total=len(dfB),
#         desc="Enriching datasetB"
#     ):
#         raw_addr = row[address_col_B]
#         mod_addr, pc, city, region = format_address(raw_addr)
#         dfB.at[idx, "modified_address"] = mod_addr
#         dfB.at[idx, "pc_postcode"]      = pc
#         dfB.at[idx, "pc_city"]          = city
#         dfB.at[idx, "pc_region"]        = region

#     # ─── 3) MULTIPROCESS‐ENABLED MATCHING ───
#     print(f"Matching first {len(dfA)} rows of A against all {len(dfB)} rows of B…")
#     num_cpus = os.cpu_count() or 1
#     n_workers = min(num_cpus - 1, int(num_cpus * 0.8))
#     n_workers = max(1, n_workers)
#     print(f"→ Detected {num_cpus} CPUs; launching {n_workers} worker processes.")

#     all_results = []
#     with Pool(
#         processes=n_workers,
#         initializer=init_worker,
#         initargs=(
#             dfA, dfB,
#             "modified_address", lat_col_A, lon_col_A,
#             "modified_address", lat_col_B, lon_col_B
#         )
#     ) as pool:
#         for sublist in tqdm(
#             pool.imap(match_row, range(len(dfA))),
#             total=len(dfA),
#             desc="Matching rows (0–99 of A)",
#             ncols=60
#         ):
#             all_results.append(sublist)

#     # ─── 4) FLATTEN AND SAVE ───
#     flat = [item for sub in all_results for item in sub]
#     if not flat:
#         print("⚠️  No matches found among the first 100 rows of A.")
#     else:
#         out_df = pd.DataFrame(flat, columns=[
#             "orig_indexA", "orig_indexB", "addressA", "addressB", "score"
#         ])
#         out_df.to_csv(output_file, index=False)
#         print(f"✅  Matching complete ({len(out_df)} matches) → “{output_file}”")
# working code
# import os
# import re
# import time
# import requests
# import pandas as pd
# import shelve
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from math import radians, cos, sin, asin, sqrt
# from postal.parser import parse_address
# from rapidfuzz import fuzz
# from multiprocessing import Pool
# from tqdm import tqdm
# import pickle  # add this because used in safe_load_cache

# # ─────────── CONFIG ───────────

# RAW_A = "kaggle/hotel_with_id.csv"
# RAW_B = "ukDataset/dataset2_final.csv"

# ENR_A = "datasetA_enriched_top100.csv"
# ENR_B = "datasetB_enriched.csv"
# MATCH_OUTPUT = "matches_first100.csv"
# threshold = 0.8

# address_col_A = "Final_address"
# lat_col_A     = "Latitude"
# lon_col_A     = "Longitude"

# address_col_B = "modified_address"
# lat_col_B     = "lat"
# lon_col_B     = "lon"

# postcode_regex = r"(GIR ?0AA|(?:[A-Z]{1,2}\d{1,2}[A-Z]?) ?\d[A-Z]{2})"
# CACHE_FILE    = "postcode_cache.db"
# MAX_THREADS   = 8
# SLEEP_BETWEEN = 0.2

# # ─────────── STEP A: ENRICH ───────────

# def normalize_pc(pc: str) -> str:
#     return pc.replace(" ", "").upper()

# def extract_postcode(raw_address: str) -> str:
#     m = re.search(postcode_regex, raw_address, re.IGNORECASE)
#     return normalize_pc(m.group(0)) if m else None

# def lookup_single(pc: str) -> tuple[str, dict]:
#     url = f"https://api.postcodes.io/postcodes/{pc}"
#     try:
#         resp = requests.get(url, timeout=5)
#         time.sleep(SLEEP_BETWEEN)
#         if resp.status_code == 200:
#             data = resp.json().get("result")
#             return pc, data
#     except Exception:
#         pass
#     return pc, None

# def safe_load_cache():
#     try:
#         with shelve.open(CACHE_FILE) as cache:
#             _ = list(cache.keys())[:1]
#         return True
#     except (pickle.UnpicklingError, EOFError, KeyError):
#         print("⚠️  Corrupted postcode cache detected. Deleting and rebuilding it.")
#         try:
#             os.remove(CACHE_FILE)
#         except OSError:
#             pass
#         return False

# def enrich_dataframe(df: pd.DataFrame, address_col: str) -> pd.DataFrame:
#     df["raw_pc"] = df[address_col].astype(str).apply(extract_postcode)
#     unique_pcs = df["raw_pc"].dropna().unique().tolist()

#     safe_load_cache()
#     with shelve.open(CACHE_FILE) as cache:
#         to_lookup = [pc for pc in unique_pcs if pc not in cache]

#     if to_lookup:
#         print(f"Fetching {len(to_lookup):,} unique postcodes…")
#         results = {}
#         with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
#             futures = {executor.submit(lookup_single, pc): pc for pc in to_lookup}
#             for future in tqdm(as_completed(futures), total=len(futures), desc="Populating postcode cache"):
#                 pc, data = future.result()
#                 results[pc] = data

#         with shelve.open(CACHE_FILE) as cache:
#             for pc, data in results.items():
#                 cache[pc] = data

#     mod_list, pc_list, city_list, region_list = [], [], [], []
#     with shelve.open(CACHE_FILE) as cache:
#         for raw_addr, raw_pc in tqdm(zip(df[address_col], df["raw_pc"]),
#                                      total=len(df),
#                                      desc="Building modified_address"):
#             try:
#                 data = cache.get(raw_pc) if raw_pc else None
#             except (pickle.UnpicklingError, EOFError):
#                 data = None

#             base = re.sub(postcode_regex, "", raw_addr, flags=re.IGNORECASE).strip(", ") if raw_pc else raw_addr
#             parts = [base]
#             city = data.get("admin_district") if data else None
#             region = data.get("region") if data else None

#             if city and city.lower() not in base.lower():
#                 parts.append(city)
#             if region and region.lower() not in base.lower():
#                 parts.append(region)
#             if raw_pc:
#                 parts.append(raw_pc)

#             clean_parts = [str(p).strip() for p in parts if p and not pd.isna(p)]
#             mod_list.append(", ".join(clean_parts))
#             pc_list.append(raw_pc)
#             city_list.append(city)
#             region_list.append(region)

#     df["modified_address"] = mod_list
#     df["pc_postcode"] = pc_list
#     df["pc_city"] = city_list
#     df["pc_region"] = region_list
#     return df

# def stepA_enrich():
#     print("Loading datasetB…")
#     dfB = pd.read_csv(RAW_B, encoding="ISO-8859-1", dtype=str)
#     dfB[lat_col_B] = dfB[lat_col_B].astype(float)
#     dfB[lon_col_B] = dfB[lon_col_B].astype(float)
#     dfB = enrich_dataframe(dfB, address_col_B)
#     dfB.to_csv(ENR_B, index=False)
#     print(f"→ Saved enriched B → {ENR_B}\n")

#     print("Loading datasetA (top 100)…")
#     dfA = pd.read_csv(RAW_A, encoding="ISO-8859-1", dtype=str, nrows=100)
#     dfA[lat_col_A] = dfA[lat_col_A].astype(float)
#     dfA[lon_col_A] = dfA[lon_col_A].astype(float)
#     dfA = enrich_dataframe(dfA, address_col_A)
#     dfA.to_csv(ENR_A, index=False)
#     print(f"→ Saved enriched A (top 100) → {ENR_A}\n")

# # ─────────── STEP B: FUZZY + GEO MATCH ───────────

# def normalize_uk_postcode(pc_raw):
#     txt = str(pc_raw).strip().upper().replace(" ", "")
#     m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})$", txt)
#     if m:
#         return f"{m.group(1)} {m.group(2)}"
#     return txt

# def normalize_address(raw):
#     s = str(raw).lower()
#     s = re.sub(r"[^\w\s]", " ", s)
#     s = re.sub(r"\s+", " ", s.strip())
#     return s

# def parse_addr(addr_str):
#     tokens = parse_address(addr_str)
#     return {field: value for value, field in tokens}

# def haversine(lat1, lon1, lat2, lon2):
#     if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
#         return float("inf")
#     R = 6371000  # meters
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
#     c = 2 * asin(sqrt(a))
#     return R * c

# def latlon_score(lat1, lon1, lat2, lon2):
#     dist = haversine(lat1, lon1, lat2, lon2)
#     if dist <= 50:
#         return 1.0
#     elif dist <= 200:
#         return max(0.5, 1.0 - ((dist - 50)/150))
#     else:
#         return 0.0

# def match_addresses(addrA, addrB, latA, lonA, latB, lonB, thresh=threshold):
#     normA = normalize_address(addrA)
#     parsedA = parse_addr(normA)
#     pcA = normalize_uk_postcode(parsedA.get("postcode", ""))

#     normB = normalize_address(addrB)
#     parsedB = parse_addr(normB)
#     pcB = normalize_uk_postcode(parsedB.get("postcode", ""))

#     pc_match = 1 if (pcA and pcA == pcB) else 0
#     roadA = parsedA.get("road", "")
#     roadB = parsedB.get("road", "")
#     road_score = fuzz.token_sort_ratio(roadA, roadB) / 100

#     hnA = re.sub(r"[^\d]", "", parsedA.get("house_number", ""))
#     hnB = re.sub(r"[^\d]", "", parsedB.get("house_number", ""))
#     hn_match = 1 if (hnA and hnA == hnB) else 0

#     cityA = parsedA.get("city", "")
#     cityB = parsedB.get("city", "")
#     city_match = 1 if cityA == cityB else 0

#     geo_score = latlon_score(latA, lonA, latB, lonB)

#     final = (
#         0.3 * pc_match +
#         0.3 * road_score +
#         0.1 * hn_match +
#         0.1 * city_match +
#         0.2 * geo_score
#     )
#     return (final >= thresh), final

# # ─────────── GLOBAL VARIABLES FOR MULTIPROCESS ───────────

# dfA = None
# dfB = None

# def init_worker(a_frame, b_frame,
#                 addrA_col, latA_col, lonA_col,
#                 addrB_col, latB_col, lonB_col):
#     global dfA, dfB
#     global address_col_A_global, lat_col_A_global, lon_col_A_global
#     global address_col_B_global, lat_col_B_global, lon_col_B_global

#     dfA = a_frame
#     dfB = b_frame
#     address_col_A_global = addrA_col
#     lat_col_A_global     = latA_col
#     lon_col_A_global     = lonA_col
#     address_col_B_global = addrB_col
#     lat_col_B_global     = latB_col
#     lon_col_B_global     = lonB_col  # <== FIXED HERE!

# def match_row(i):
#     results = []
#     rowA = dfA.iloc[i]
#     addrA = rowA["modified_address"]
#     latA  = rowA[lat_col_A]
#     lonA  = rowA[lon_col_A]
#     origA = rowA["orig_indexA"]

#     for j, rowB in dfB.iterrows():
#         addrB = rowB["modified_address"]
#         latB  = rowB[lat_col_B]
#         lonB  = rowB[lon_col_B]
#         origB = rowB["orig_indexB"]

#         is_match, score = match_addresses(addrA, addrB, latA, lonA, latB, lonB)
#         if is_match:
#             results.append({
#                 "orig_indexA": origA,
#                 "orig_indexB": origB,
#                 "addressA": addrA,
#                 "addressB": addrB,
#                 "score": round(score, 4)
#             })
#     return results

# def stepB_match():
#     print("Loading enriched files…")
#     dfA_e = pd.read_csv(ENR_A, encoding="ISO-8859-1", dtype=str)
#     dfB_e = pd.read_csv(ENR_B, encoding="ISO-8859-1", dtype=str)

#     dfA_e[lat_col_A] = dfA_e[lat_col_A].astype(float)
#     dfA_e[lon_col_A] = dfA_e[lon_col_A].astype(float)
#     dfB_e[lat_col_B] = dfB_e[lat_col_B].astype(float)
#     dfB_e[lon_col_B] = dfB_e[lon_col_B].astype(float)

#     if "orig_indexA" not in dfA_e.columns:
#         dfA_e["orig_indexA"] = dfA_e.index
#     if "orig_indexB" not in dfB_e.columns:
#         dfB_e["orig_indexB"] = dfB_e.index

#     print(f"Matching first {len(dfA_e)} rows of A vs all {len(dfB_e)} rows of B…")

#     import psutil
#     cores = psutil.cpu_count(logical=True)
#     phys_cores = psutil.cpu_count(logical=False)
#     total_ram = round(psutil.virtual_memory().total / (1024**3), 1)
#     avail_ram = round(psutil.virtual_memory().available / (1024**3), 1)
#     workers = int(min(cores - 1, cores * 0.8)) or 1
#     print(f"Detected {phys_cores} physical cores, {cores} logical CPUs.")
#     print(f"RAM: {total_ram} GB total, {avail_ram} GB available.")
#     print(f"→ Launching {workers} worker processes.\n")

#     all_results = []
#     with Pool(
#         processes=workers,
#         initializer=init_worker,
#         initargs=(
#             dfA_e, dfB_e,
#             "modified_address", lat_col_A, lon_col_A,
#             "modified_address", lat_col_B, lon_col_B
#         )
#     ) as pool:
#         for sublist in tqdm(
#             pool.imap(match_row, range(len(dfA_e))),
#             total=len(dfA_e),
#             desc="Matching rows (0–99 of A)",
#             ncols=60
#         ):
#             all_results.append(sublist)

#     flat = [item for sub in all_results for item in sub]
#     if not flat:
#         print("⚠️  No matches found among the first", len(dfA_e), "rows of A.")
#         return

#     out_df = pd.DataFrame(flat, columns=[
#         "orig_indexA", "orig_indexB", "addressA", "addressB", "score"
#     ])
#     out_df.to_csv(MATCH_OUTPUT, index=False)
#     print(f"✅  Matching complete ({len(out_df)} matches) → “{MATCH_OUTPUT}”")

# # ─────────── ENTRY POINT ───────────

# if __name__ == "__main__":
#     stepA_enrich()
#     stepB_match()

# import os
# import re
# import time
# import requests
# import pandas as pd
# import shelve
# import pickle
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from math import radians, cos, sin, asin, sqrt
# from postal.parser import parse_address
# from rapidfuzz import fuzz
# from multiprocessing import Pool
# from tqdm import tqdm
# import argparse

# # ─────────── CONFIG ───────────

# # File paths
# RAW_A = "kaggle/hotel_with_id.csv"
# RAW_B = "ukDataset/dataset2_final.csv"

# ENR_A = "datasetA_enriched_top100.csv"
# ENR_B = "datasetB_enriched.csv"

# threshold = 0.65

# # Columns in raw CSVs
# address_col_A = "Final_address"
# lat_col_A = "Latitude"
# lon_col_A = "Longitude"

# address_col_B = "modified_address"
# lat_col_B = "lat"
# lon_col_B = "lon"

# # Postcode caching
# postcode_regex = r"(GIR ?0AA|(?:[A-Z]{1,2}\d{1,2}[A-Z]?) ?\d[A-Z]{2})"
# CACHE_FILE = "postcode_cache.db"
# MAX_THREADS = 8
# SLEEP_BETWEEN = 0.2

# # ─────────── STEP A: ENRICH ───────────

# def normalize_pc(pc: str) -> str:
#     return pc.replace(" ", "").upper()

# def extract_postcode(raw_address: str) -> str:
#     m = re.search(postcode_regex, raw_address, re.IGNORECASE)
#     return normalize_pc(m.group(0)) if m else None

# def lookup_single(pc: str) -> tuple[str, dict]:
#     url = f"https://api.postcodes.io/postcodes/{pc}"
#     try:
#         resp = requests.get(url, timeout=5)
#         time.sleep(SLEEP_BETWEEN)
#         if resp.status_code == 200:
#             data = resp.json().get("result")
#             return pc, data
#     except Exception:
#         pass
#     return pc, None

# def safe_load_cache():
#     try:
#         with shelve.open(CACHE_FILE) as cache:
#             _ = list(cache.keys())[:1]
#         return True
#     except (pickle.UnpicklingError, EOFError, KeyError):
#         print("⚠️  Corrupted postcode cache detected. Deleting and rebuilding it.")
#         try:
#             os.remove(CACHE_FILE)
#         except OSError:
#             pass
#         return False

# def enrich_dataframe(df: pd.DataFrame, address_col: str) -> pd.DataFrame:
#     df["raw_pc"] = df[address_col].astype(str).apply(extract_postcode)
#     unique_pcs = df["raw_pc"].dropna().unique().tolist()

#     safe_load_cache()
#     with shelve.open(CACHE_FILE) as cache:
#         to_lookup = [pc for pc in unique_pcs if pc not in cache]

#     if to_lookup:
#         print(f"Fetching {len(to_lookup):,} unique postcodes…")
#         results = {}
#         with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
#             futures = {executor.submit(lookup_single, pc): pc for pc in to_lookup}
#             for future in tqdm(as_completed(futures), total=len(futures), desc="Populating postcode cache"):
#                 pc, data = future.result()
#                 results[pc] = data

#         with shelve.open(CACHE_FILE) as cache:
#             for pc, data in results.items():
#                 cache[pc] = data

#     mod_list, pc_list, city_list, region_list = [], [], [], []
#     with shelve.open(CACHE_FILE) as cache:
#         for raw_addr, raw_pc in tqdm(zip(df[address_col], df["raw_pc"]),
#                                      total=len(df),
#                                      desc="Building modified_address"):
#             try:
#                 data = cache.get(raw_pc) if raw_pc else None
#             except (pickle.UnpicklingError, EOFError):
#                 data = None

#             base = re.sub(postcode_regex, "", raw_addr, flags=re.IGNORECASE).strip(", ") if raw_pc else raw_addr
#             parts = [base]
#             city = data.get("admin_district") if data else None
#             region = data.get("region") if data else None

#             if city and city.lower() not in base.lower():
#                 parts.append(city)
#             if region and region.lower() not in base.lower():
#                 parts.append(region)
#             if raw_pc:
#                 parts.append(raw_pc)

#             clean_parts = [str(p).strip() for p in parts if p and not pd.isna(p)]
#             mod_list.append(", ".join(clean_parts))
#             pc_list.append(raw_pc)
#             city_list.append(city)
#             region_list.append(region)

#     df["modified_address"] = mod_list
#     df["pc_postcode"] = pc_list
#     df["pc_city"] = city_list
#     df["pc_region"] = region_list
#     return df

# def stepA_enrich():
#     print("Loading datasetB…")
#     dfB = pd.read_csv(RAW_B, encoding="ISO-8859-1", dtype=str)
#     dfB[lat_col_B] = dfB[lat_col_B].astype(float)
#     dfB[lon_col_B] = dfB[lon_col_B].astype(float)
#     dfB = enrich_dataframe(dfB, address_col_B)
#     dfB.to_csv(ENR_B, index=False)
#     print(f"→ Saved enriched B → {ENR_B}\n")

#     print("Loading datasetA (top 100)…")
#     dfA = pd.read_csv(RAW_A, encoding="ISO-8859-1", dtype=str)
#     dfA[lat_col_A] = dfA[lat_col_A].astype(float)
#     dfA[lon_col_A] = dfA[lon_col_A].astype(float)
#     dfA = enrich_dataframe(dfA, address_col_A)
#     dfA.to_csv(ENR_A, index=False)
#     print(f"→ Saved enriched A → {ENR_A}\n")

# # ─────────── STEP B: FUZZY + GEO MATCH ───────────

# def normalize_uk_postcode(pc_raw):
#     txt = str(pc_raw).strip().upper().replace(" ", "")
#     m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})$", txt)
#     if m:
#         return f"{m.group(1)} {m.group(2)}"
#     return txt

# def normalize_address(raw):
#     s = str(raw).lower()
#     s = re.sub(r"[^\w\s]", " ", s)
#     s = re.sub(r"\s+", " ", s.strip())
#     return s

# def parse_addr(addr_str):
#     tokens = parse_address(addr_str)
#     return {field: value for value, field in tokens}

# def haversine(lat1, lon1, lat2, lon2):
#     if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
#         return float("inf")
#     R = 6371000
#     lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
#     c = 2 * asin(sqrt(a))
#     return R * c

# def latlon_score(lat1, lon1, lat2, lon2):
#     dist = haversine(lat1, lon1, lat2, lon2)
#     if dist <= 50:
#         return 1.0
#     elif dist <= 200:
#         return max(0.5, 1.0 - ((dist - 50)/150))
#     else:
#         return 0.0

# def match_addresses(addrA, addrB, latA, lonA, latB, lonB, thresh=threshold):
#     normA = normalize_address(addrA)
#     parsedA = parse_addr(normA)
#     pcA = normalize_uk_postcode(parsedA.get("postcode", ""))

#     normB = normalize_address(addrB)
#     parsedB = parse_addr(normB)
#     pcB = normalize_uk_postcode(parsedB.get("postcode", ""))

#     pc_match = 1 if (pcA and pcA == pcB) else 0
#     roadA = parsedA.get("road", "")
#     roadB = parsedB.get("road", "")
#     road_score = fuzz.token_sort_ratio(roadA, roadB) / 100

#     hnA = re.sub(r"[^\d]", "", parsedA.get("house_number", ""))
#     hnB = re.sub(r"[^\d]", "", parsedB.get("house_number", ""))
#     hn_match = 1 if (hnA and hnA == hnB) else 0

#     cityA = parsedA.get("city", "")
#     cityB = parsedB.get("city", "")
#     city_match = 1 if cityA == cityB else 0

#     geo_score = latlon_score(latA, lonA, latB, lonB)

#     final = (
#         0.3 * pc_match +
#         0.3 * road_score +
#         0.1 * hn_match +
#         0.1 * city_match +
#         0.2 * geo_score
#     )
#     return (final >= thresh), final

# dfA = None
# dfB = None

# def init_worker(a_frame, b_frame,
#                 addrA_col, latA_col, lonA_col,
#                 addrB_col, latB_col, lonB_col):
#     global dfA, dfB
#     global address_col_A_global, lat_col_A_global, lon_col_A_global
#     global address_col_B_global, lat_col_B_global, lon_col_B_global

#     dfA = a_frame
#     dfB = b_frame
#     address_col_A_global = addrA_col
#     lat_col_A_global = latA_col
#     lon_col_A_global = lonA_col 
#     address_col_B_global = addrB_col
#     lat_col_B_global = latB_col
#     lon_col_B_global = lonB_col

# def match_row(i):
#     results = []
#     rowA = dfA.iloc[i]
#     addrA = rowA["modified_address"]
#     latA = rowA[lat_col_A_global]
#     lonA = rowA[lon_col_A_global]
#     origA = rowA["orig_indexA"]

#     for j, rowB in dfB.iterrows():
#         addrB = rowB["modified_address"]
#         latB = rowB[lat_col_B_global]
#         lonB = rowB[lon_col_B_global]
#         origB = rowB["orig_indexB"]

#         is_match, score = match_addresses(addrA, addrB, latA, lonA, latB, lonB)
#         if is_match:
#             results.append({
#                 "orig_indexA": origA,
#                 "orig_indexB": origB,
#                 "addressA": addrA,
#                 "addressB": addrB,
#                 "score": round(score, 4)
#             })
#     return results

# def stepB_match(start_idx=0, end_idx=100, match_output="matches.csv"):
#     print("Loading enriched files…")
#     dfA_e = pd.read_csv(ENR_A, encoding="ISO-8859-1", dtype=str)
#     dfB_e = pd.read_csv(ENR_B, encoding="ISO-8859-1", dtype=str)

#     dfA_e[lat_col_A] = dfA_e[lat_col_A].astype(float)
#     dfA_e[lon_col_A] = dfA_e[lon_col_A].astype(float)
#     dfB_e[lat_col_B] = dfB_e[lat_col_B].astype(float)
#     dfB_e[lon_col_B] = dfB_e[lon_col_B].astype(float)

#     if "orig_indexA" not in dfA_e.columns:
#         dfA_e["orig_indexA"] = dfA_e.index
#     if "orig_indexB" not in dfB_e.columns:
#         dfB_e["orig_indexB"] = dfB_e.index

#     dfA_slice = dfA_e.iloc[start_idx:end_idx]

#     print(f"Matching rows {start_idx}–{end_idx} of A vs all {len(dfB_e)} rows of B…")

#     import psutil
#     cores = psutil.cpu_count(logical=True)
#     phys_cores = psutil.cpu_count(logical=False)
#     total_ram = round(psutil.virtual_memory().total / (1024**3), 1)
#     avail_ram = round(psutil.virtual_memory().available / (1024**3), 1)
#     workers = int(min(cores - 1, cores * 0.8)) or 1
#     print(f"Detected {phys_cores} physical cores, {cores} logical CPUs.")
#     print(f"RAM: {total_ram} GB total, {avail_ram} GB available.")
#     print(f"→ Launching {workers} worker processes.\n")

#     all_results = []
#     with Pool(
#         processes=workers,
#         initializer=init_worker,
#         initargs=(
#             dfA_slice, dfB_e,
#             "modified_address", lat_col_A, lon_col_A,
#             "modified_address", lat_col_B, lon_col_B
#         )
#     ) as pool:
#         for sublist in tqdm(
#             pool.imap(match_row, range(len(dfA_slice))),
#             total=len(dfA_slice),
#             desc=f"Matching rows ({start_idx}–{end_idx} of A)",
#             ncols=60
#         ):
#             all_results.append(sublist)

#     flat = [item for sub in all_results for item in sub]
#     if not flat:
#         print("⚠️  No matches found in rows", start_idx, "to", end_idx)
#         return

#     out_df = pd.DataFrame(flat, columns=[
#         "orig_indexA", "orig_indexB", "addressA", "addressB", "score"
#     ])

#     if os.path.exists(match_output):
#         out_df.to_csv(match_output, index=False, mode='a', header=False)
#         print(f"✅ Appended {len(out_df)} matches → “{match_output}”")
#     else:
#         out_df.to_csv(match_output, index=False)
#         print(f"✅ Saved {len(out_df)} matches → “{match_output}”")

# # ─────────── ENTRY POINT ───────────

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Address Matcher with Range Selection")
#     parser.add_argument("--start", type=int, default=0, help="Start index of dataset A")
#     parser.add_argument("--end", type=int, default=100, help="End index of dataset A")
#     parser.add_argument("--output", type=str, default="matches.csv", help="Output CSV filename")
#     parser.add_argument("--skip_enrich", action='store_true', help="Skip Step A (enrich) if already done")
#     args = parser.parse_args()

#     if not args.skip_enrich:
#         stepA_enrich()

#     stepB_match(start_idx=args.start, end_idx=args.end, match_output=args.output)

#batches code
import os
import re
import time
import requests
import pandas as pd
import shelve
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import radians, cos, sin, asin, sqrt
from postal.parser import parse_address
from rapidfuzz import fuzz
from multiprocessing import Pool
from tqdm import tqdm
import argparse

# ─────────── CONFIG ───────────

# File paths
RAW_A = "kaggle/hotel_with_id.csv"
RAW_B = "ukDataset/dataset2_final.csv"

ENR_A = "datasetA_enriched_top100.csv"
ENR_B = "datasetB_enriched.csv"

threshold = 0.65

# Columns in raw CSVs
address_col_A = "Final_address"
lat_col_A = "Latitude"
lon_col_A = "Longitude"

address_col_B = "modified_address"
lat_col_B = "lat"
lon_col_B = "lon"

# Postcode caching
postcode_regex = r"(GIR ?0AA|(?:[A-Z]{1,2}\d{1,2}[A-Z]?) ?\d[A-Z]{2})"
CACHE_FILE = "postcode_cache.db"
MAX_THREADS = 8                # cap for postcode‐lookup threads
SLEEP_BETWEEN = 0.2

# Step B matching
MAX_MATCH_WORKERS = 8          # cap for Pool worker processes
SAVE_BATCH_SIZE = 100          # save to CSV after every N rows of dfA_slice

# ─────────── STEP A: ENRICH ───────────

def normalize_pc(pc: str) -> str:
    return pc.replace(" ", "").upper()

def extract_postcode(raw_address: str) -> str:
    m = re.search(postcode_regex, raw_address, re.IGNORECASE)
    return normalize_pc(m.group(0)) if m else None

def lookup_single(pc: str) -> tuple[str, dict]:
    url = f"https://api.postcodes.io/postcodes/{pc}"
    try:
        resp = requests.get(url, timeout=5)
        time.sleep(SLEEP_BETWEEN)
        if resp.status_code == 200:
            data = resp.json().get("result")
            return pc, data
    except Exception:
        pass
    return pc, None

def safe_load_cache():
    try:
        with shelve.open(CACHE_FILE) as cache:
            _ = list(cache.keys())[:1]
        return True
    except (pickle.UnpicklingError, EOFError, KeyError):
        print("⚠️  Corrupted postcode cache detected. Deleting and rebuilding it.")
        try:
            os.remove(CACHE_FILE)
        except OSError:
            pass
        return False

def enrich_dataframe(df: pd.DataFrame, address_col: str) -> pd.DataFrame:
    df["raw_pc"] = df[address_col].astype(str).apply(extract_postcode)
    unique_pcs = df["raw_pc"].dropna().unique().tolist()

    safe_load_cache()
    with shelve.open(CACHE_FILE) as cache:
        to_lookup = [pc for pc in unique_pcs if pc not in cache]

    if to_lookup:
        print(f"Fetching {len(to_lookup):,} unique postcodes…")
        results = {}
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = {executor.submit(lookup_single, pc): pc for pc in to_lookup}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Populating postcode cache"
            ):
                pc, data = future.result()
                results[pc] = data

        with shelve.open(CACHE_FILE) as cache:
            for pc, data in results.items():
                cache[pc] = data

    mod_list, pc_list, city_list, region_list = [], [], [], []
    with shelve.open(CACHE_FILE) as cache:
        for raw_addr, raw_pc in tqdm(
            zip(df[address_col], df["raw_pc"]),
            total=len(df),
            desc="Building modified_address"
        ):
            try:
                data = cache.get(raw_pc) if raw_pc else None
            except (pickle.UnpicklingError, EOFError):
                data = None

            # strip existing postcode from raw_addr, then add back in normalized form
            base = (
                re.sub(postcode_regex, "", raw_addr, flags=re.IGNORECASE).strip(", ")
                if raw_pc
                else raw_addr
            )
            parts = [base]
            city = data.get("admin_district") if data else None
            region = data.get("region") if data else None

            if city and city.lower() not in base.lower():
                parts.append(city)
            if region and region.lower() not in base.lower():
                parts.append(region)
            if raw_pc:
                parts.append(raw_pc)

            clean_parts = [str(p).strip() for p in parts if p and not pd.isna(p)]
            mod_list.append(", ".join(clean_parts))
            pc_list.append(raw_pc)
            city_list.append(city)
            region_list.append(region)

    df["modified_address"] = mod_list
    df["pc_postcode"] = pc_list
    df["pc_city"] = city_list
    df["pc_region"] = region_list
    return df

def stepA_enrich():
    print("Loading datasetB…")
    dfB = pd.read_csv(RAW_B, encoding="ISO-8859-1", dtype=str)
    dfB[lat_col_B] = dfB[lat_col_B].astype(float)
    dfB[lon_col_B] = dfB[lon_col_B].astype(float)
    dfB = enrich_dataframe(dfB, address_col_B)
    dfB.to_csv(ENR_B, index=False)
    print(f"→ Saved enriched B → {ENR_B}\n")

    print("Loading datasetA (top 100)…")
    dfA = pd.read_csv(RAW_A, encoding="ISO-8859-1", dtype=str)
    dfA[lat_col_A] = dfA[lat_col_A].astype(float)
    dfA[lon_col_A] = dfA[lon_col_A].astype(float)
    dfA = enrich_dataframe(dfA, address_col_A)
    dfA.to_csv(ENR_A, index=False)
    print(f"→ Saved enriched A → {ENR_A}\n")

# ─────────── STEP B: FUZZY + GEO MATCH ───────────

def normalize_uk_postcode(pc_raw):
    txt = str(pc_raw).strip().upper().replace(" ", "")
    m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})$", txt)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return txt

def normalize_address(raw):
    s = str(raw).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s

def parse_addr(addr_str):
    tokens = parse_address(addr_str)
    return {field: value for value, field in tokens}

def haversine(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return float("inf")
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def latlon_score(lat1, lon1, lat2, lon2):
    dist = haversine(lat1, lon1, lat2, lon2)
    if dist <= 50:
        return 1.0
    elif dist <= 200:
        return max(0.5, 1.0 - ((dist - 50)/150))
    else:
        return 0.0

def match_addresses(addrA, addrB, latA, lonA, latB, lonB, thresh=threshold):
    normA = normalize_address(addrA)
    parsedA = parse_addr(normA)
    pcA = normalize_uk_postcode(parsedA.get("postcode", ""))

    normB = normalize_address(addrB)
    parsedB = parse_addr(normB)
    pcB = normalize_uk_postcode(parsedB.get("postcode", ""))

    pc_match = 1 if (pcA and pcA == pcB) else 0
    roadA = parsedA.get("road", "")
    roadB = parsedB.get("road", "")
    road_score = fuzz.token_sort_ratio(roadA, roadB) / 100

    hnA = re.sub(r"[^\d]", "", parsedA.get("house_number", ""))
    hnB = re.sub(r"[^\d]", "", parsedB.get("house_number", ""))
    hn_match = 1 if (hnA and hnA == hnB) else 0

    cityA = parsedA.get("city", "")
    cityB = parsedB.get("city", "")
    city_match = 1 if cityA == cityB else 0

    geo_score = latlon_score(latA, lonA, latB, lonB)

    final = (
        0.3 * pc_match +
        0.3 * road_score +
        0.1 * hn_match +
        0.1 * city_match +
        0.2 * geo_score
    )
    return (final >= thresh), final

# These globals will be initialized once per worker process
dfA = None
dfB = None

def init_worker(a_frame, b_frame,
                addrA_col, latA_col, lonA_col,
                addrB_col, latB_col, lonB_col):
    global dfA, dfB
    global address_col_A_global, lat_col_A_global, lon_col_A_global
    global address_col_B_global, lat_col_B_global, lon_col_B_global

    dfA = a_frame
    dfB = b_frame
    address_col_A_global = addrA_col
    lat_col_A_global = latA_col
    lon_col_A_global = lonA_col 
    address_col_B_global = addrB_col
    lat_col_B_global = latB_col
    lon_col_B_global = lonB_col

def match_row(i):
    """
    This will run inside each Pool worker. It compares one row of dfA (global)
    against all of dfB (global) and returns a list of match‐dicts.
    """
    results = []
    rowA = dfA.iloc[i]
    addrA = rowA["modified_address"]
    latA = rowA[lat_col_A_global]
    lonA = rowA[lon_col_A_global]
    origA = rowA["orig_indexA"]

    # Iterate over every row in dfB (global dfB_e)
    for j, rowB in dfB.iterrows():
        addrB = rowB["modified_address"]
        latB = rowB[lat_col_B_global]
        lonB = rowB[lon_col_B_global]
        origB = rowB["orig_indexB"]

        is_match, score = match_addresses(
            addrA, addrB,
            latA, lonA,
            latB, lonB
        )
        if is_match:
            results.append({
                "orig_indexA": origA,
                "orig_indexB": origB,
                "addressA": addrA,
                "addressB": addrB,
                "score": round(score, 4)
            })
    return results

def stepB_match(start_idx=0, end_idx=100, match_output="matches.csv"):
    print("Loading enriched files…")
    dfA_e = pd.read_csv(ENR_A, encoding="ISO-8859-1", dtype=str)
    dfB_e = pd.read_csv(ENR_B, encoding="ISO-8859-1", dtype=str)

    dfA_e[lat_col_A] = dfA_e[lat_col_A].astype(float)
    dfA_e[lon_col_A] = dfA_e[lon_col_A].astype(float)
    dfB_e[lat_col_B] = dfB_e[lat_col_B].astype(float)
    dfB_e[lon_col_B] = dfB_e[lon_col_B].astype(float)

    if "orig_indexA" not in dfA_e.columns:
        dfA_e["orig_indexA"] = dfA_e.index
    if "orig_indexB" not in dfB_e.columns:
        dfB_e["orig_indexB"] = dfB_e.index

    # Only slice the rows we want to match in this run
    dfA_slice = dfA_e.iloc[start_idx:end_idx]
    total_A_rows = len(dfA_slice)

    print(f"Matching rows {start_idx}–{end_idx} of A vs all {len(dfB_e)} rows of B…")

    # Determine CPU count, then cap it to MAX_MATCH_WORKERS
    import psutil
    cores = psutil.cpu_count(logical=True) or 1
    workers = min(cores, MAX_MATCH_WORKERS)
    print(f"Detected {psutil.cpu_count(logical=False)} physical cores, {cores} logical CPUs.")
    print(f"RAM: {round(psutil.virtual_memory().total/(1024**3),1)} GB total, "
          f"{round(psutil.virtual_memory().available/(1024**3),1)} GB available.")
    print(f"→ Launching {workers} worker processes.\n")

    # Prepare CSV header if needed
    if not os.path.exists(match_output):
        header_df = pd.DataFrame(
            columns=["orig_indexA", "orig_indexB", "addressA", "addressB", "score"]
        )
        header_df.to_csv(match_output, index=False)

    # We'll accumulate matches in small batches, write them out, then clear
    batch_counter = 0
    batch_list = []  # temporarily hold match‐dicts

    with Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(
            dfA_slice, dfB_e,
            "modified_address", lat_col_A, lon_col_A,
            "modified_address", lat_col_B, lon_col_B
        )
    ) as pool:
        for idx, sublist in enumerate(
            tqdm(
                pool.imap(match_row, range(total_A_rows)),
                total=total_A_rows,
                desc=f"Matching rows ({start_idx}–{end_idx} of A)",
                ncols=60
            )
        ):
            # sublist is a list of dicts of matched records for row i
            if sublist:
                batch_list.extend(sublist)

            # Every SAVE_BATCH_SIZE rows of dfA_slice, flush to disk
            if (idx + 1) % SAVE_BATCH_SIZE == 0 or (idx + 1) == total_A_rows:
                if batch_list:
                    out_df = pd.DataFrame(batch_list, columns=[
                        "orig_indexA", "orig_indexB", "addressA", "addressB", "score"
                    ])
                    out_df.to_csv(match_output, index=False, mode="a", header=False)
                    print(f"  • Flushed {len(out_df)} matches at row {start_idx + idx}")
                    batch_list.clear()

    print(f"\n✅ Finished matching rows {start_idx}–{end_idx}. Output → “{match_output}”")

# ─────────── ENTRY POINT ───────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Address Matcher with Range Selection")
    parser.add_argument("--start", type=int, default=0, help="Start index of dataset A")
    parser.add_argument("--end", type=int, default=100, help="End index of dataset A")
    parser.add_argument("--output", type=str, default="matches.csv", help="Output CSV filename")
    parser.add_argument("--skip_enrich", action='store_true', help="Skip Step A (enrich) if already done")
    args = parser.parse_args()

    if not args.skip_enrich:
        stepA_enrich()

    stepB_match(start_idx=args.start, end_idx=args.end, match_output=args.output)

    
