# import pandas as pd
# import re
# from postal.parser import parse_address
# from rapidfuzz import fuzz
# from math import radians, cos, sin, asin, sqrt
# from multiprocessing import Pool, cpu_count
# from tqdm import tqdm   # ← New: import tqdm

# # -----------------------------
# # --- Normalization Helpers ---
# # -----------------------------
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

# # -----------------------------
# # --- Geo Distance Scoring ---
# # -----------------------------
# def haversine(lat1, lon1, lat2, lon2):
#     if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
#         return float("inf")
#     R = 6371000  # Earth radius in meters
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
#         return max(0.5, 1.0 - (dist - 50)/150)
#     else:
#         return 0.0

# # -----------------------------
# # --- Matching Logic (Single Pair) ---
# # -----------------------------
# def match_addresses(addrA, addrB, latA, lonA, latB, lonB, thresh=0.90):
#     normA = normalize_address(addrA)
#     parsedA = parse_addr(normA)
#     pcA = normalize_uk_postcode(parsedA.get("postcode", ""))

#     normB = normalize_address(addrB)
#     parsedB = parse_addr(normB)
#     pcB = normalize_uk_postcode(parsedB.get("postcode", ""))

#     # Field scores
#     pc_match = 1 if pcA and pcA == pcB else 0

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

# # -----------------------------
# # --- Worker Initialization & Globals ---
# # -----------------------------
# dfA = None
# dfB = None

# def init_worker(dataA, dataB):
#     """
#     Called once per worker process. Sets global variables that match_row will use.
#     dataA, dataB are tuples: (DataFrame, address_col, lat_col, lon_col)
#     """
#     global dfA, dfB
#     dfA = dataA[0]
#     dfB = dataB[0]

# # -----------------------------
# # --- Worker Function (Per Row of A) ---
# # -----------------------------
# def match_row(i):
#     """
#     Compares row i of dfA to every row of dfB in parallel workers.
#     Returns a list of match dicts for that row.
#     """
#     matches = []
#     rowA = dfA.iloc[i]
#     addrA = rowA[address_col_A_global]
#     latA = rowA[lat_col_A_global]
#     lonA = rowA[lon_col_A_global]

#     for j, rowB in dfB.iterrows():
#         addrB = rowB[address_col_B_global]
#         latB = rowB[lat_col_B_global]
#         lonB = rowB[lon_col_B_global]

#         is_match, score = match_addresses(addrA, addrB, latA, lonA, latB, lonB)
#         if is_match:
#             matches.append({
#                 "datasetA_row": i,
#                 "datasetB_row": j,
#                 "addressA": addrA,
#                 "addressB": addrB,
#                 "latA": latA,
#                 "lonA": lonA,
#                 "latB": latB,
#                 "lonB": lonB,
#                 "score": round(score, 4)
#             })
#     return matches

# # -----------------------------
# # --- Main Matching Routine (Parallel) ---
# # -----------------------------
# def run_parallel_only(
#     fileA, fileB,
#     address_col_A, lat_col_A, lon_col_A,
#     address_col_B, lat_col_B, lon_col_B,
#     output_file, threshold=0.90
# ):
#     global address_col_A_global, lat_col_A_global, lon_col_A_global
#     global address_col_B_global, lat_col_B_global, lon_col_B_global

#     # 1) Load datasets
#     dfA_local = pd.read_csv(fileA)
#     dfB_local = pd.read_csv(fileB)

#     # 2) Store column names in globals so worker can access
#     address_col_A_global = address_col_A
#     lat_col_A_global     = lat_col_A
#     lon_col_A_global     = lon_col_A
#     address_col_B_global = address_col_B
#     lat_col_B_global     = lat_col_B
#     lon_col_B_global     = lon_col_B

#     # 3) Assign globals via initializer arguments
#     dataA = (dfA_local, address_col_A, lat_col_A, lon_col_A)
#     dataB = (dfB_local, address_col_B, lat_col_B, lon_col_B)

#     # 4) Launch pool of workers with tqdm-wrapped imap
#     num_cpus = cpu_count()
#     n_workers = max(1, int(num_cpus * 0.4))
#     print(f"Launching {n_workers} worker processes (only parallelizing row loops)...")

#     all_results = []  # we will collect per-row match lists here
#     with Pool(processes=n_workers, initializer=init_worker, initargs=(dataA, dataB)) as pool:
#         # Replace pool.map(...) with pool.imap(...) + tqdm
#         for result in tqdm(
#             pool.imap(match_row, range(len(dfA_local))),
#             total=len(dfA_local),
#             desc="Matching rows",
#             ncols=80,
#         ):
#             all_results.append(result)

#     # 5) Flatten results and save
#     flat_matches = [match for sublist in all_results for match in sublist]
#     pd.DataFrame(flat_matches).to_csv(output_file, index=False)
#     print(f"✅ Matching complete: {len(flat_matches)} total matches saved to {output_file}")

# # -----------------------------
# # --- Run It ------------------
# # -----------------------------
# if __name__ == "__main__":
#     # Specify your file names and column names here:
#     fileA = "kaggle/hotel_with_id.csv"
#     fileB = "ukDataset/dataset2_final.csv"
#     address_col_A = "Final_address"
#     lat_col_A     = "Latitude"
#     lon_col_A     = "Longitude"
#     address_col_B = "modified_address"
#     lat_col_B     = "lat"
#     lon_col_B     = "lon"
#     output_file   = "matched_addresses_parallel.csv"

#     run_parallel_only(
#         fileA=fileA,
#         fileB=fileB,
#         address_col_A=address_col_A, lat_col_A=lat_col_A, lon_col_A=lon_col_A,
#         address_col_B=address_col_B, lat_col_B=lat_col_B, lon_col_B=lon_col_B,
#         output_file=output_file,
#         threshold=0.90
#     )

import os
import pandas as pd
import re
from postal.parser import parse_address
from rapidfuzz import fuzz
from math import radians, cos, sin, asin, sqrt
from multiprocessing import Pool
from tqdm import tqdm

# ─── Helper functions: must all be at top level ───

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
        return max(0.5, 1.0 - (dist - 50)/150)
    else:
        return 0.0

def match_addresses(addrA, addrB, latA, lonA, latB, lonB, thresh=0.90):
    normA = normalize_address(addrA)
    parsedA = parse_addr(normA)
    pcA = normalize_uk_postcode(parsedA.get("postcode", ""))

    normB = normalize_address(addrB)
    parsedB = parse_addr(normB)
    pcB = normalize_uk_postcode(parsedB.get("postcode", ""))

    pc_match = 1 if pcA and pcA == pcB else 0
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
    final_score = (
        0.3 * pc_match +
        0.3 * road_score +
        0.1 * hn_match +
        0.1 * city_match +
        0.2 * geo_score
    )
    return final_score >= thresh, final_score

# ─── End of helpers ───


dfA = None
dfB = None

def init_worker(dataA, dataB):
    global dfA, dfB
    dfA = dataA[0]
    dfB = dataB[0]

def match_row(i):
    matches = []
    rowA = dfA.iloc[i]
    addrA = rowA[address_col_A_global]
    latA = rowA[lat_col_A_global]
    lonA = rowA[lon_col_A_global]

    for j, rowB in dfB.iterrows():
        addrB = rowB[address_col_B_global]
        latB = rowB[lat_col_B_global]
        lonB = rowB[lon_col_B_global]
        # Now this call will succeed because match_addresses is already defined above
        is_match, score = match_addresses(addrA, addrB, latA, lonA, latB, lonB)
        if is_match:
            matches.append({
                "datasetA_row": i,
                "datasetB_row": j,
                "addressA": addrA,
                "addressB": addrB,
                "latA": latA,
                "lonA": lonA,
                "latB": latB,
                "lonB": lonB,
                "score": round(score, 4)
            })
    return matches

def run_parallel_only(
    fileA, fileB,
    address_col_A, lat_col_A, lon_col_A,
    address_col_B, lat_col_B, lon_col_B,
    output_file, threshold=0.90
):
    global address_col_A_global, lat_col_A_global, lon_col_A_global
    global address_col_B_global, lat_col_B_global, lon_col_B_global

    # 1) Load datasets
    dfA_local = pd.read_csv(fileA)
    dfB_local = pd.read_csv(fileB)

    # 2) Store column names in globals for worker access
    address_col_A_global = address_col_A
    lat_col_A_global     = lat_col_A
    lon_col_A_global     = lon_col_A
    address_col_B_global = address_col_B
    lat_col_B_global     = lat_col_B
    lon_col_B_global     = lon_col_B

    # 3) Prepare initializer data
    dataA = (dfA_local, address_col_A, lat_col_A, lon_col_A)
    dataB = (dfB_local, address_col_B, lat_col_B, lon_col_B)

    # 4) Pick a safe number of workers (80% of cores, but reserve at least 1)
    num_cpus = os.cpu_count() or 1
    n_workers = min(num_cpus - 1, int(num_cpus * 0.8))
    n_workers = max(1, n_workers)
    print(f"Detected {num_cpus} cores; launching {n_workers} worker processes.")

    # 5) Run the pool with tqdm‐wrapped imap
    all_results = []
    with Pool(processes=n_workers, initializer=init_worker, initargs=(dataA, dataB)) as pool:
        for result in tqdm(
            pool.imap(match_row, range(len(dfA_local))),
            total=len(dfA_local),
            desc="Matching rows",
            ncols=60
        ):
            all_results.append(result)

    # 6) Flatten and save
    flat_matches = [m for sublist in all_results for m in sublist]
    pd.DataFrame(flat_matches).to_csv(output_file, index=False)
    print(f"✅ Matching complete: {len(flat_matches)} matches → “{output_file}”")


if __name__ == "__main__":
    fileA = "kaggle/hotel_with_id.csv"
    fileB = "ukDataset/dataset2_final.csv"
    address_col_A = "Final_address"
    lat_col_A     = "Latitude"
    lon_col_A     = "Longitude"
    address_col_B = "modified_address"
    lat_col_B     = "lat"
    lon_col_B     = "lon"
    output_file   = "matched_addresses_parallel.csv"

    run_parallel_only(
        fileA=fileA,
        fileB=fileB,
        address_col_A=address_col_A, lat_col_A=lat_col_A, lon_col_A=lon_col_A,
        address_col_B=address_col_B, lat_col_B=lat_col_B, lon_col_B=lon_col_B,
        output_file=output_file,
        threshold=0.90
    )






