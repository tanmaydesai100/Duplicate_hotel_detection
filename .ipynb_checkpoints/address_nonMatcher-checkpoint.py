#!/usr/bin/env python3
import os
import re
import time
import requests
import pandas as pd
import shelve
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import radians, cos, sin, asin, sqrt
from postal.parser import parse_address
from rapidfuzz import fuzz

# ─────────── CONFIG ───────────

RAW_A      = "kaggle/hotel_with_id.csv"
RAW_B      = "ukDataset/dataset2_final.csv"
ENR_A      = "datasetA_enriched_top100.csv"
ENR_B      = "datasetB_enriched.csv"
CACHE_FILE = "postcode_cache.db"

address_col_A = "Final_address"
lat_col_A     = "Latitude"
lon_col_A     = "Longitude"

address_col_B = "modified_address"
lat_col_B     = "lat"
lon_col_B     = "lon"

# We’re looking for non‐matches in this score range:
LOWER_THRESH = 0.5
UPPER_THRESH = 0.6

# postcode extractor
postcode_regex = r"(GIR ?0AA|(?:[A-Z]{1,2}\d{1,2}[A-Z]?) ?\d[A-Z]{2})"

# ─────────── UTILS ───────────

def normalize_pc(pc: str) -> str:
    return pc.replace(" ", "").upper()

def extract_postcode(raw_address: str) -> str:
    m = re.search(postcode_regex, raw_address, re.IGNORECASE)
    return normalize_pc(m.group(0)) if m else None

def lookup_single(pc: str) -> tuple[str, dict]:
    url = f"https://api.postcodes.io/postcodes/{pc}"
    try:
        resp = requests.get(url, timeout=5)
        time.sleep(0.2)
        if resp.status_code == 200:
            return pc, resp.json().get("result")
    except Exception:
        pass
    return pc, None

def safe_load_cache():
    try:
        with shelve.open(CACHE_FILE):
            pass
        return True
    except Exception:
        # corrupted
        try: os.remove(CACHE_FILE)
        except OSError: pass
        return False

def enrich_dataframe(df: pd.DataFrame, address_col: str) -> pd.DataFrame:
    df["raw_pc"] = df[address_col].astype(str).apply(extract_postcode)
    pcs = df["raw_pc"].dropna().unique().tolist()
    safe_load_cache()
    with shelve.open(CACHE_FILE) as cache:
        to_lookup = [pc for pc in pcs if pc not in cache]
    if to_lookup:
        with ThreadPoolExecutor(max_workers=8) as exe:
            for pc, data in exe.map(lookup_single, to_lookup):
                with shelve.open(CACHE_FILE) as cache:
                    cache[pc] = data
    mod_list, pc_list, city_list, region_list = [], [], [], []
    with shelve.open(CACHE_FILE) as cache:
        for raw_addr, raw_pc in zip(df[address_col], df["raw_pc"]):
            data = cache.get(raw_pc) if raw_pc else {}
            base = re.sub(postcode_regex, "", raw_addr, flags=re.IGNORECASE).strip(", ") if raw_pc else raw_addr
            parts = [base]
            city   = data.get("admin_district")
            region = data.get("region")
            if city   and city.lower()   not in base.lower(): parts.append(city)
            if region and region.lower() not in base.lower(): parts.append(region)
            if raw_pc: parts.append(raw_pc)
            clean = [p.strip() for p in parts if p]
            mod_list.append(", ".join(clean))
            pc_list.append(raw_pc)
            city_list.append(city)
            region_list.append(region)
    df["modified_address"] = mod_list
    df["pc_postcode"]    = pc_list
    df["pc_city"]        = city_list
    df["pc_region"]      = region_list
    return df

def normalize_address(raw: str) -> str:
    s = raw.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_uk_postcode(pc: str) -> str:
    txt = pc.strip().upper().replace(" ", "")
    m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})$", txt)
    return f"{m.group(1)} {m.group(2)}" if m else txt

def parse_addr(addr: str) -> dict:
    return {fld: val for val, fld in parse_address(addr)}

def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return float("inf")
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2*asin(sqrt(a))
    return R*c

def latlon_score(lat1, lon1, lat2, lon2):
    d = haversine(lat1, lon1, lat2, lon2)
    if d <= 50:    return 1.0
    if d <= 200:   return max(0.5, 1.0 - (d-50)/150)
    return 0.0

def match_score(addrA, addrB, latA, lonA, latB, lonB):
    # normalize & parse
    nA, nB = normalize_address(addrA), normalize_address(addrB)
    pA, pB = parse_addr(nA), parse_addr(nB)
    pcA = normalize_uk_postcode(pA.get("postcode",""))
    pcB = normalize_uk_postcode(pB.get("postcode",""))
    pc_match = 1 if pcA and pcA == pcB else 0
    road_score = fuzz.token_sort_ratio(pA.get("road",""), pB.get("road",""))/100
    hnA = re.sub(r"[^\d]","", pA.get("house_number",""))
    hnB = re.sub(r"[^\d]","", pB.get("house_number",""))
    hn_match = 1 if hnA and hnA == hnB else 0
    city_match = 1 if pA.get("city","") == pB.get("city","") else 0
    geo = latlon_score(latA, lonA, latB, lonB)
    score = (0.3*pc_match + 0.3*road_score +
             0.1*hn_match + 0.1*city_match + 0.2*geo)
    return score

# ─────────── MAIN ───────────

def main(start: int, end: int, output_csv: str):
    # ensure enrichment done
    if not os.path.exists(ENR_A) or not os.path.exists(ENR_B):
        print("Enriching raw data…")
        dfB = pd.read_csv(RAW_B, encoding="ISO-8859-1", dtype=str)
        dfB[[lat_col_B, lon_col_B]] = dfB[[lat_col_B, lon_col_B]].astype(float)
        enrich_dataframe(dfB, address_col_B).to_csv(ENR_B, index=False)

        dfA = pd.read_csv(RAW_A, encoding="ISO-8859-1", dtype=str)
        dfA[[lat_col_A, lon_col_A]] = dfA[[lat_col_A, lon_col_A]].astype(float)
        enrich_dataframe(dfA, address_col_A).to_csv(ENR_A, index=False)

    # load enriched
    dfA = pd.read_csv(ENR_A, dtype=str)
    dfB = pd.read_csv(ENR_B, dtype=str)
    dfA[[lat_col_A, lon_col_A]] = dfA[[lat_col_A, lon_col_A]].astype(float)
    dfB[[lat_col_B, lon_col_B]] = dfB[[lat_col_B, lon_col_B]].astype(float)

    # slice A
    dfA_slice = dfA.iloc[start:end]

    results = []
    for _, a in dfA_slice.iterrows():
        for _, b in dfB.iterrows():
            sc = match_score(
                a["modified_address"], b["modified_address"],
                a[lat_col_A], a[lon_col_A],
                b[lat_col_B], b[lon_col_B]
            )
            if LOWER_THRESH <= sc < UPPER_THRESH:
                results.append({
                    "indexA":    a.name,
                    "addressA":  a["modified_address"],
                    "indexB":    b.name,
                    "addressB":  b["modified_address"],
                    "score":     round(sc,4)
                })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"→ Saved {len(out_df)} non‑matches with scores in [{LOWER_THRESH},{UPPER_THRESH}) → {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start",  type=int, default=0,   help="start row of A")
    p.add_argument("--end",    type=int, default=100, help="end row of A")
    p.add_argument("--output", type=str, default="non_matches.csv",
                   help="CSV file for non‑matches")
    args = p.parse_args()
    main(args.start, args.end, args.output)
