import pandas as pd
import re
import requests
import time
import os
from tqdm import tqdm

# Config
input_file = "uk_hotels.csv"
output_file = "uk_hotels_modified.csv"
address_col = "Address"
postcode_regex = r"(GIR ?0AA|(?:[A-Z]{1,2}\d{1,2}[A-Z]?) ?\d[A-Z]{2})"
postcode_cache = {}

# 🔁 CHANGE THESE TWO for your chunk range
start_row = 60000      # ← Start index for this run (e.g., 0, 1001, ...)
end_row = 70000     # ← End index for this run (e.g., 1000, 2000, ...)

def lookup_postcode(postcode):
    postcode = postcode.replace(" ", "").upper()
    if postcode in postcode_cache:
        return postcode_cache[postcode]
    url = f"https://api.postcodes.io/postcodes/{postcode}"
    try:
        response = requests.get(url)
        time.sleep(0.6)
        if response.status_code == 200:
            data = response.json().get("result")
            postcode_cache[postcode] = data
            return data
    except Exception:
        pass
    postcode_cache[postcode] = None
    return None

def format_address(address):
    if not isinstance(address, str):
        return address
    match = re.search(postcode_regex, address, re.IGNORECASE)
    if not match:
        return address
    postcode = match.group(0).upper().strip()
    data = lookup_postcode(postcode)
    address_wo_postcode = re.sub(postcode_regex, '', address, flags=re.IGNORECASE).strip(", ")
    parts = [address_wo_postcode]
    if data:
        city = data.get("admin_district")
        region = data.get("region")
        if city and city.lower() not in address_wo_postcode.lower():
            parts.append(city)
        if region and region.lower() not in address_wo_postcode.lower():
            parts.append(region)
    parts.append(postcode)
    return ', '.join(part.strip() for part in parts if part)

# Load chunk from CSV
print(f"Processing rows {start_row} to {end_row}...")
df = pd.read_csv(input_file, encoding="ISO-8859-1", skiprows=range(1, start_row + 1), nrows=end_row - start_row)

# Ensure headers are correct
header_df = pd.read_csv(input_file, encoding="ISO-8859-1", nrows=0)
df.columns = header_df.columns

# Apply address formatting
df["modified_address"] = df[address_col].apply(format_address)

# Save to output file
if start_row == 0 and not os.path.exists(output_file):
    df.to_csv(output_file, index=False)
else:
    df.to_csv(output_file, mode='a', header=False, index=False)

print(f"✅ Finished rows {start_row} to {end_row}.")
