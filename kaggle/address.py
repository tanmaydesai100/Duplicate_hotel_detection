# import pandas as pd
# import re

# # Load full CSV
# df = pd.read_csv('uk_hotels_modified.csv')

# # --- Split 'Map' column into 'Latitude' and 'Longitude' ---
# df[['Latitude', 'Longitude']] = df['Map'].str.split('|', expand=True)
# df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
# df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

# # --- Format Final_address as per earlier requirement ---
# def format_address(row):
#     hotel_name = str(row['HotelName']).strip()
#     mod_addr = str(row['modified_address']).strip()
#     pin_code = str(row['PinCode']).strip()

#     # Prepend hotel name if not already at start
#     if not mod_addr.startswith(hotel_name):
#         mod_addr = f"{hotel_name}, {mod_addr}"

#     # Check for UK postcode and append PinCode if missing
#     postcode_pattern = r'[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}'
#     if not re.search(postcode_pattern, mod_addr):
#         mod_addr = f"{mod_addr}, {pin_code}"

#     return mod_addr

# df['Final_address'] = df.apply(format_address, axis=1)

# # Save result to 'test.csv'
# df.to_csv('test.csv', index=False)
import pandas as pd

# Load the CSV file (replace 'your_file.csv' with your actual file name)
df = pd.read_csv('test.csv')

# Add the index as a column named 'Id_datasetA' at the start
df.insert(0, 'Id_datasetA', df.index)

# Save the modified DataFrame back to a new CSV file (or overwrite the original)
df.to_csv('your_file_with_id.csv', index=False)
