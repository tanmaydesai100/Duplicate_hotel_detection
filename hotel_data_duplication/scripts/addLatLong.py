
# import pandas as pd

# # Load both CSV files
# target = pd.read_csv('merged_matches.csv')
# source = pd.read_csv('../newDataset2Python/cleaned_output.csv')

# # If 'orig_indexB' in target is 1-based, uncomment and adjust:
# # target['orig_indexB'] = target['orig_indexB'] - 1

# # Ensure 'orig_indexB' is integer type
# target['orig_indexB'] = pd.to_numeric(target['orig_indexB'], errors='coerce').astype('Int64')

# # Adjust source index so that first row maps to orig_indexB = 2
# source_indexed = source[['lat', 'lon']].copy()
# source_indexed.index = source_indexed.index + 2
# source_indexed.index.name = 'orig_indexB'
# source_indexed = source_indexed.reset_index()

# # Merge all rows: left join ensures missing indices become NaN
# merged = target.merge(
#     source_indexed,
#     on='orig_indexB',
#     how='left',
#     validate='many_to_one'
# )

# # Rename merged latitude/longitude columns
# merged = merged.rename(columns={
#     'lat': 'lat_indexB',
#     'lon': 'lon_indexB'
# })

# # Save the augmented DataFrame to CSV file
# output_file = 'merged_matches_with_latlon.csv'
# merged.to_csv(output_file, index=False)
# print(f"Merged {len(merged)} rows. Output saved to '{output_file}'.")

import pandas as pd

# Load CSV files
merged = pd.read_csv('merged_matches.csv')
hotels = pd.read_csv('../kaggle/hotel_with_id.csv')

# Ensure 'orig_indexA' and 'Id_datasetA' are integer types
merged['orig_indexA'] = pd.to_numeric(merged['orig_indexA'], errors='coerce').astype('Int64')
hotels['Id_datasetA'] = pd.to_numeric(hotels['Id_datasetA'], errors='coerce').astype('Int64')

# Prepare hotel DataFrame for merging
hotel_coords = hotels[['Id_datasetA', 'Latitude', 'Longitude']].copy()

# Merge Latitude/Longitude into merged_matches based on orig_indexA
result = merged.merge(
    hotel_coords,
    left_on='orig_indexA',
    right_on='Id_datasetA',
    how='left',
    validate='many_to_one'
)

# Rename new columns
result = result.rename(columns={
    'Latitude': 'lat_indexA',
    'Longitude': 'lon_indexA'
})

# Drop auxiliary columns if any
if 'Id_datasetA' in result.columns:
    result = result.drop(columns=['Id_datasetA'])

# Save the result to a new CSV file
output_file = 'merged_matches_with_hotel_latlon.csv'
result.to_csv(output_file, index=False)
print(f"Merged {len(result)} rows. Output saved to '{output_file}'.")
