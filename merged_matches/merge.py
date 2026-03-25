# import pandas as pd

# # 1. Read the source CSV
# df = pd.read_csv("matches_065_079.csv")

# # 2. Filter to rows where Equal is Yes or yes
# df_yes = df[df['Equal'].str.lower() == 'yes']

# # 3. Save the filtered DataFrame to a new CSV
# df_yes.to_csv("matches_065_079_filtered.csv", index=False)

# print(f"Filtered down to {len(df_yes)} rows and saved to matches_065_079_filtered.csv")

import pandas as pd

# Load the CSV files
df1 = pd.read_csv('matches_065_079.csv')
df2 = pd.read_csv('matches_080_089.csv')
df3 = pd.read_csv('matches_090_100.csv')

# Merge all dataframes
merged_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save the merged file
merged_df.to_csv('merged_matches.csv', index=False)
