import pandas as pd

# 1. Load all columns as strings (avoids mixed-type issues)
df = pd.read_csv('nodes_output.csv', dtype=str, low_memory=False)

# 2. Auto-discover any column with keywords
keywords = ['email', 'url', 'website', 'facebook']
cols_to_merge = [c for c in df.columns if any(k in c.lower() for k in keywords)]

# 3. Blank/whitespace → NaN
df[cols_to_merge] = df[cols_to_merge].replace(r'^\s*$', pd.NA, regex=True)

# 4. Coalesce **all** non-null values per row, joined by commas
df['email'] = df[cols_to_merge].apply(
    lambda row: ','.join(str(v).strip() for v in row if pd.notna(v)),
    axis=1
)

# 5. Export only the new column
df[['email']].to_csv('only_email_column.csv', index=False)
print("Wrote only_email_column.csv with a single ‘email’ column.")
