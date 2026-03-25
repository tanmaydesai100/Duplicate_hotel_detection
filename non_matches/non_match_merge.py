import pandas as pd

# read each file
df1 = pd.read_csv('non_match1.csv')
df2 = pd.read_csv('non_match2.csv')

# concatenate them (vertical stack)
merged = pd.concat([df1, df2], ignore_index=True)

# write out
merged.to_csv('merged_non_match.csv', index=False)
