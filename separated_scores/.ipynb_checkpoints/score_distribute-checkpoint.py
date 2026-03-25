import pandas as pd

# Load the original matches file
df = pd.read_csv("../matches.csv")

# Ensure score column is float
df["score"] = df["score"].astype(float)

# Define score ranges
range1 = df[(df["score"] >= 0.65) & (df["score"] < 0.80)]
range2 = df[(df["score"] >= 0.80) & (df["score"] < 0.90)]
range3 = df[(df["score"] >= 0.90) & (df["score"] <= 1.00)]

# Save each range to its own CSV file
range1.to_csv("matches_065_079.csv", index=False)
range2.to_csv("matches_080_089.csv", index=False)
range3.to_csv("matches_090_100.csv", index=False)

print("✅ Split into 3 files based on score ranges.")
