import pandas as pd

# Load the CSV file
df = pd.read_csv('dataset2_cruc.csv')

# Add the index as a column named 'Id_datasetA' at the start
df.insert(0, 'Id_datasetB', df.index)

# Create the 'modified_address' column by concatenating 'name', a comma, and 'address'
df['modified_address'] = df['name'] + ', ' + df['address']

# Save the modified DataFrame to a new CSV file
df.to_csv('dataset2_final.csv', index=False)
