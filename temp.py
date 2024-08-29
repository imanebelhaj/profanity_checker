import pandas as pd

# Load the existing and new datasets
file_path_existing = 'data300en.csv'
file_path_new = 'data1000000en.csv'

# Load the datasets
df_existing = pd.read_csv(file_path_existing)
df_new = pd.read_csv(file_path_new)

# Ensure that both DataFrames have the same columns
# If columns are not same, adjust this accordingly
if list(df_existing.columns) != list(df_new.columns):
    raise ValueError("The columns in the datasets do not match.")

# Combine the datasets
df_combined = pd.concat([df_existing, df_new], ignore_index=True)

# Save the merged dataset to a new CSV file
output_file_path = 'engfinal.csv'
df_combined.to_csv(output_file_path, index=False)

print(f"Datasets have been merged and saved to {output_file_path}")
