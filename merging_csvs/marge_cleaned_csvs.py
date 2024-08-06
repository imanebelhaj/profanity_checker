import pandas as pd

# Load the two CSV files
df1 = pd.read_csv('standardized_all.csv')
df2 = pd.read_csv('updated_bank.csv')

# Concatenate the DataFrames
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_final.csv', index=False)

print("Merged CSV file created successfully.")
