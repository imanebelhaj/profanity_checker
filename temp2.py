import pandas as pd

# Load the dataset
file_path = 'balanceddata5.csv'
df = pd.read_csv(file_path)

# Display the first few rows to check the structure
print(df.head())

# Remove rows where 'language' is 'ar' or 'darija'
df_cleaned = df[~df['language'].isin(['ar', 'darija','fr'])]



# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv('y.csv', index=False)

print("Filtered dataset saved to 'balanced_data5_cleaned.csv'.")
