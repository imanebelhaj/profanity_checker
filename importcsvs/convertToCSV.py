import pandas as pd

# Load the newly created CSV file
csv_file_path = 'combined_data.csv'  # Replace with your CSV file path
output_data = pd.read_csv(csv_file_path)

# Print the columns and the first few rows of the CSV file
print("Columns in the CSV file:")
print(output_data.columns)

print("\nFirst few rows of the CSV file:")
print(output_data.head())
