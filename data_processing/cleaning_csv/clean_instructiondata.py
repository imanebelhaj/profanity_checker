import pandas as pd

# Load the CSV file
file_path = 'instructions_data.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Remove the {{Order Number}} placeholder from the 'content' column
df['content'] = df['content'].str.replace('{{Order Number}}', '', regex=False)

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_instructions_data.csv', index=False)

print("CSV file with placeholders removed created successfully.")