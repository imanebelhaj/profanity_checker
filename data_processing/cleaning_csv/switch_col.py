import pandas as pd

# Load the existing CSV file
df = pd.read_csv('updated_dataset.csv')

# Rename the 'content' column to 'text'
df.rename(columns={'content': 'text'}, inplace=True)

# Reorder the columns
df = df[['text', 'label', 'language']]

# Save the updated DataFrame to a new CSV file
df.to_csv('reordered_dataset.csv', index=False)

print("Reordered CSV file created successfully.")
