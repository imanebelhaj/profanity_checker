import pandas as pd

# Load your dataset
df = pd.read_csv('en.csv')  # Replace 'your_dataset.csv' with the actual filename

# Reorder the columns
df = df[['text', 'language', 'label']]

# Save the reordered dataset to a new CSV file
df.to_csv('en.csv', index=False)
