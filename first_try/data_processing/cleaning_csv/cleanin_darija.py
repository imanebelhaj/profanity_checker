import pandas as pd

# Load the Darija dataset
darija_df = pd.read_csv('Darija_Dataset.csv')

# Add the language column
darija_df['language'] = 'darija'

# Filter out rows where label is 0
darija_df = darija_df[darija_df['label'] == 1]

# Replace label 1 with the word 'insult'
darija_df['label'] = 'insult'

# Keep only the 'text', 'label', and 'language' columns
darija_df = darija_df[['text', 'label', 'language']]

# Save the updated dataset
darija_df.to_csv('processed_darija_dataset.csv', index=False)

print("Processed Darija dataset saved successfully.")
