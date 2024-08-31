import pandas as pd
import string

# Load the dataset
df = pd.read_csv('merged_darija_data.csv')

# Function to remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Concatenate columns into a single 'text' column
text_columns = [
    'indef', 'def', 'n1', 'n2', 'n3', 'n4', 'eng', 'ana', 'nta', 'nti',
    'howa', 'hia', '7na', 'ntoma', 'homa', 'root', 'aafaf', 'darija',
    'english', 'specific1_darija', 'specific2_darija', 'general_darija',
    'darija_ar', 'root_darija', 'aala', 'masculine', 'feminine',
    'masc_plural', 'fem_plural', 'verb', 'noun', 'n5', 'n6'
]

# Create a new 'text' column by concatenating the specified columns
df['text'] = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)

# Remove punctuation from the 'text' column
df['text'] = df['text'].apply(remove_punctuation)

# Add 'language' and 'label' columns
df['language'] = 'darija'  # Assuming all data is in Darija
df['label'] = 'non-insult'  # Assuming all data is labeled as non-insult

# Select only the required columns
df = df[['text', 'language', 'label']]

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_darija_data.csv', index=False)

print("The dataset has been cleaned and saved as 'cleaned_darija_data.csv'")
