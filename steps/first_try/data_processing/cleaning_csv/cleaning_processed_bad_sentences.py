import pandas as pd
import re

# Load the original dataset
file_path = 'processed_bad_sentences.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Function to clean text data
def clean_text(text):
    # Remove leading and trailing whitespace
    text = text.strip()
    # Remove special characters and multiple spaces
    text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Apply the cleaning function to the 'text' column
df['text'] = df['text'].apply(clean_text)

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_bad_sentences.csv', index=False)

# Function to print only the insults
def print_insults(file_path):
    df = pd.read_csv(file_path)
    insults_df = df[df['label'] == 'insult']
    print(insults_df[['text', 'label', 'language']])

# Print the insults from the cleaned dataset
print_insults('cleaned_bad_sentences.csv')
