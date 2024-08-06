import pandas as pd

# Load the bad_words CSV file
file_path = 'bad_words.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Create a new DataFrame with 'text', 'label', and 'language'
new_df = pd.DataFrame()
new_df['text'] = df['bad_word']       # Use the 'bad_word' column as 'text'
new_df['label'] = 'insult'            # Set the label to 'insult'
new_df['language'] = df['language_code']  # Use the 'language_code' column for language

# Save the new DataFrame to a new CSV file
new_df.to_csv('processed_bad_words.csv', index=False)

print("Processed CSV file created successfully.")
