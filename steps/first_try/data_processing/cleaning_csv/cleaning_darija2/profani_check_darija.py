import pandas as pd
import string

# Load the main dataset
main_df = pd.read_csv('cleaned_darija_with_empty_labels.csv')  # Replace with your actual file name

# Load the insults dataset
insults_df = pd.read_csv('processed_darija_dataset.csv')  # Replace with your actual file name

# Assuming the insults dataset has a column named 'text' that contains the insults
insult_list = insults_df['text'].str.lower().tolist()  # Extract insults and convert to lowercase

# Function to remove punctuation and check for profanity
def check_for_profanity(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Check if any word in the text is in the insult list
    words = text.split()
    for word in words:
        if word in insult_list:
            return 'insult'
    return 'non-insult'

# Assuming 'text' is the column containing the text data in your main dataset
main_df['label'] = main_df['text'].apply(check_for_profanity)

# Display or save results
print(main_df[['text', 'label']].head())  # Display the first few rows of the dataframe with labels
main_df.to_csv('dataset_with_labels.csv', index=False)  # Save the updated dataset to a new CSV file

print("Profanity check completed and results saved.")
