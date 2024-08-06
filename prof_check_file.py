from profanity_check import predict
import pandas as pd

# Load your dataset
file_path = 'bank_dataset.csv'  # Replace with your file path
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()
except pd.errors.EmptyDataError:
    print("No data found in the file.")
    exit()
except pd.errors.ParserError:
    print("Error parsing the file.")
    exit()

# Define a function to check for profanity
def check_profanity(text):
    try:
        return 'insult' if predict([text])[0] == 1 else 'non-insult'
    except Exception as e:
        print(f"Error processing text: {text} - {e}")
        return 'error'

# Apply the profanity check to your 'text' column
try:
    df['label'] = df['text'].apply(check_profanity)
except KeyError:
    print("Column 'text' not found in the dataset.")
    exit()

# Save the updated DataFrame to a new CSV file
output_file_path = 'updated_bank.csv'
try:
    df.to_csv(output_file_path, index=False)
    print("Updated CSV file created successfully.")
except Exception as e:
    print(f"Error saving the updated CSV file: {e}")
