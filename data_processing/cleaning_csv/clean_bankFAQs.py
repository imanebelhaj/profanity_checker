import pandas as pd

# Load the 'bank' dataset
bank_df = pd.read_csv('BankFAQs.csv')  # Replace with your file path

# Remove the 'Class' column
bank_df = bank_df.drop(columns=['Class'])

# Combine 'Question' and 'Answer' into a single 'text' column
bank_df['text'] = bank_df['Question'] + ' ' + bank_df['Answer']

# Add the 'language' and 'label' columns
bank_df['language'] = 'en'
bank_df['label'] = 'label'  # Replace 'label' with the appropriate label if needed

# Select and reorder the columns
bank_df = bank_df[['text', 'language', 'label']]

# Save the modified DataFrame to a new CSV file
bank_df.to_csv('bank_dataset.csv', index=False)

print("Dataset processed and saved to 'modified_bank_dataset.csv'.")
