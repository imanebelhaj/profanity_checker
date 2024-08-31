# import pandas as pd
# from profanity_check import predict

# # Load the cleaned CSV file
# file_path = 'cleaned_instructions_data.csv'  # Replace with your actual file path
# df = pd.read_csv(file_path)

# # Apply profanity check
# df['is_profane'] = predict(df['content'].tolist())

# # Map prediction to label
# df['label'] = df['is_profane'].apply(lambda x: 'insult' if x == 1 else 'non-insult')
# df['language'] = 'eng'  # Assuming all content is in English

# # Save the updated DataFrame to a new CSV file
# df.to_csv('final_instructions_data_with_labels.csv', index=False)

# print("CSV file with profanity labels created successfully.")

import pandas as pd

# Load the updated CSV file
file_path = 'updated_bank.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Filter the rows where the label is 'insult'
insults_df = df[df['label'] == 'insult']

# Print the filtered DataFrame
print(insults_df)
