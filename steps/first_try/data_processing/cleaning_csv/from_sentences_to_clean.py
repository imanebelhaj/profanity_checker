import pandas as pd

# Load the existing CSV file
file_path = 'bad_sentences.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Create a new DataFrame with the desired columns
output_data = pd.DataFrame()
output_data['text'] = df['tweet']
output_data['label'] = 'insult'
output_data['language'] = 'eng'

# Save the new DataFrame to a CSV file
output_file_path = 'formatted_data.csv'  # Replace with your desired output file path
output_data.to_csv(output_file_path, index=False)

print("New CSV file created successfully.")
