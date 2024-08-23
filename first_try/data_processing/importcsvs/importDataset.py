from datasets import load_dataset
import pandas as pd

# ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

# Extract the 'instruction' column
# Assuming 'train' split is available; change to 'test' or 'validation' if needed
data = ds['train']  # Adjust split as needed
instruction_column = data['instruction']

# Create a DataFrame
output_data = pd.DataFrame()
output_data['content'] = instruction_column
output_data['label'] = 'non-insult'  # Set default label for all rows
output_data['language'] = 'eng'  # Set language for all rows

# Save to CSV
output_data.to_csv('instructions_data.csv', index=False)

print("CSV file created successfully.")