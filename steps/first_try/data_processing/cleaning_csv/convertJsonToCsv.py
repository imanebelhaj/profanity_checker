import pandas as pd
import json

def process_json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create an empty DataFrame to collect all bad words
    df = pd.DataFrame(columns=['text', 'label', 'language'])

    # Process each language in the JSON
    for lang, bad_words in data.items():
        temp_df = pd.DataFrame(bad_words, columns=['text'])
        temp_df['label'] = 'insult'
        temp_df['language'] = lang
        df = pd.concat([df, temp_df], ignore_index=True)
    
    # Save to CSV
    df.to_csv(csv_file, index=False, encoding='utf-8')

# Call the function with your file names
process_json_to_csv('swear.json', 'bad_words.csv')
