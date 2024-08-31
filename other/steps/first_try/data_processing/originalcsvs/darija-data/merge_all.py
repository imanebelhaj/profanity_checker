import pandas as pd
import os

# List of all CSV files to be merged
csv_files = [
    '(in)definite.csv', 'adjectives.csv', 'adverbs.csv', 'animals.csv', 'art.csv',
    'clothes.csv', 'colors.csv', 'conjug_past.csv', 'conjug_present.csv',
    'determiners.csv', 'economy.csv', 'education.csv', 'emotions.csv',
    'environment.csv', 'family.csv', 'femalenames.csv', 'food.csv', 'health.csv',
    'humanbody.csv', 'idioms.csv', 'imagenet_b_darija.csv', 'imperatives.csv',
    'malenames.csv', 'masculine_feminine_plural.csv', 'nouns.csv', 'numbers.csv',
    'places.csv', 'plants.csv', 'possessives.csv', 'prepositions.csv',
    'professions.csv', 'pronouns.csv', 'proverbs.csv', 'religion.csv',
    'sentences.csv', 'sentences2.csv', 'sport.csv', 'technology.csv', 'time.csv',
    'utils.csv', 'verb-to-noun.csv', 'verbs.csv', 'weird.csv'
]

# Initialize an empty list to hold the dataframes
dataframes = []

# Read each CSV file and append it to the list of dataframes
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Concatenate all dataframes into one
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_darija_data.csv', index=False)

print("All files have been merged into 'merged_darija_data.csv'")
