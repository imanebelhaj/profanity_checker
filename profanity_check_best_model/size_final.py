import pandas as pd

# Load the dataset
file_path = 'final.csv'
df = pd.read_csv(file_path)

# Display the total number of sentences
total_sentences = len(df)
print(f"Total number of sentences: {total_sentences}")

# Display the number of unique labels
unique_labels = df['label'].nunique()
print(f"Number of unique labels: {unique_labels}")

# Display the number of unique languages
unique_languages = df['language'].nunique()
print(f"Number of unique languages: {unique_languages}")

# Count the number of insults and non-insults
label_counts = df['label'].value_counts()
num_insults = label_counts.get('insult', 0)
num_non_insults = label_counts.get('non-insult', 0)

print(f"\nOverall Counts:")
print(f"Number of insults: {num_insults}")
print(f"Number of non-insults: {num_non_insults}")

# Count the number of insults and non-insults for each language
print("\nCounts per language:")
language_label_counts = df.groupby(['language', 'label']).size().unstack(fill_value=0)
print(language_label_counts)
