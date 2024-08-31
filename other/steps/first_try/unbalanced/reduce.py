import pandas as pd

df = pd.read_csv('final.csv')
# Assuming your dataset is in a DataFrame 'df' with 'text' and 'label' columns
# Separate the non-insult and insult samples
non_insult_df = df[df['label'] == 'non-insult']
insult_df = df[df['label'] == 'insult']

# Randomly sample the non-insult DataFrame to match the number of insult samples
non_insult_reduced = non_insult_df.sample(n=len(insult_df), random_state=42)

# Combine the reduced non-insult samples with the insult samples
balanced_df = pd.concat([non_insult_reduced, insult_df])

# Shuffle the combined DataFrame
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the new class distribution
print(balanced_df['label'].value_counts())

