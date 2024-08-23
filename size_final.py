import pandas as pd

# Load the CSV file
df = pd.read_csv('balanced_data5.csv')

# Count the occurrences of each label in each language
language_label_counts = df.groupby(['language', 'label']).size().unstack(fill_value=0)

# Display the results
print(language_label_counts)
