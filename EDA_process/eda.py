import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'final.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Basic Information:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst Few Rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Describe the dataset for numerical features
print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# Distribution of the 'label' column
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='label')
plt.title('Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Distribution of text length
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# Sample text examples
print("\nSample Text Examples:")
print(df[['text', 'label']].sample(10))

# Check for duplicates
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Value counts for languages (if applicable)
if 'language' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='language')
    plt.title('Distribution of Languages')
    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.show()

print("\nValue Counts for Labels:")
print(df['label'].value_counts())
 
#  Explanation of Each Step
# Load the Dataset: Reads the dataset into a DataFrame.
# Basic Information: Provides details about the DataFrame’s structure, including data types and non-null counts.
# First Few Rows: Shows a preview of the first few records to understand the data format.
# Missing Values: Identifies any missing values in the dataset.
# Descriptive Statistics: Provides statistical summaries of the dataset’s features.
# Label Distribution: Visualizes the distribution of different labels.
# Text Length Distribution: Analyzes the length of the text data.
# Sample Text Examples: Displays a few sample text entries for a quick look at the data.
# Duplicate Rows: Checks for and counts any duplicate entries in the dataset.
# Language Distribution: (If applicable) Shows the distribution of different languages if present.
# Value Counts for Labels: Displays the count of each label in the dataset.