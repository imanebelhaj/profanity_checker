import pandas as pd

# Load your dataset
df = pd.read_csv('final.csv')

# Check the distribution of each class
class_counts = df['label'].value_counts()
total_samples = len(df)

# Print class distribution
print("Class Distribution:")
print(class_counts)

# Calculate and print the percentage of each class
class_percentages = (class_counts / total_samples) * 100
print("\nClass Percentages:")
print(class_percentages)

# Determine the imbalance ratio
most_common_class = class_counts.idxmax()
least_common_class = class_counts.idxmin()
imbalance_ratio = class_counts[most_common_class] / class_counts[least_common_class]

print("\nImbalance Ratio (Most Common / Least Common):")
print(imbalance_ratio)

# Optionally, calculate the number of samples needed to balance the dataset
# Assuming you want a 1:1 balance
target_class_count = class_counts[most_common_class]
additional_samples_needed = (target_class_count - class_counts[least_common_class])

print("\nAdditional Samples Needed to Balance Dataset:")
print(additional_samples_needed)
