import pandas as pd
import random

# Load the existing dataset
balanced_df = pd.read_csv('balanced_data5.csv')

# Separate the data by labels and language (Arabic)
non_insult_ar_df = balanced_df[(balanced_df['label'] == 'non-insult') & (balanced_df['language'] == 'ar')]
insult_ar_df = balanced_df[(balanced_df['label'] == 'insult') & (balanced_df['language'] == 'ar')]

# Count the existing sentences in each category
non_insult_count = len(non_insult_ar_df)
insult_count = len(insult_ar_df)

# Number of sentences to add for each label
required_count = 50000
non_insult_needed = required_count - non_insult_count
insult_needed = required_count - insult_count

# Print the current counts and needed sentences
print(f"Current non-insult count in Arabic: {non_insult_count}")
print(f"Current insult count in Arabic: {insult_count}")
print(f"Non-insult sentences needed: {non_insult_needed}")
print(f"Insult sentences needed: {insult_needed}")

if non_insult_needed > 0 or insult_needed > 0:
    # Sample sentence templates for variations in Arabic
    non_insult_templates = [
        "شكراً لمساعدتك.",
        "تلقيت رسالتك.",
        "أنا هنا لمساعدتك.",
        "ملاحظاتك مهمة لنا.",
        "عملنا معك كان جيداً.",
        "شكراً لصبرك.",
        "أنا سعيد بمساعدتك في أي شيء.",
        "أنت إضافة رائعة للفريق.",
        "ملاحظاتك مقدرة.",
        "دائماً هنا للمساعدة."
    ]

    insult_templates = [
        "أنت غبي جداً.",
        "العمل معك مزعج.",
        "أنت لا تفهم شيئاً.",
        "أنت مزعج جداً.",
        "أنت سيء.",
        "لا أفهم ماذا تريد.",
        "أنت بلا فائدة.",
        "لا تتوقع أي خير منك.",
        "العمل معك متعب.",
        "تعبنا منك."
    ]

    # Generate more non-insult and insult sentences
    new_non_insult_sentences = [random.choice(non_insult_templates) for _ in range(non_insult_needed)]
    new_insult_sentences = [random.choice(insult_templates) for _ in range(insult_needed)]

    print(f"Generated {len(new_non_insult_sentences)} non-insult sentences.")
    print(f"Generated {len(new_insult_sentences)} insult sentences.")

    # Create DataFrames for new data
    new_non_insult_df = pd.DataFrame({
        'text': new_non_insult_sentences,
        'label': ['non-insult'] * len(new_non_insult_sentences),
        'language': ['ar'] * len(new_non_insult_sentences)
    })

    new_insult_df = pd.DataFrame({
        'text': new_insult_sentences,
        'label': ['insult'] * len(new_insult_sentences),
        'language': ['ar'] * len(new_insult_sentences)
    })

    # Append new data to the existing balanced data
    balanced_df_updated = pd.concat([balanced_df, new_non_insult_df, new_insult_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    balanced_df_updated.to_csv('balanced_data5.csv', index=False)

    print(f"Added {len(new_non_insult_sentences) + len(new_insult_sentences)} Arabic sentences to balanced_data5.csv")
else:
    print("The dataset already has enough sentences for both categories in Arabic.")
