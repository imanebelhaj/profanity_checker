import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

# Initialize tools
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load the saved model and vectorizer
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the LabelEncoder with the same labels used during training
sample_labels = ['non-insult', 'insult']  # Replace with your actual labels
label_encoder = LabelEncoder()
label_encoder.fit(sample_labels)

# Initialize text processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text, lang='en'):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = text.split()  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Removing English stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)


# test_cases = [
#     # Non-insult examples
#     ("nice", 'en'),
#     ("i need help creating a new bank aacoount", 'en'),
#     ("i have problem with the bank app.", 'en'),
#     ("my card is blocked", 'en'),
#     ("i have an issue", 'en'),
#     ("give me time when banks are open", 'en'),
#     ("closest bank , banks neaby", 'en'),

#     # Insult examples
#     ("fuck you ", 'en'),
#     ("dirty peice of shit.", 'en'),
#     ("ugly", 'en'),
#     ("youre useless", 'en'),
#     ("bitch.", 'en'),
#     ("whore", 'en'),
#     ("mother fucker", 'en')
# ]



# test_cases = [
#     # Non-insult cases
#     ("Thank you for your help.", 'en'),
#     ("I appreciate your support.", 'en'),
#     ("You did a great job on this project.", 'en'),
#     ("I am really glad we had this conversation.", 'en'),
#     ("Your effort is truly commendable.", 'en'),
#     ("I value your opinion on this matter.", 'en'),
#     ("This is a wonderful suggestion.", 'en'),
#     ("You have been very helpful.", 'en'),
#     ("I look forward to working with you again.", 'en'),
#     ("Your feedback was very constructive.", 'en'),
#     ("I am pleased with the results.", 'en'),
#     ("You are doing a fantastic job.", 'en'),
#     ("Thank you for being understanding.", 'en'),
#     ("I enjoyed our meeting today.", 'en'),
#     ("Your dedication is impressive.", 'en'),
#     ("I appreciate your attention to detail.", 'en'),
#     ("This solution works perfectly.", 'en'),
#     ("You handled the situation well.", 'en'),
#     ("Your contributions have been valuable.", 'en'),
#     ("I am happy with the progress we’ve made.", 'en'),
# ]



test_cases = [
    # Non-insult examples
    ("Thank you, you're incredibly helpful!", 'en'),
    ("This service is exactly what I needed, appreciate your assistance.", 'en'),
    ("My experience with the bank has been excellent, keep up the great work.", 'en'),
    ("The support I received today was top-notch.", 'en'),
    ("Everything was explained so clearly, I'm very satisfied.", 'en'),
    ("The agent was very understanding and resolved my issue quickly.", 'en'),
    ("I'm pleased with how smoothly my transaction was processed.", 'en'),
    ("The chatbot answered all my questions efficiently.", 'en'),
    ("Thanks to your help, everything is sorted now.", 'en'),
    ("The process was seamless and stress-free, great job.", 'en'),

    # Insult examples
    ("You're a complete idiot, how hard is it to do your job?", 'en'),
    ("This bank is full of morons, can't believe how dumb you all are.", 'en'),
    ("You're all a bunch of incompetent assholes.", 'en'),
    ("Your service is absolute shit, I've had enough of this crap.", 'en'),
    ("I've never seen such stupidity in my life, what a joke.", 'en'),
    ("You're fucking useless, can't even handle a simple request.", 'en'),
    ("This is bullshit, your bank is a total scam.", 'en'),
    ("You're a pathetic excuse for customer service.", 'en'),
    ("Your whole team is worthless, can't get anything right.", 'en'),
    ("This is fucking ridiculous, do your damn job properly.", 'en')
]


# Counters for results
insult_count = 0
non_insult_count = 0

# Process each test case
for text, lang in test_cases:
    cleaned_text = clean_text(text, lang)
    X_new = loaded_vectorizer.transform([cleaned_text]).toarray()
    prediction = loaded_model.predict(X_new)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    # Update counters based on the prediction
    if predicted_label == 'insult':
        insult_count += 1
    elif predicted_label == 'non-insult':
        non_insult_count += 1
    
    # Print the results
    print(f"Input: {text} (Language: {lang})")
    print(f"Predicted Label: {predicted_label}")
    print("-" * 50)

# Print the final counts
print("Total Insults:", insult_count)
print("Total Non-Insults:", non_insult_count)
