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

# Define Arabic stopwords (sample list)
arabic_stopwords = set([
    'و', 'في', 'من', 'إلى', 'على', 'مع', 'كان', 'عن', 'هذا', 'ذلك', 'هؤلاء', 'ما', 'ليس', 'له', 'من', 'كانوا', 'أين', 'ماذا', 'إذا'
    # Add more stopwords as needed
])

# Load the saved model and vectorizer
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the LabelEncoder with the same labels used during training
sample_labels = ['non-insult', 'insult']  # Replace with your actual labels
label_encoder = LabelEncoder()
label_encoder.fit(sample_labels)

# Initialize text processing tools
stop_words = set(stopwords.words('english'))
stop_words.update(arabic_stopwords)  # Add Arabic stopwords
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text, lang='en'):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = text.split()  # Tokenization
    if lang == 'ar':
        tokens = [word for word in tokens if word not in arabic_stopwords]  # Removing Arabic stop words
    else:
        tokens = [word for word in tokens if word not in stop_words]  # Removing English stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Test cases for different languages
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


for text, lang in test_cases:
    cleaned_text = clean_text(text, lang)
    X_new = loaded_vectorizer.transform([cleaned_text]).toarray()
    prediction = loaded_model.predict(X_new)
    predicted_label = label_encoder.inverse_transform(prediction)
    print(f"Input: {text} (Language: {lang})")
    print(f"Predicted Label: {predicted_label[0]}")
    print("-" * 50)
