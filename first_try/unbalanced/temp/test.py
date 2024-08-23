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
# Assuming you have a sample of labels used in the training process
# If not, you should replace the following with the actual labels or re-fit with training labels
sample_labels = ['non-insult', 'insult']  # Replace with your actual labels
label_encoder = LabelEncoder()
label_encoder.fit(sample_labels)

# Initialize text processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = text.split()  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Removing stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# List of test cases
test_cases = [
    "You are so stupid and worthless!",
    "I really enjoyed the movie last night.",
    "I hate this product. It's the worst.",
    "Thank you for your help, I appreciate it.",
    "What a fantastic job you did!",
    "You are a genius! This is amazing.",
    "Why are you always so lazy?",
    "I'm excited for the new game release.",
    "This is a terrible service, I want a refund!",
    "Great work on the project, it looks awesome!"
]

# Process and predict test cases
for text in test_cases:
    cleaned_text = clean_text(text)
    X_new = loaded_vectorizer.transform([cleaned_text]).toarray()
    prediction = loaded_model.predict(X_new)
    predicted_label = label_encoder.inverse_transform(prediction)
    print(f"Input: {text}")
    print(f"Predicted Label: {predicted_label[0]}")
    print("-" * 50)
