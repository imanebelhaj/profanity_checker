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
arabic_stopwords = set(['و', 'في', 'من', 'إلى', 'على', 'مع', 'كان', 'عن'])

# Load the saved model and vectorizer
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the LabelEncoder with the same labels used during training
sample_labels = ['non-insult', 'insult']
label_encoder = LabelEncoder()
label_encoder.fit(sample_labels)

# Initialize text processing tools
stop_words = set(stopwords.words('english'))
stop_words.update(arabic_stopwords)
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text, lang='en'):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    if lang == 'ar':
        tokens = [word for word in tokens if word not in arabic_stopwords]
    else:
        tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Test cases for different languages
test_cases = [
    ("you're amazing", 'en'),
    ("you are amazing", 'en'),
    ("You're not serious", 'en'),
    ("You're shit", 'en'),
    ("You're a bitch", 'en'),
    ("You're fat", 'en'),
    ("You're", 'en'),
]

for text, lang in test_cases:
    cleaned_text = clean_text(text, lang)
    X_new = loaded_vectorizer.transform([cleaned_text]).toarray()
    prediction = loaded_model.predict(X_new)
    predicted_label = label_encoder.inverse_transform(prediction)
    print(f"Input: {text} (Language: {lang})")
    print(f"Predicted Label: {predicted_label[0]}")
    print("-" * 50)
