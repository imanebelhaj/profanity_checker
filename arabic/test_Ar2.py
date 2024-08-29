import pandas as pd
import re
import joblib

# Load Arabic stopwords from a file
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords_ar = file.read().splitlines()
    return set(stopwords_ar)

# Load stopwords
stop_words_ar = load_stopwords('arabic_stop_words.txt')

# Load the saved model and vectorizer
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function for text cleaning
def clean_text(text, lang='ar'):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = text.split()  # Tokenization
    
    if lang == 'ar':
        tokens = [word for word in tokens if word not in stop_words_ar]  # Removing Arabic stop words
    return ' '.join(tokens)

# Example prediction function
def predict(text, lang='ar'):
    cleaned_text = clean_text(text, lang)
    X_new = loaded_vectorizer.transform([cleaned_text])
    prediction = loaded_model.predict(X_new)
    return label_encoder.inverse_transform(prediction)[0]

# Define test cases
test_cases = [
    ("أنت غبي جدا", 'ar'),  # Insult
    ("كلامك تافه", 'ar'),  # Insult
    ("أنت شخص محترم", 'ar'),  # Non-insult
    ("شكرا لك على مساعدتك", 'ar'),  # Non-insult
    ("أنت أفضل من الجميع", 'ar'),  # Non-insult
    ("لماذا تتحدث بهذه الطريقة؟", 'ar'),  # Non-insult
    ("أنت شخص بغيض", 'ar'),  # Insult
    ("وجودك هنا ليس له معنى", 'ar'),  # Insult
    ("كل ما تفعله هو محاولة الإزعاج", 'ar'),  # Insult
    ("أنا سعيد بلقائك", 'ar')  # Non-insult
]

# Load label encoder
label_encoder = joblib.load('label_encoder.pkl')

# Function to run predictions
def run_tests(test_cases):
    for text, lang in test_cases:
        prediction = predict(text, lang)
        print(f"Text: {text}")
        print(f"Prediction: {prediction}")
        print("------")

# Run the defined test cases
run_tests(test_cases)
