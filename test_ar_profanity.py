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
from nltk.stem import WordNetLemmatizer

# Load Arabic stopwords from a file
with open('arabic_stop_words.txt', 'r', encoding='utf-8') as file:
    arabic_stopwords = set(file.read().splitlines())

# Load the saved model and vectorizer
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the LabelEncoder with the same labels used during training
sample_labels = ['non-insult', 'insult']
label_encoder = LabelEncoder()
label_encoder.fit(sample_labels)

# Initialize text processing tools
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text, lang='ar'):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenize text into words
    
    # Remove Arabic stopwords
    if lang == 'ar':
        tokens = [word for word in tokens if word not in arabic_stopwords]
    
    # Lemmatize the tokens (optional)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

test_cases = [
    # Insult cases
    ("أنت أكثر بؤسًا من كلب ضال", 'ar'),
    ("لا تستحق حتى أن تكون برازًا على حذائي", 'ar'),
    ("أنت نفايات بشرية، لا قيمة لك", 'ar'),
    ("وجهك يسبب الغثيان أكثر من أي شيء آخر", 'ar'),
    ("وجودك هنا يلوث هذا المكان", 'ar'),
    ("أنت عار على كل إنسانية", 'ar'),
    ("لم أرى شخصًا أكثر بؤسًا منك في حياتي", 'ar'),
    ("عقلك أصغر من عقل حشرة", 'ar'),
    ("كل ما تفعله هو إضاعة الهواء الثمين", 'ar'),
    ("وجودك هنا يشبه قذارة من الجحيم", 'ar'),

    # Non-insult cases
    ("كيف حالك اليوم؟", 'ar'),
    ("شكراً لمساعدتك", 'ar'),
    ("أتمنى لك يوماً سعيداً", 'ar'),
    ("هل تحتاج إلى أي مساعدة؟", 'ar'),
    ("لقد قمت بعمل رائع", 'ar'),
    ("أنت إنسان طيب", 'ar'),
    ("أنا ممتن لك", 'ar'),
    ("أنت شخص مميز", 'ar'),
    ("أشكرك على جهدك", 'ar'),
    ("عملك ممتاز", 'ar')
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
