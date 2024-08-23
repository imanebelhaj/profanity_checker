import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Initialize NLTK tools
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Stop words for multiple languages
stop_words_en = set(stopwords.words('english'))
stop_words_ar = {'في', 'من', 'على', 'إلى', 'و', 'أن', 'أنها', 'أنهم', 'لكن', 'كان', 'بأن', 'كانوا'}
stop_words_darija = {'ف', 'ع', 'على', 'من', 'و', 'أ', 'ش', 'هذا', 'ت'}
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text, lang='en'):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = text.split()  # Tokenization
    
    if lang == 'ar':
        tokens = [word for word in tokens if word not in stop_words_ar]  # Removing Arabic stop words
    elif lang == 'darija':
        tokens = [word for word in tokens if word not in stop_words_darija]  # Removing Darija stop words
    else:
        tokens = [word for word in tokens if word not in stop_words_en]  # Removing English stop words
        
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('balanced_data5.csv')

# Handle missing values
df['text'] = df['text'].astype(str)
df['text'] = df['text'].fillna('')

# Apply text cleaning
df['text'] = df.apply(lambda row: clean_text(row['text'], row['language']), axis=1)

# Encoding labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(df['text']).toarray()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Define the model
model = LogisticRegression()

# Define the parameter grid for GridSearch
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Perform Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Load the saved model and vectorizer for future predictions
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example prediction
def predict(text, lang='en'):
    cleaned_text = clean_text(text, lang)
    X_new = loaded_vectorizer.transform([cleaned_text]).toarray()
    prediction = loaded_model.predict(X_new)
    return label_encoder.inverse_transform(prediction)[0]

# Test the prediction function
test_cases = [
    # Non-insult examples
    ("You're a great colleague", 'en'),
    ("I love your work", 'en')
]

for text, lang in test_cases:
    print(f"Input: {text} (Language: {lang})")
    print(f"Predicted Label: {predict(text, lang)}")
    print("-" * 50)
