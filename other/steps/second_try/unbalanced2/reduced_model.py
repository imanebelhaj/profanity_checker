import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load your dataset
df = pd.read_csv('balanced_data5.csv')

# Initialize tools
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text):
    if not isinstance(text, str):  # Ensure text is a string
        return ''
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = text.split()  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Removing stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Handle missing values
df['text'] = df['text'].fillna('')

# Apply text cleaning
df['text'] = df['text'].apply(clean_text)

# Encoding labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Reduced features
X = tfidf_vectorizer.fit_transform(df['text']).toarray()

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

# Define the model
model = MultinomialNB()  # Faster model

# Define the parameter grid
param_dist = {
    'alpha': [0.1, 1, 10]
}

# Perform Randomized Search
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, scoring='accuracy', random_state=42)
random_search.fit(X_train, y_train)

# Get the best model
best_model = random_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model and vectorizer
import joblib
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Load the saved model and vectorizer for future predictions
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example prediction
new_text = "example text to classify"
cleaned_text = clean_text(new_text)
X_new = loaded_vectorizer.transform([cleaned_text]).toarray()
prediction = loaded_model.predict(X_new)
print("Predicted Label:", label_encoder.inverse_transform(prediction))
