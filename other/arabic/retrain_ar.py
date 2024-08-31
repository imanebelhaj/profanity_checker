import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load Arabic stop words from the file
with open('arabic_stop_words.txt', 'r', encoding='utf-8') as file:
    arabic_stopwords = set(file.read().splitlines())

# Function for text cleaning
def clean_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = text.split()  # Tokenization
    tokens = [word for word in tokens if word not in arabic_stopwords]  # Removing stop words
    return ' '.join(tokens)

# Load your dataset
df = pd.read_csv('balanced_data5.csv')  # Update this path if necessary

# Filter Arabic data
arabic_df = df[df['language'] == 'ar']

# Handle missing values
arabic_df['text'] = arabic_df['text'].astype(str)
arabic_df['text'] = arabic_df['text'].fillna('')

# Apply text cleaning
arabic_df['text'] = arabic_df['text'].apply(clean_text)

# Load the existing model and vectorizer
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Vectorize the Arabic text data using the existing vectorizer
X_arabic = loaded_vectorizer.transform(arabic_df['text'])

# Encode labels if necessary
label_encoder = LabelEncoder()
y_arabic = label_encoder.fit_transform(arabic_df['label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_arabic, y_arabic, test_size=0.2, random_state=42)

# Retrain the model on the Arabic data
loaded_model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = loaded_model.predict(X_test)
print("Accuracy on Arabic data:", accuracy_score(y_test, y_pred))
print("Classification Report for Arabic data:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Arabic Insult Detection')
plt.show()

# Save the retrained model into the original file
joblib.dump(loaded_model, 'best_model.pkl')

