import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Initialize NLTK tools
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define stopwords for each language
stopwords_dict = {
    'en': set(stopwords.words('english')),
    'fr': set(stopwords.words('french')),
    'ar': set(stopwords.words('arabic')),
    'darija': {'f', 'a', '3la', 'mn', 'w', 'b', 'hd', 't', 's', 'k', 'nt', 'lh', 'kifach', 'khdmt', 't9dr', 'hadi'}  # Customize as needed
}

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text, lang='en'):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = text.split()  # Tokenization
    
    if lang in stopwords_dict:
        tokens = [word for word in tokens if word not in stopwords_dict[lang]]  # Removing stop words
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Load your dataset
df = pd.read_csv('balanced_data5.csv')  # Update this path

# Handle missing values
df['text'] = df['text'].astype(str)
df['text'] = df['text'].fillna('')

# Apply text cleaning
df['text'] = df.apply(lambda row: clean_text(row['text'], row['language']), axis=1)

# Load the saved model and vectorizer
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize LabelEncoder
sample_labels = ['non-insult', 'insult']
label_encoder = LabelEncoder()
label_encoder.fit(sample_labels)

# Transform features
X = loaded_vectorizer.transform(df['text'])
y = df['label']
languages = df['language']

# Predict
y_pred = loaded_model.predict(X)

# Add predictions to the dataframe
df['predicted_label'] = label_encoder.inverse_transform(y_pred)

# Initialize lists for plotting
accuracy_list = []
languages_list = []

# Generate metrics and visualizations for each language
for lang in df['language'].unique():
    lang_data = df[df['language'] == lang]
    y_true = lang_data['label']
    y_pred_lang = lang_data['predicted_label']
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred_lang)
    accuracy_list.append(accuracy)
    languages_list.append(lang)
    print(f'Accuracy for language {lang}: {accuracy}')
    
    # Classification Report
    print(f'Classification Report for language {lang}:')
    print(classification_report(y_true, y_pred_lang, target_names=sample_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_lang)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sample_labels, yticklabels=sample_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Language {lang}')
    plt.show()

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=languages_list, y=accuracy_list, palette='viridis')
plt.xlabel('Languages')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison by Language')
plt.show()
