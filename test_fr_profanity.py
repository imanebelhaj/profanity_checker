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

# Define French stopwords
french_stopwords = set(stopwords.words('french'))

# Load the saved model and vectorizer
loaded_model = joblib.load('best_model.pkl')
loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize the LabelEncoder with the same labels used during training
sample_labels = ['non-insult', 'insult']  # Replace with your actual labels
label_encoder = LabelEncoder()
label_encoder.fit(sample_labels)

# Initialize text processing tools
stop_words = french_stopwords
lemmatizer = WordNetLemmatizer()

# Function for text cleaning
def clean_text(text, lang='fr'):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenize text into words
    
    # If language is French, remove French stop words
    if lang == 'fr':
        tokens = [word for word in tokens if word not in french_stopwords]
    
    # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Test cases for French
test_cases = [
    ("Vous êtes des salauds sans cervelle.", 'fr'),
    ("Vous n'êtes que des bâtards inutiles.", 'fr'),
    ("Votre idioticité est sans limites, espèce de débiles.", 'fr'),
    ("Vous êtes des andouilles, totalement incapables.", 'fr'),
    ("Vous ne méritez que le mépris, bande de losers.", 'fr'),
    ("Votre existence est une erreur, vous êtes des sous-hommes.", 'fr'),
    ("Vous êtes des crevards, vous ne valez rien.", 'fr'),
    ("Vous êtes des imbéciles finis, vous faites honte à l'humanité.", 'fr'),
    ("Vous êtes des putes sans scrupules, dégagez de ma vue.", 'fr'),
    ("Vous êtes des idiots pathétiques, vous ne comprenez rien.", 'fr'),
    ("Vous êtes des sous-chiens, bon à rien.", 'fr'),
    ("Vous êtes des fainéants de merde, incapable de faire quoi que ce soit.", 'fr'),
    ("Vous êtes des incapables, vous ne valez pas mieux que des parasites.", 'fr'),
    ("Votre stupidité est monumentale, allez vous faire voir.", 'fr'),
    ("Vous êtes des déchets humains, à peine dignes d'exister.", 'fr'),
    ("Vous êtes des connards détestables, tout ce que vous touchez se détruit.", 'fr'),
    ("Vous êtes des sous-merdes sans âme, vos efforts sont pitoyables.", 'fr'),
    ("Vous êtes des larves incapables, faites un effort pour changer.", 'fr'),
    ("Vous êtes des vermines insupportables, que faites-vous encore ici ?", 'fr'),
    ("Vous êtes des monstres d'ineptie, vous ne servez à rien.", 'fr')
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
