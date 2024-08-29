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
    #  ("Va te faire foutre, espèce de connard !", 'fr'),
    # ("T'es qu'une merde, rien de plus.", 'fr'),
    # ("Personne ne veut de toi ici, va te faire voir.", 'fr'),
    # ("T'es vraiment qu'un imbécile, personne ne peut te supporter.", 'fr'),
    # ("Tu ferais mieux de disparaître, le monde serait mieux sans toi.", 'fr'),
    # ("T'es qu'un bon à rien, tu ne fais jamais rien de bien.", 'fr'),
    # ("Va jouer ailleurs, t'es trop stupide pour comprendre.", 'fr'),
    # ("T'es juste une merde humaine, tu ne sers à rien.", 'fr'),
    # ("Ferme-la, t’es qu’un gros porc.", 'fr'),
    # ("Ton existence est une erreur, personne ne te supporte.", 'fr'),
    # ("T’es qu’un raté, tu ne réussiras jamais rien.", 'fr'),
    # ("T’es un sous-homme, t’es rien pour moi.", 'fr'),
    # ("Tu es un parasite, ne reste pas ici.", 'fr'),
    # ("Va crever, tu es une honte pour la société.", 'fr'),
    # ("T’es un débile profond, incapable de comprendre quoi que ce soit.", 'fr'),
    # ("Personne ne veut t’entendre, va t’en.", 'fr'),
    # ("Tu es un fardeau, toujours à faire des bêtises.", 'fr'),
    # ("T’es qu’une sous-merde, jamais à la hauteur.", 'fr'),
    # ("Tu n’es qu’un incapable, jamais à la hauteur.", 'fr'),
    # ("Ton absence ferait plaisir à tout le monde, tu ne sers à rien.", 'fr'),
    # ("Pouvez-vous me donner des informations sur les crédits immobiliers ?", 'fr'), #####
    # ("Je voudrais consulter mon solde bancaire.", 'fr'),
    # ("Comment puis-je obtenir un relevé de compte ?", 'fr'),
    # ("J'ai besoin d'aide pour réinitialiser mon mot de passe.", 'fr'),
    # ("Quel est le taux d'intérêt actuel pour un prêt personnel ?", 'fr'),
    # ("Pouvez-vous me dire comment effectuer un virement bancaire ?", 'fr'),
    # ("Je voudrais savoir comment activer ma carte de crédit.", 'fr'),
    # ("Quels sont les horaires d'ouverture de votre agence ?", 'fr'),
    # ("Comment puis-je déposer un chèque ?", 'fr'),
    # ("J'aimerais modifier mon adresse de correspondance.", 'fr'),
    # ("Quels documents sont nécessaires pour un prêt étudiant ?", 'fr'),
    # ("Pouvez-vous m'expliquer les frais bancaires de mon compte ?", 'fr'),
    # ("Je voudrais savoir si vous proposez des comptes épargne.", 'fr'),
    # ("Comment puis-je bloquer ma carte bancaire en cas de perte ?", 'fr'),
    # ("Je veux savoir comment faire une réclamation.", 'fr'),
    # ("Quels services en ligne proposez-vous ?", 'fr'),
    # ("Je souhaite programmer un rendez-vous avec un conseiller.", 'fr'),
    # ("Comment puis-je mettre à jour mes informations personnelles ?", 'fr'),
    # ("J'aimerais en savoir plus sur vos offres de prêts auto.", 'fr'),
    # ("Bonjour, j'aimerais ouvrir un compte.", 'fr')

# insultes 10

    # ("Espèce d'imbécile !", 'fr'),
    # ("Tu n'es qu'un bon à rien.", 'fr'),
    # ("Va te faire foutre.", 'fr'),
    # ("Quel abruti !", 'fr'),
    # ("Connard !", 'fr'),
    # ("T'es vraiment un idiot fini.", 'fr'),
    # ("Sale hypocrite !", 'fr'),
    # ("Fermes ta gueule !", 'fr'),
    # ("Tu es une vraie ordure.", 'fr'),
    # ("T'es qu'un petit con.", 'fr')

 # Insults
    # ("T'es une vraie merde, tu sers à rien.", 'fr'),
    # ("Ferme ta gueule, espèce de con !", 'fr'),
    # ("T'es qu'un putain d'incapable.", 'fr'),
    # ("Tu ferais mieux de crever, tout le monde s'en porterait mieux.", 'fr'),
    # ("Je te hais, t'es qu'une merde humaine.", 'fr'),
    # ("Tu n'es qu'un déchet, personne ne t'aime.", 'fr'),
    # ("Va te pendre, t'es qu'un perdant.", 'fr'),
    # ("T'es un gros con, c'est tout.", 'fr'),
    # ("Personne ne veut de toi, casse-toi.", 'fr'),
    # ("T'es qu'un foutu parasite, dégage !", 'fr'),
    # ("Tu fais honte à tout le monde, dégage.", 'fr'),
    # ("Ton existence est une erreur, personne ne te supporte.", 'fr'),
    # ("Ferme-la, t'es qu'une nuisance.", 'fr'),
    # ("Va te noyer, t'es qu'un poids mort.", 'fr'),
    # ("Tu me dégoûtes, espèce de crétin.", 'fr'),
    # ("T'es qu'un bon à rien, tu fais tout de travers.", 'fr'),
    # ("Tout ce que tu fais, c'est de la merde.", 'fr'),
    # ("Ton existence est un fardeau pour tout le monde.", 'fr'),
    # ("T'es qu'un putain de boulet.", 'fr'),
    # ("Personne ne veut t'entendre, casse-toi.", 'fr'),





    # # Non-Insults
    # ("Pouvez-vous m'aider avec ma carte bancaire ?", 'fr'),
    # ("Je voudrais savoir comment déposer de l'argent sur mon compte.", 'fr'),
    # ("Quels sont les tarifs pour ouvrir un compte ?", 'fr'),
    # ("Comment puis-je vérifier mes transactions récentes ?", 'fr'),
    # ("Je veux changer mon mot de passe en ligne.", 'fr'),
    # ("Pouvez-vous me fournir un relevé de compte mensuel ?", 'fr'),
    # ("Comment puis-je demander un prêt étudiant ?", 'fr'),
    # ("J'aimerais savoir comment activer ma nouvelle carte bancaire.", 'fr'),
    # ("Je veux savoir comment configurer un virement automatique.", 'fr'),
    # ("Quels sont les services offerts pour les comptes professionnels ?", 'fr'),
    # ("Comment puis-je contacter le service client ?", 'fr'),
    # ("Je veux savoir si ma carte bancaire est bloquée.", 'fr'),
    # ("Pouvez-vous m'aider à comprendre les frais bancaires ?", 'fr'),
    # ("Je souhaite fermer mon compte bancaire.", 'fr'),
    # ("Comment puis-je transférer de l'argent à l'étranger ?", 'fr'),
    # ("Je voudrais connaître les conditions pour un prêt personnel.", 'fr'),
    # ("Comment puis-je ajouter un bénéficiaire à mon compte ?", 'fr'),
    # ("Quels sont les avantages d'un compte épargne ?", 'fr'),
    # ("Je veux en savoir plus sur vos options de prêt immobilier.", 'fr'),
    # ("J'aimerais prendre un rendez-vous avec un conseiller financier.", 'fr')
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
