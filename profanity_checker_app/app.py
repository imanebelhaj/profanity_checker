from flask import Flask, render_template, request, jsonify
import joblib
import nltk
import re
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the model and vectorizer
model = joblib.load('model/best_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Initialize the LabelEncoder with the same labels used during training
sample_labels = ['non-insult', 'insult']  # Replace with your actual labels if different
label_encoder = LabelEncoder()
label_encoder.fit(sample_labels)

# Load stopwords
stop_words_en = set(stopwords.words('english'))
stop_words_fr = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text, lang='en'):
    text = text.lower()  # Lowercasing
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    tokens = text.split()  # Tokenization 
    if lang == 'fr':
        tokens = [word for word in tokens if word not in stop_words_fr]  # Removing French stop words
    else:
        tokens = [word for word in tokens if word not in stop_words_en]  # Removing English stop words
        
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

# Prediction function
def predict_insult(text, lang='en'):
    cleaned_text = clean_text(text, lang)  # Clean the input text
    X_new = vectorizer.transform([cleaned_text]).toarray()
    prediction = model.predict(X_new)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction API route
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()  # Get JSON data
    if not data:
        print("No JSON data received")  # Debugging line
        return jsonify({'error': 'No JSON data received'}), 400
    if 'text' not in data or 'lang' not in data:
        print(f"Missing keys in data: {data}")  # Debugging line
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']
    lang = data['lang']
    print(f"Received text: {text}, lang: {lang}")  # Debugging line
    
    label = predict_insult(text, lang)
    result = "This is an insult" if label == 'insult' else "This is not an insult"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)



# @app.route('/predict',methods=['POST'])
# def predict():
#         data = request.get_json()
#         if data is None:
#             return jsonify(result="No JSON data received"), 400
#         text = request.form['text']
#         lang = request.form['lang']
#         label = predict_insult(text, lang)
#         if not text:
#             return jsonify(result="No text provided"), 400
#         label = predict_insult(text, lang)
#         result = "This is an insult" if label == 'insult' else "This is not an insult"
#         return jsonify(result=result)
