import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('balanced_data5.csv')

# Preprocess the dataset (assuming you have a preprocessing function)
df['cleaned_text'] = df.apply(lambda row: clean_text(row['text'], row['language']), axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

# Transform the text data using the saved TF-IDF vectorizer
X_train_tfidf = loaded_vectorizer.transform(X_train)
X_test_tfidf = loaded_vectorizer.transform(X_test)

# Train the model on the training data
loaded_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = loaded_model.predict(X_test_tfidf)
y_prob = loaded_model.predict_proba(X_test_tfidf)

# True labels and predictions
y_true = y_test
