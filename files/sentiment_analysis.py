# http://127.0.0.1:5000/
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

from flask import Flask, request, jsonify, render_template_string

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Load and preprocess data
data = [
    ("This product is great!", "positive"),
    ("I don't like this product.", "negative"),
    ("Amazing quality and good service!", "positive"),
    ("Worst purchase ever.", "negative"),
    ("The item was damaged upon arrival.", "negative"),
    ("Excellent customer support!", "positive"),
    ("The product broke after a week.", "negative"),
    ("I'm very satisfied with my purchase.", "positive"),
    ("Terrible experience, would not recommend.", "negative"),
    ("This exceeded my expectations!", "positive")
]

df = pd.DataFrame(data, columns=['review', 'sentiment'])
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Convert labels to numeric
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment_encoded'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test_tfidf)
print("Model Evaluation:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Flask app
app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        textarea { width: 100%; height: 100px; margin-bottom: 10px; }
        button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        #result { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Sentiment Analysis</h1>
    <textarea id="review" placeholder="Enter your review here..."></textarea>
    <button onclick="analyzeSentiment()">Analyze Sentiment</button>
    <div id="result"></div>

    <script>
        function analyzeSentiment() {
            const review = document.getElementById('review').value;
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({review: review}),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerText = `Sentiment: ${data.sentiment}`;
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.json['review']
    cleaned_review = preprocess_text(review)
    tfidf_review = tfidf_vectorizer.transform([cleaned_review])
    prediction = rf_model.predict(tfidf_review)[0]
    sentiment = label_encoder.inverse_transform([prediction])[0]
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)