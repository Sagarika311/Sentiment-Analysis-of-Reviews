# preprocess.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download necessary NLTK data (quietly)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
_stopwords = set(stopwords.words('english'))

def tokenize(text):
    """
    Tokenizer used by TfidfVectorizer.
    - lowercases, removes URLs, non-alpha chars
    - tokenizes with nltk
    - lemmatizes and removes stopwords
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)           # remove URLs
    text = re.sub(r'[^a-z\s]', ' ', text)                  # keep letters & spaces
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in _stopwords and len(t) > 1]
    return tokens

""" 
def tokenize(text):
    # Lowercase and keep only words
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # Remove stopwords + lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens
"""