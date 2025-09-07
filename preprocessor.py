# preprocessor.py
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def transform(self, texts):
        cleaned = []
        for text in texts:
            text = text.lower()
            text = text.translate(str.maketrans("", "", string.punctuation))
            words = text.split()
            words = [w for w in words if w not in self.stop_words]
            words = [self.lemmatizer.lemmatize(w) for w in words]
            cleaned.append(" ".join(words))
        return cleaned
