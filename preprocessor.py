"""Text preprocessing utilities for fake news detection."""

from __future__ import annotations

import re
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible text preprocessor.

    Steps:
    1. lowercase
    2. remove non alphabetic characters
    3. optional stopword removal
    4. optional lemmatization
    """

    def __init__(self, remove_stopwords: bool = True, do_lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.do_lemmatize = do_lemmatize
        self._stopwords = set()
        self._lemmatizer = None
        self._only_letters = re.compile(r"[^a-zA-Z\s]+")

    @staticmethod
    def ensure_nltk_data() -> None:
        """Ensure required NLTK corpora are available."""
        resources = {
            "corpora/stopwords": "stopwords",
            "corpora/wordnet": "wordnet",
            "corpora/omw-1.4": "omw-1.4",
        }
        for path, package in resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(package, quiet=True)

    def fit(self, X: Iterable[str], y=None):
        self.ensure_nltk_data()
        if self.remove_stopwords:
            self._stopwords = set(stopwords.words("english"))
        if self.do_lemmatize:
            self._lemmatizer = WordNetLemmatizer()
        return self

    def _process_one(self, text: str) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        text = text.lower()
        text = self._only_letters.sub(" ", text)
        tokens = [tok for tok in text.split() if tok]
        if self.remove_stopwords:
            tokens = [tok for tok in tokens if tok not in self._stopwords]
        if self._lemmatizer is not None:
            tokens = [self._lemmatizer.lemmatize(tok) for tok in tokens]
        return " ".join(tokens)

    def transform(self, X: Iterable[str]) -> List[str]:
        return [self._process_one(x) for x in X]
