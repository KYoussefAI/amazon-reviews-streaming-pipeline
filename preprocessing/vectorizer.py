from preprocessing.clean import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer()  # TF * IDF , TF : occurence in the same doc / IDF : occurence across all docs


def fit_transform(texts):
    return vectorizer.fit_transform(texts)


def transform(texts):
    return vectorizer.transform(texts)


texts = [
    clean_text("This product is AMAZING"),
    clean_text("I love this product"),
]

X = fit_transform(texts)

print(X.toarray())

# Rare across documents → high importance (high IDF)
# Common across documents → low importance (low IDF)
