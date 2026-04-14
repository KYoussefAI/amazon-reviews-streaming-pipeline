from sklearn.feature_extraction.text import TfidfVectorizer


vectorizer = TfidfVectorizer()  # TF * IDF , TF : occurence in the same doc / IDF : occurence across all docs


def fit_transform(texts):
    return vectorizer.fit_transform(texts)


def transform(texts):
    return vectorizer.transform(texts)


# idf(word) = log(N / df)
# Rare across documents → high importance (high IDF)
# Common across documents → low importance (low IDF)
# idf = log((1 + N) / (1 + df)) + 1 => normalization instead of v / ||v||
#
# fit
# builds vocabulary (word → index)
# computes IDF values (global importance per word)
#
# transform
# converts text → TF (counts / frequencies)
# applies TF × IDF
# applies normalization
