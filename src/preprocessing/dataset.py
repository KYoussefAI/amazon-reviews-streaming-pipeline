from src.preprocessing.clean import clean_text
from src.preprocessing.vectorizer import fit_transform
from src.preprocessing.label import get_label


def build_dataset(texts, scores):
    # clean
    cleaned = [clean_text(t) for t in texts]

    # vectorize
    X = fit_transform(cleaned)

    # labels
    y = [get_label(s) for s in scores]

    return X, y
