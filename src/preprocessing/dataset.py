from src.preprocessing.clean import clean_text
from src.preprocessing.vectorizer import fit_transform
from src.preprocessing.label import get_label
import pandas as pd


def load_dataset():

    df = pd.read_csv("data/raw/Reviews.csv").head(10000)

    texts = df["Text"].tolist()
    scores = df["Score"].tolist()

    # clean
    cleaned = [clean_text(t) for t in texts]

    # vectorize
    X = fit_transform(cleaned)  # to => scipy.sparse.csr_matrix

    # labels
    y = [get_label(s) for s in scores]

    return X, y
