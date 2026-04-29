from src.experiments.preprocessing.clean import clean_text
from src.experiments.preprocessing.label import get_label
from sklearn.model_selection import train_test_split
import pandas as pd


def load_dataset():

    df = pd.read_csv("data/raw/Reviews.csv").head(10000)

    texts = df["Text"].tolist()
    scores = df["Score"].tolist()

    # clean
    cleaned = [clean_text(t) for t in texts]

    # labels
    labels = [get_label(s) for s in scores]

    x_train, x_temp, y_train, y_temp = train_test_split(
        cleaned, labels, test_size=0.2, random_state=42, stratify=labels)

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp,  test_size=0.5, stratify=y_temp)

    return x_train, x_val, x_test, y_train, y_val, y_test
