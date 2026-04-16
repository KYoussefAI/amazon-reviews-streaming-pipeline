import pandas as pd
from src.preprocessing.dataset import build_dataset

df = pd.read_csv("data/raw/Reviews.csv")

texts = df["Text"].tolist()
scores = df["Score"].tolist()

X, y = build_dataset(texts, scores)

print(X.shape)
print(len(y))
