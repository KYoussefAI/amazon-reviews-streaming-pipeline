import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return " ".join(words)


sample = "This product is AMAZING!!! I love it 100%"
print(clean_text(sample))
