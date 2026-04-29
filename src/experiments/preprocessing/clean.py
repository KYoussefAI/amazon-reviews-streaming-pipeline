import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
STOPWORDS.add('im')
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)  # ^ -> not remove lower-case letter \ -> OR spaces , and replace with ''

    words = text.split()

    words = [w for w in words if w not in STOPWORDS]

    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)


# sample = "This product is AMAZING!!! I'm Happy !! "
# print(clean_text(sample))
