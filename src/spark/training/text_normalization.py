import re

from nltk.stem import WordNetLemmatizer
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


_LEMMATIZER = WordNetLemmatizer()
_NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")
_WORDNET_AVAILABLE = True


def _lemmatize_token(token):
    global _WORDNET_AVAILABLE

    if not _WORDNET_AVAILABLE:
        return token

    try:
        return _LEMMATIZER.lemmatize(token)
    except LookupError:
        _WORDNET_AVAILABLE = False
        return token


def lemmatize_review_text(text):
    if not isinstance(text, str):
        return ""

    normalized = _NON_LETTER_PATTERN.sub(" ", text.lower())
    tokens = [
        _lemmatize_token(token)
        for token in normalized.split()
    ]

    return " ".join(tokens)


lemmatize_review_text_udf = F.udf(
    lemmatize_review_text,
    StringType(),
)


def add_lemmatized_text_column(df, source_col="text", output_col="text"):
    return df.withColumn(
        output_col,
        lemmatize_review_text_udf(F.col(source_col)),
    )
