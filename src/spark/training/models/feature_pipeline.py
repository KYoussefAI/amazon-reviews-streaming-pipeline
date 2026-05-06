from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    StringIndexer,
    NGram,
    VectorAssembler,
)


def build_text_feature_stages(
    vocab_size=10000,
    min_df=4,
    use_bigrams=True,
):
    """
    Build reusable Spark ML text feature stages.

    Input column:
        text

    Output column:
        features

    Pattern:
        text
        -> RegexTokenizer
        -> StopWordsRemover
        -> CountVectorizer
        -> IDF
        -> features

    If use_bigrams=True:
        unigrams TF-IDF + bigrams TF-IDF are assembled into one features vector.
    """

    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="words",
        pattern="\\W+",
        toLowercase=True,
    )

    remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words",
    )

    if not use_bigrams:
        count_vectorizer = CountVectorizer(
            inputCol="filtered_words",
            outputCol="raw_features",
            vocabSize=vocab_size,
            minDF=min_df,
        )

        idf = IDF(
            inputCol="raw_features",
            outputCol="features",
        )

        return [
            tokenizer,
            remover,
            count_vectorizer,
            idf,
        ]

    ngram = NGram(
        n=2,
        inputCol="filtered_words",
        outputCol="bigrams",
    )

    unigram_vectorizer = CountVectorizer(
        inputCol="filtered_words",
        outputCol="unigram_raw_features",
        vocabSize=vocab_size,
        minDF=min_df,
    )

    unigram_idf = IDF(
        inputCol="unigram_raw_features",
        outputCol="unigram_features",
    )

    bigram_vectorizer = CountVectorizer(
        inputCol="bigrams",
        outputCol="bigram_raw_features",
        vocabSize=vocab_size,
        minDF=min_df,
    )

    bigram_idf = IDF(
        inputCol="bigram_raw_features",
        outputCol="bigram_features",
    )

    assembler = VectorAssembler(
        inputCols=[
            "unigram_features",
            "bigram_features",
        ],
        outputCol="features",
    )

    return [
        tokenizer,
        remover,
        ngram,
        unigram_vectorizer,
        unigram_idf,
        bigram_vectorizer,
        bigram_idf,
        assembler,
    ]


def build_label_indexer():
    return StringIndexer(
        inputCol="label",
        outputCol="label_index",
    )
