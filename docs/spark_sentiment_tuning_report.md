# Spark Sentiment Pipeline Tuning Report

## 1. Purpose of This Document

This document summarizes the complete tuning journey for the Spark sentiment analysis model used in the Big Data project.

The goal was not only to improve metrics, but also to understand each decision step by step and keep the final model suitable for the next phase:

```text
Phase 8:
Kafka → Spark Structured Streaming → Load saved model → Predict sentiment
```

The final streaming phase must **not train** the model. It must only load the saved Spark `PipelineModel` and apply it to new review text.

---

## 2. Dataset and Labeling Rule

Dataset used:

```text
Amazon Fine Food Reviews
```

Main columns used:

```text
Text
Score
```

Sentiment labeling rule:

```text
Score < 3  → negative
Score = 3  → neutral
Score > 3  → positive
```

The dataset is strongly imbalanced:

```text
positive = dominant class
negative = minority class
neutral  = smallest and most ambiguous class
```

For the tuning phase, we used:

```python
SAMPLE_SIZE = 100000
```

The split strategy was:

```text
Train      = 80%
Validation = 10%
Test       = 10%
```

The model was trained only on the training set. Validation was used for tuning. Test was used once at the end for final evaluation.

---

## 3. Final Data Split Used During Tuning

For the 100k sample, the observed class distribution was:

```text
Full data:
positive = 76800
negative = 15180
neutral  = 8020
```

Split sizes:

```text
Train rows      = 79901
Validation rows = 10037
Test rows       = 10062
```

Training distribution:

```text
positive = 61252
negative = 12203
neutral  = 6446
```

Validation distribution:

```text
positive = 7785
negative = 1437
neutral  = 815
```

Test distribution:

```text
positive = 7763
negative = 1540
neutral  = 759
```

---

## 4. Metrics Used

Accuracy alone was not enough because the dataset is imbalanced.

So we tracked:

```text
Accuracy
Macro F1
Positive F1
Negative F1
Neutral F1
```

The most important metric for tuning was:

```text
Macro F1
```

Reason:

```text
Macro F1 = average F1 across positive, negative, and neutral
```

It prevents the model from looking good only because it predicts the dominant positive class well.

Neutral F1 was especially important because neutral was the weakest and most difficult class.

---

## 5. First Spark-Only Production Pipeline

The first production-oriented Spark pipeline used only Spark ML components.

Pipeline:

```text
text
→ RegexTokenizer
→ StopWordsRemover
→ CountVectorizer
→ IDF
→ StringIndexer
→ LogisticRegression
```

Main idea:

```text
Raw review text
→ tokens
→ filtered words
→ word count vector
→ TF-IDF vector
→ Logistic Regression prediction
```

This replaced the earlier hybrid experiment pipeline:

```text
Python/sklearn TF-IDF
→ manual conversion to Spark SparseVector
→ Spark Logistic Regression
```

The Spark-only version is better for production and streaming because the saved `PipelineModel` can later be loaded directly inside Spark Structured Streaming.

---

## 6. Important Learning: Fit vs Transform

A key concept clarified during tuning was the difference between `fit()` and `transform()`.

### CountVectorizer

```python
cv_model = count_vectorizer.fit(cleaned_df)
vectorized_df = cv_model.transform(cleaned_df)
```

Meaning:

```text
fit()
→ scans the training rows
→ learns vocabulary
→ stores word → index mapping

transform()
→ applies the learned vocabulary to each row
→ creates raw_features sparse vectors
```

### IDF

```python
idf_model = idf.fit(vectorized_df)
final_df = idf_model.transform(vectorized_df)
```

Meaning:

```text
fit()
→ learns global IDF weights from training data

transform()
→ multiplies raw word counts by IDF weights
→ creates final TF-IDF features
```

Important:

```text
TF-IDF is fitted only on training data.
Validation and test are transformed using the training vocabulary and IDF values.
```

This avoids data leakage.

---

## 7. Logistic Regression Understanding

Logistic Regression receives the final TF-IDF feature vector.

For each class, it learns one set of weights:

```text
positive class weights
negative class weights
neutral class weights
```

For one review, it computes a raw score for each class:

```text
score_positive = TFIDF(word1) × weight_positive(word1) + ...
score_negative = TFIDF(word1) × weight_negative(word1) + ...
score_neutral  = TFIDF(word1) × weight_neutral(word1)  + ...
```

Then Spark applies softmax:

```text
raw scores → probabilities
```

The class with the highest probability becomes the prediction.

Important clarification:

```text
The probability column is not accuracy.
It is the model confidence distribution for one row.
```

Example:

```text
probability = [0.90, 0.08, 0.02]
```

means:

```text
positive probability = 0.90
negative probability = 0.08
neutral probability  = 0.02
```

assuming the mapping is:

```text
0.0 → positive
1.0 → negative
2.0 → neutral
```

---

## 8. First Imbalance Solution: Class Weights

The dataset was imbalanced, so class weights were added.

Formula used:

```text
class_weight = total_training_rows / (number_of_classes × class_count)
```

Observed class weights:

```text
positive -> 0.4348
negative -> 2.1826
neutral  -> 4.1318
```

Meaning:

```text
positive errors are penalized less
negative errors are penalized more
neutral errors are penalized most
```

This helped the model pay more attention to minority classes.

---

## 9. Manual Hyperparameter Tuning

Before using Simulated Annealing, several manual tests were performed.

Main parameters tested:

```text
vocabSize
minDF
maxIter
regParam
```

### 9.1 vocabSize

`vocabSize` controls how many vocabulary terms are kept.

Example:

```python
vocabSize=5000
```

means Spark keeps up to 5000 selected terms.

Spark chooses vocabulary based on document frequency after `fit()`.

Larger vocabulary can capture more words, but it can also add noise.

Manual tests showed that increasing vocabulary helped at first, but very large vocabulary did not always improve minority-class performance.

### 9.2 minDF

`minDF` means minimum document frequency.

Example:

```python
minDF=2
```

means:

```text
Keep only words that appear in at least 2 review documents.
```

A document here means:

```text
one row = one review
```

Increasing `minDF` removes rare words and noise.

Later tuning showed that a higher `minDF` became useful when using unigrams + bigrams.

### 9.3 maxIter

`maxIter` is the maximum number of optimization rounds for Logistic Regression.

Important clarification:

```text
Spark LR does not update weights after each row.
```

Instead, one iteration is closer to:

```text
scan training partitions
compute global loss/gradient information
update model weights globally
```

More `maxIter` gives the optimizer more chances to converge, but it does not always improve validation metrics.

### 9.4 regParam

`regParam` controls regularization.

Regularization helps reduce overfitting by discouraging overly large weights.

We tested small values such as:

```text
0.0
0.0001
0.00005
0.000025
```

Very light regularization performed best.

---

## 10. Simulated Annealing Tuning

After manual tuning, we created a Simulated Annealing tuner.

File:

```text
src/spark/training/tune_spark_pipeline_sa.py
```

The tuning script reused the training code instead of duplicating the whole training script.

Reusable functions included:

```text
create_spark_session()
load_raw_data()
add_label_column()
split_dataset()
add_class_weights()
build_pipeline()
extract_metrics()
```

This improved architecture because the tuner searched configurations while the main training script remained the source of truth for the pipeline.

---

## 11. Why Simulated Annealing Was Used

Simulated Annealing was selected first because:

```text
It is simpler than Genetic Algorithms.
It can escape local optima.
It is efficient for a small search space.
It is easy to explain and implement.
```

---

## 12. How the Simulated Annealing Tuner Worked

Each configuration contained:

```python
{
    "vocab_size": ...,
    "min_df": ...,
    "max_iter": ...,
    "reg_param": ...,
    "use_bigrams": ...
}
```

The tuner:

```text
1. Starts from an initial configuration.
2. Creates a neighboring configuration by changing one parameter.
3. Trains a full Spark pipeline on train data.
4. Evaluates on validation data.
5. Accepts the candidate if it is better.
6. Sometimes accepts worse candidates depending on temperature.
7. Saves every result to CSV and Markdown.
```

Acceptance rule:

```text
If candidate_score >= current_score:
    accept

Else:
    accept with probability exp((candidate_score - current_score) / temperature)
```

The score optimized was:

```text
macro_f1
```

because the dataset is imbalanced.

---

## 13. First SA Search Without Bigrams

The first SA tuner searched around the best manual unigram pipeline.

Search included:

```text
vocabSize
minDF
maxIter
regParam
```

Best no-bigram result found:

```python
vocab_size = 20000
min_df = 2
max_iter = 30
reg_param = 0.0001
use_bigrams = False
```

Validation metrics:

```text
Accuracy    = 0.7786
Macro F1    = 0.6016
Positive F1 = 0.8770
Negative F1 = 0.5882
Neutral F1  = 0.3398
```

This slightly improved over manual tuning, but neutral remained weak.

---

## 14. Next Improvement Idea: Bigrams

The next idea was to add bigrams.

Why?

Unigrams alone see isolated words:

```text
good
bad
great
worth
```

But sentiment often depends on word combinations:

```text
not good
not worth
waste money
very good
would buy
```

So bigrams can capture phrase meaning.

---

## 15. Failed Attempt: Bigrams Only

The first bigram implementation replaced unigrams with bigrams.

Pipeline became:

```text
filtered_words
→ NGram
→ bigrams
→ CountVectorizer
→ IDF
→ LogisticRegression
```

This used only two-word phrases and lost single-word signals.

Result:

```text
Accuracy    = 0.7638
Macro F1    = 0.5706
Positive F1 = 0.8700
Negative F1 = 0.5388
Neutral F1  = 0.3029
```

This was worse than the unigram baseline.

Reason:

```text
The model lost useful single-word information such as:
great, bad, terrible, average, disappointed, delicious.
```

Conclusion:

```text
Bigrams alone are not good.
```

---

## 16. Better Solution: Unigrams + Bigrams

We then changed the approach.

Instead of replacing unigrams with bigrams, we combined both.

Final feature strategy:

```text
filtered_words → unigram TF-IDF
filtered_words → NGram → bigram TF-IDF
unigram_features + bigram_features → final features
```

Conceptually:

```text
review = "not good product"

unigrams:
not
good
product

bigrams:
not good
good product
```

This allowed the model to keep both:

```text
good
```

and:

```text
not good
```

This was the first big improvement.

---

## 17. Unigrams + Bigrams First Result

Using:

```python
vocab_size = 20000
min_df = 2
max_iter = 30
reg_param = 0.0001
use_bigrams = True
```

Validation result:

```text
Accuracy    = 0.8156
Macro F1    = 0.6301
Positive F1 = 0.9053
Negative F1 = 0.6207
Neutral F1  = 0.3644
```

This clearly improved over:

```text
unigrams only
bigrams only
```

Conclusion:

```text
Unigrams + bigrams should stay.
```

---

## 18. SA Search With Bigrams Enabled

The SA tuner was updated to include:

```python
use_bigrams
```

The search space was kept small for efficiency because unigrams + bigrams creates a larger feature vector.

First bigram-aware SA run found:

```python
vocab_size = 15000
min_df = 3
max_iter = 30
reg_param = 0.0001
use_bigrams = True
```

Validation result:

```text
Accuracy    = 0.8189
Macro F1    = 0.6369
Positive F1 = 0.9071
Negative F1 = 0.6387
Neutral F1  = 0.3650
```

Why it helped:

```text
vocabSize decreased from 20000 to 15000
minDF increased from 2 to 3
```

This removed more rare/noisy terms while keeping useful unigrams and bigrams.

---

## 19. Local Refinement Around the Bigram Best

The next local refinement searched around:

```python
vocab_size = 15000
min_df = 3
max_iter = 30
reg_param = 0.0001
use_bigrams = True
```

Search space:

```python
VOCAB_OPTIONS = [12000, 15000, 18000]
MIN_DF_OPTIONS = [3, 4]
MAX_ITER_OPTIONS = [20, 30]
REG_PARAM_OPTIONS = [0.00005, 0.0001, 0.0002]
BIGRAM_OPTIONS = [True]
```

Best result found:

```python
vocab_size = 12000
min_df = 3
max_iter = 20
reg_param = 0.00005
use_bigrams = True
```

Validation result:

```text
Accuracy    = 0.8163
Macro F1    = 0.6404
Positive F1 = 0.9034
Negative F1 = 0.6357
Neutral F1  = 0.3822
```

This improved Macro F1 mainly by improving neutral F1.

Interpretation:

```text
Smaller vocabulary helped.
Fewer iterations helped.
The model became less noisy and more balanced.
```

---

## 20. Final Micro-Refinement

A final smaller search was performed around the previous best.

Search space:

```python
VOCAB_OPTIONS = [10000, 12000, 14000]
MIN_DF_OPTIONS = [3, 4]
MAX_ITER_OPTIONS = [15, 20, 25]
REG_PARAM_OPTIONS = [0.0, 0.000025, 0.00005, 0.0001]
BIGRAM_OPTIONS = [True]
```

Best result found:

```python
vocab_size = 10000
min_df = 4
max_iter = 15
reg_param = 0.000025
use_bigrams = True
```

Validation result:

```text
Accuracy    = 0.8159
Macro F1    = 0.6470
Positive F1 = 0.9031
Negative F1 = 0.6419
Neutral F1  = 0.3961
```

This became the final selected validation configuration.

---

## 21. Final Validation Comparison Table

| Stage | vocabSize | minDF | maxIter | regParam | Bigrams | Accuracy | Macro F1 | Positive F1 | Negative F1 | Neutral F1 |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|
| Best unigram SA | 20000 | 2 | 30 | 0.0001 | False | 0.7786 | 0.6016 | 0.8770 | 0.5882 | 0.3398 |
| Bigrams only | 20000 | 2 | 30 | 0.0001 | Only bigrams | 0.7638 | 0.5706 | 0.8700 | 0.5388 | 0.3029 |
| Unigrams + bigrams first | 20000 | 2 | 30 | 0.0001 | True | 0.8156 | 0.6301 | 0.9053 | 0.6207 | 0.3644 |
| Bigram SA best | 15000 | 3 | 30 | 0.0001 | True | 0.8189 | 0.6369 | 0.9071 | 0.6387 | 0.3650 |
| Local refinement best | 12000 | 3 | 20 | 0.00005 | True | 0.8163 | 0.6404 | 0.9034 | 0.6357 | 0.3822 |
| Final micro-refinement best | 10000 | 4 | 15 | 0.000025 | True | 0.8159 | 0.6470 | 0.9031 | 0.6419 | 0.3961 |

---

## 22. Final Selected Configuration

Final selected production configuration:

```python
pipeline = build_pipeline(
    vocab_size=10000,
    min_df=4,
    max_iter=15,
    reg_param=0.000025,
    use_bigrams=True
)
```

Why this was selected:

```text
Best validation Macro F1
Best validation Neutral F1
Strong Positive F1
Strong Negative F1
Smaller and more efficient feature space
Less overfitting risk than larger vocabularies
```

Approximate feature size:

```text
10000 unigram features + 10000 bigram features = about 20000 total features
```

This is more efficient than earlier bigram versions:

```text
20000 + 20000 = about 40000 features
15000 + 15000 = about 30000 features
```

---

## 23. Final Training and Saved Model

After selecting the final parameters, the model was trained and saved as a Spark `PipelineModel`.

Model output path:

```text
src/spark/model/sentiment_pipeline_model
```

Training script command:

```bash
spark-submit src/spark/training/train_spark_pipeline.py 2>/dev/null | grep -E "==========|Accuracy|Macro F1|Positive F1|Negative F1|Neutral F1|Model saved"
```

Output confirmed:

```text
Model saved to: src/spark/model/sentiment_pipeline_model
```

Important Git decision:

```text
Do not commit the saved model to GitHub.
```

Reason:

```text
It is a generated artifact.
It can be large.
It can be recreated by running the training script.
```

Recommended `.gitignore`:

```gitignore
src/spark/model/
src/spark/models/
*.model
data/raw/Reviews.csv
data/raw/*.csv
```

---

## 24. Final Validation and Test Metrics

Final validation result:

```text
Accuracy    = 0.8159
Macro F1    = 0.6470
Positive F1 = 0.9031
Negative F1 = 0.6419
Neutral F1  = 0.3961
```

Final test result:

```text
Accuracy    = 0.8130
Macro F1    = 0.6349
Positive F1 = 0.9018
Negative F1 = 0.6558
Neutral F1  = 0.3470
```

Interpretation:

```text
Accuracy stayed close.
Positive F1 stayed close.
Negative F1 improved on test.
Neutral F1 dropped on test.
```

The model generalizes reasonably well, but neutral remains the most unstable class.

Important rule:

```text
Do not continue tuning based on test results.
```

The test set was used as a final unbiased evaluation.

---

## 25. What Failed and Why

### 25.1 Accuracy-only thinking

Problem:

```text
Accuracy looked good because positive is dominant.
```

Solution:

```text
Use Macro F1 and per-class F1.
```

### 25.2 Unigrams only

Problem:

```text
Could not capture phrase sentiment such as "not good".
```

Solution:

```text
Add bigram information.
```

### 25.3 Bigrams only

Problem:

```text
Removed important single-word signals.
```

Solution:

```text
Use unigrams + bigrams together.
```

### 25.4 Too large vocabulary

Problem:

```text
More features added noise and increased cost.
```

Solution:

```text
Reduce vocabSize and increase minDF.
```

### 25.5 Too many iterations

Problem:

```text
More maxIter did not improve generalization.
```

Solution:

```text
Reduce maxIter to 15.
```

### 25.6 Over-tuning risk

Problem:

```text
Too many validation experiments can indirectly overfit validation.
```

Solution:

```text
Stop after final validation tuning and evaluate test once.
```

---

## 26. Final Lessons Learned

Important technical lessons:

```text
1. Split before fitting transformations.
2. Fit vocabulary and IDF only on train data.
3. Use validation for tuning.
4. Keep test untouched until the end.
5. Accuracy is not enough for imbalanced datasets.
6. Macro F1 is better for selecting balanced models.
7. Class weights help minority classes.
8. Bigrams are useful only when combined with unigrams.
9. Smaller feature spaces can generalize better.
10. Simulated Annealing helps search efficiently.
11. The saved Spark PipelineModel is required for streaming inference.
```

---

## 27. Final Decision Before Streaming

The final model is ready for Phase 8.

Use:

```text
src/spark/model/sentiment_pipeline_model
```

Next phase:

```text
Kafka → Spark Structured Streaming → Load saved model → Predict sentiment → Console output
```

The streaming script must only perform inference:

```text
No training
No validation
No test evaluation
No model tuning
```

It should:

```text
1. Read review JSON messages from Kafka.
2. Parse text and score.
3. Add dummy label/class_weight columns if required by the saved pipeline.
4. Apply PipelineModel.transform().
5. Convert prediction index back to label.
6. Print predictions to console.
```

---

## 28. Final Commit Title Suggestion

Recommended commit title:

```text
Optimize Spark sentiment pipeline and save final trained model
```

Alternative:

```text
Add Spark ML tuning, bigram features, and saved sentiment pipeline
```

---

## 29. Current Best Summary

```text
Best validation configuration:
vocab_size  = 10000
min_df      = 4
max_iter    = 15
reg_param   = 0.000025
use_bigrams = True

Validation:
Accuracy    = 0.8159
Macro F1    = 0.6470
Positive F1 = 0.9031
Negative F1 = 0.6419
Neutral F1  = 0.3961

Test:
Accuracy    = 0.8130
Macro F1    = 0.6349
Positive F1 = 0.9018
Negative F1 = 0.6558
Neutral F1  = 0.3470
```

Final conclusion:

```text
The final model is a Spark-only TF-IDF + Logistic Regression pipeline using class weights and combined unigram + bigram features. The selected configuration gives the best validation Macro F1 while keeping the feature space efficient and is ready to be loaded by Spark Structured Streaming for real-time prediction.
```
