# NLP & Sentiment Analysis — TF-IDF + Naive Bayes/SVM

**Data Science Project 4 | DecodeLabs Industrial Training Kit | Batch 2026**

## Live Demo

**Try it live:** https://nlp-sentiment-analysis-tfidf-abc123.streamlit.app

Or run locally:
```
Run `streamlit run streamlit_app.py` to launch an interactive dashboard where you can type any review and see it classified as Positive or Negative in real time, using the saved `sentiment_analysis_best_model.joblib` pipeline.
```
## Overview

An NLP project that builds a full text pre-processing and classification pipeline to predict whether a review is Positive or Negative — converting unstructured human language into mathematical arrays via TF-IDF, then training and comparing a Naive Bayes and an SVM classifier.

**Source data:** NLTK's `movie_reviews` corpus (Pang & Lee polarity dataset) — 2,000 raw, unmodified reviews, perfectly balanced 1,000 positive / 1,000 negative. This is the standard benchmark dataset for exactly this kind of exercise, downloaded automatically via `nltk.download()` (no manual file needed).

No missing values or duplicate reviews — verified in the notebook before modeling.

## Approach

1. **Load & clean-check** — confirm the raw corpus has no nulls or duplicates, and the classes are balanced.
2. **Negation-aware stopwords** — default stopword lists remove words like "not", which flips sentiment meaning entirely ("not happy" → "happy" if removed). Negations are explicitly excluded from the stopword set.
3. **POS-guided lemmatization** — `WordNetLemmatizer` needs a part-of-speech tag to reduce words correctly (e.g. "went" → "go" only when tagged as a verb); Treebank POS tags are mapped to WordNet's tag format before lemmatizing.
4. **Text pre-processing pipeline** — lowercase → strip HTML/punctuation → tokenize → remove stopwords (keeping negations) → POS-tag → lemmatize.
5. **Split before vectorizing** — same leakage-safe principle as Project 2: the TF-IDF vocabulary and IDF weights are learned only from training data.
6. **TF-IDF vectorization** — unigrams + bigrams, with `max_features`/`min_df` bounds to control vocabulary size, stored automatically in SciPy CSR sparse format.
7. **Train & compare classifiers** — Multinomial Naive Bayes (with Laplace smoothing, `alpha=1.0`) and a Linear SVM, evaluated on the same held-out test set.
8. **Save the best pipeline** — vectorizer and classifier bundled together via `joblib`, so new text can be classified without re-fitting the vocabulary.

## Results

| Model | Accuracy |
|---|---|
| Multinomial Naive Bayes | 0.818 |
| **Linear SVM** | **0.835** |

**Best model: Linear SVM** — slightly higher accuracy and a better precision/recall balance across both classes than Naive Bayes on this dataset.

## Known Limitations

- **Informal/slang text is unreliable.** The training data (Pang & Lee polarity dataset) is formal, professionally written movie criticism from the early 2000s. Casual internet slang (e.g. "ew", "meh", "lit") rarely or never appears in it, so the model has little to no learned signal for these words and effectively ignores them.
- **Very short reviews are especially weak.** With `min_df=2` in the TF-IDF vectorizer, a word must appear in at least 2 training documents to get its own feature — rare/unseen words are dropped entirely. A short review (a handful of words) has little else for the model to go on if one of its few words gets dropped this way, making the prediction closer to a coin-flip than a confident classification.
- **Domain mismatch in general.** A model trained on movie reviews will carry some bias toward movie-review vocabulary and framing; performance on a genuinely different domain (e.g. product reviews, restaurant reviews) would likely be somewhat lower without retraining or fine-tuning on in-domain data.

## Repo Structure

```
nlp-sentiment-analysis-tfidf/
├── project4_nlp_sentiment_analysis.ipynb   # main notebook, cell by cell
├── movie_reviews.csv                       # raw dataset
├── sentiment_analysis_best_model.joblib    # saved best model (TF-IDF + SVM pipeline)
├── streamlit_app.py                        # Streamlit live demo
├── README.md                               # this file
├── requirements.txt
└── .gitignore
```

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Open `project4_nlp_sentiment_analysis.ipynb` and run all cells top to bottom. The notebook downloads the required NLTK corpora (`movie_reviews`, `punkt`, `stopwords`, `wordnet`, POS tagger) automatically on first run.

### Using the Saved Model

```python
import joblib

model = joblib.load("sentiment_analysis_best_model.joblib")
predictions = model.predict([cleaned_text])   # text must go through the same preprocess() function first
```

## Key Takeaways

1. **Default stopword lists can silently destroy meaning** — removing negations like "not" flips sentiment entirely; they must be explicitly excluded.
2. **Lemmatization without a POS tag is unreliable** — it defaults to treating every word as a noun, so verbs and adjectives don't reduce correctly.
3. **Splitting before vectorizing prevents leakage** — the same principle from Project 2's SMOTE pipeline applies here to TF-IDF's learned vocabulary.
4. **Naive Bayes and SVM aren't interchangeable by default** — comparing both instead of assuming one is "the" NLP classifier surfaced a real (if small) performance difference.

## Tools Used

`pandas`, `numpy`, `nltk` (tokenization, stopwords, WordNetLemmatizer, POS tagging), `scikit-learn` (TfidfVectorizer, MultinomialNB, LinearSVC, Pipeline), `matplotlib`, `seaborn`, `joblib`
