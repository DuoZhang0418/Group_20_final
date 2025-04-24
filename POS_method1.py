###############################################################################
# 0. Imports (plus NLTK resources used for POS tagging)
###############################################################################
import os, re, numpy as np
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS 

# --- NEW: make sure both taggers are present -------------------------------
for pkg in ["punkt",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",   # <---- added
            "maxent_ne_chunker",
            "words"]:
    try:
        nltk.data.find(f"{'taggers' if 'tagger' in pkg else 'tokenizers'}/{pkg}")
    except LookupError:
        nltk.download(pkg)
# ---------------------------------------------------------------------------

from nltk import word_tokenize, pos_tag

from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.pipeline         import Pipeline, FeatureUnion, make_pipeline
from sklearn.datasets         import fetch_20newsgroups
from sklearn.model_selection  import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition    import TruncatedSVD
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import accuracy_score, classification_report

from gensim.models.doc2vec    import Doc2Vec, TaggedDocument
from gensim.utils             import simple_preprocess


###############################################################################
# 1. POS-aware tokenizer  (Method 1)
###############################################################################
def pos_tokenize(text: str):
    """Tokenizes, filters stop words, then appends POS tags."""
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens, lang="eng")
    # Filter out stop words BEFORE adding POS tags
    return [f"{tok}_{tag}" for tok, tag in pos_tags if tok not in ENGLISH_STOP_WORDS]


###############################################################################
# 2. Load GloVe vectors
###############################################################################
def load_glove_embedding(file_path):
    glove_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            if len(values) < 2:
                continue
            glove_dict[values[0]] = np.asarray(values[1:], dtype="float32")
    return glove_dict


###############################################################################
# 3. GloVeVectorizer (POS-aware, still looks up plain tokens)
###############################################################################
class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, glove_dict):
        self.glove_dict = glove_dict
        self.dim = len(next(iter(glove_dict.values()))) if glove_dict else 300

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        docs = []
        for doc in X:
            tokens = pos_tokenize(doc)
            vecs   = [self.glove_dict[t.split("_")[0]]
                      for t in tokens if t.split("_")[0] in self.glove_dict]
            docs.append(np.mean(vecs, axis=0) if vecs else np.zeros(self.dim))
        return np.vstack(docs)


###############################################################################
# 4. Doc2VecTransformer (trains on POS-tagged tokens)
###############################################################################
class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=20, dm=1):
        self.vector_size = vector_size
        self.window      = window
        self.min_count   = min_count
        self.epochs      = epochs
        self.dm          = dm

    def fit(self, X, y=None):
        corpus = [TaggedDocument(pos_tokenize(doc), [i]) for i, doc in enumerate(X)]
        self.model = Doc2Vec(vector_size=self.vector_size,
                             window=self.window,
                             min_count=self.min_count,
                             dm=self.dm,
                             epochs=self.epochs,
                             workers=4)
        self.model.build_vocab(corpus)
        self.model.train(corpus, total_examples=self.model.corpus_count, epochs=self.epochs)
        return self

    def transform(self, X):
        return np.vstack([self.model.infer_vector(pos_tokenize(doc), epochs=self.epochs)
                          for doc in X])


###############################################################################
# 5. Load 20 Newsgroups & split
###############################################################################
train_data = fetch_20newsgroups(subset="train")
test_data  = fetch_20newsgroups(subset="test")

X_train_full, X_dev, y_train_full, y_dev = train_test_split(
    train_data.data, train_data.target, test_size=0.20, random_state=42
)


###############################################################################
# 6. TF-IDF baseline (POS tokenizer)
###############################################################################
tfidf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=pos_tokenize,
        token_pattern=None,
        stop_words=None,  # Stop words already handled by tokenizer  # <-- Changed
    )),
    ("clf", LogisticRegression(max_iter=1000, C=1))
])

param_grid_tfidf = {
    "tfidf__max_features": [10_000, 20_000],
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
}

grid_search_tfidf = GridSearchCV(tfidf_pipeline,
                                 param_grid_tfidf,
                                 cv=3,
                                 scoring="accuracy",
                                 n_jobs=-1)
grid_search_tfidf.fit(X_train_full, y_train_full)
best_tfidf_model = grid_search_tfidf.best_estimator_

print("Best TF-IDF params:", grid_search_tfidf.best_params_)
print("[TF-IDF] Dev:",  accuracy_score(y_dev, best_tfidf_model.predict(X_dev)))
print("[TF-IDF] Test:", accuracy_score(test_data.target,
                                       best_tfidf_model.predict(test_data.data)))


###############################################################################
# 7. GloVe baseline
###############################################################################
glove_dict     = load_glove_embedding("glove.6B.300d.txt")
glove_pipeline = make_pipeline(GloveVectorizer(glove_dict),
                               LogisticRegression(max_iter=1000, C=1))
glove_pipeline.fit(X_train_full, y_train_full)

print("\n[GloVe] Dev:",  accuracy_score(y_dev, glove_pipeline.predict(X_dev)))
print("[GloVe] Test:", accuracy_score(test_data.target,
                                      glove_pipeline.predict(test_data.data)))


###############################################################################
# 8. Doc2Vec baseline
###############################################################################
doc2vec_pipeline = Pipeline([
    ("doc2vec", Doc2VecTransformer()),
    ("clf", LogisticRegression(max_iter=1000, C=1)),
])

param_grid_doc2vec = {
    "doc2vec__vector_size": [300, 600],
    "doc2vec__window":      [5, 8],
    "doc2vec__dm":          [0, 1],
}

grid_search_doc2vec = GridSearchCV(doc2vec_pipeline,
                                   param_grid_doc2vec,
                                   cv=3,
                                   scoring="accuracy",
                                   n_jobs=-1)
grid_search_doc2vec.fit(X_train_full, y_train_full)
best_doc2vec_model = grid_search_doc2vec.best_estimator_

print("\nBest Doc2Vec params:", grid_search_doc2vec.best_params_)
print("[Doc2Vec] Dev:",  accuracy_score(y_dev, best_doc2vec_model.predict(X_dev)))
print("[Doc2Vec] Test:", accuracy_score(test_data.target,
                                        best_doc2vec_model.predict(test_data.data)))


###############################################################################
# 9. Hybrid (TF-IDF + Doc2Vec + GloVe)
###############################################################################
best_tfidf_params   = grid_search_tfidf.best_params_
best_doc2vec_params = grid_search_doc2vec.best_params_

tfidf_block = Pipeline([
    ("tfidf", TfidfVectorizer(
        tokenizer=pos_tokenize,
        token_pattern=None,
        stop_words=None,  # Stop words already handled by tokenizer  # <-- Changed
        max_features=best_tfidf_params["tfidf__max_features"],
        ngram_range=best_tfidf_params["tfidf__ngram_range"]
    )),
    ("svd",    TruncatedSVD(n_components=3000, random_state=42)),
    ("scaler", StandardScaler())
])

doc2vec_block = Pipeline([
    ("doc2vec", Doc2VecTransformer(vector_size=300,
                                   window=best_doc2vec_params["doc2vec__window"],
                                   dm=best_doc2vec_params["doc2vec__dm"],
                                   epochs=20,
                                   min_count=2)),
    ("scaler",  StandardScaler())
])

glove_block = Pipeline([
    ("glove",  GloveVectorizer(glove_dict)),
    ("scaler", StandardScaler())
])

hybrid_pipeline = Pipeline([
    ("features", FeatureUnion([
        ("tfidf300",   tfidf_block),
        ("doc2vec300", doc2vec_block),
        ("glove300",   glove_block)
    ])),
    ("clf", LogisticRegression(max_iter=1000, C=1, n_jobs=-1))
])

hybrid_pipeline.fit(X_train_full, y_train_full)

print("\n[Hybrid] Dev:",
      accuracy_score(y_dev, hybrid_pipeline.predict(X_dev)))
print("[Hybrid] Test:",
      accuracy_score(test_data.target,
                     hybrid_pipeline.predict(test_data.data)))
