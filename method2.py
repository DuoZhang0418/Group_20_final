###############################################################################
# 0. Imports  +  NLTK resources
###############################################################################
import os, re, numpy as np, nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

for pkg in ["punkt",
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
            "maxent_ne_chunker",
            "words"]:
    subdir = "taggers" if "tagger" in pkg else "tokenizers"
    try:
        nltk.data.find(f"{subdir}/{pkg}")
    except LookupError:
        nltk.download(pkg)

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
# 1. Simple tokenizer   (we’ll reuse it for POS weighting)
###############################################################################
def pos_tokenize(text: str):
    tokens   = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens, lang="eng")
    # filter stop-words *before* attaching POS tags
    return [f"{tok}_{tag}" for tok, tag in pos_tags if tok not in ENGLISH_STOP_WORDS]


###############################################################################
# 2. Load GloVe vectors
###############################################################################
def load_glove_embedding(file_path):
    glove = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 2:               # skip blank / malformed lines
                continue
            glove[parts[0]] = np.asarray(parts[1:], dtype="float32")
    return glove


###############################################################################
# 3. Weighted-GloVe Vectorizer  (Method 2)
###############################################################################

POS_WEIGHTS = {
    "NN": 2.0, "NNS": 2.0,        # nouns
    "NNP": 2.0, "NNPS": 2.0,      # proper nouns
    "JJ": 1.5,                    # adjectives
    "VB": 1.0, "VBD": 1.0, "VBG": 1.0, "VBN": 1.0, "VBP": 1.0, "VBZ": 1.0,
    "RB": 0.8,                    # adverbs
    "_OTHER_": 0.5                # fallback
}

class WeightedGloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, glove_dict, pos_weights=POS_WEIGHTS):
        self.glove_dict  = glove_dict
        self.dim         = len(next(iter(glove_dict.values()))) if glove_dict else 300
        self.pos_weights = pos_weights

    # split token_TAG → (token, weight)
    def _tok_and_w(self, tagged_tok):
        if "_" not in tagged_tok:
            return tagged_tok, self.pos_weights["_OTHER_"]
        tok, tag = tagged_tok.rsplit("_", 1)
        w = self.pos_weights.get(tag, self.pos_weights.get(tag[:2], self.pos_weights["_OTHER_"]))
        return tok, w

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for doc in X:
            vec_sum = np.zeros(self.dim, dtype="float32")
            w_sum   = 0.0
            for tagged in pos_tokenize(doc):
                tok, w = self._tok_and_w(tagged)
                if tok in self.glove_dict:
                    vec_sum += w * self.glove_dict[tok]
                    w_sum   += w
            out.append(vec_sum / w_sum if w_sum else np.zeros(self.dim, dtype="float32"))
        return np.vstack(out)


###############################################################################
# 4. Doc2Vec transformer  (unchanged)
###############################################################################
class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=20, dm=1):
        self.vector_size = vector_size
        self.window      = window
        self.min_count   = min_count
        self.epochs      = epochs
        self.dm          = dm

    def fit(self, X, y=None):
        corpus = [TaggedDocument(simple_preprocess(doc), [i])
                  for i, doc in enumerate(X)]
        self.model = Doc2Vec(vector_size=self.vector_size,
                             window=self.window,
                             min_count=self.min_count,
                             dm=self.dm,
                             epochs=self.epochs,
                             workers=4)
        self.model.build_vocab(corpus)
        self.model.train(corpus,
                         total_examples=self.model.corpus_count,
                         epochs=self.epochs)
        return self

    def transform(self, X):
        return np.vstack([
            self.model.infer_vector(simple_preprocess(doc), epochs=self.epochs)
            for doc in X
        ])


###############################################################################
# 5. Load 20-Newsgroups & split
###############################################################################
train   = fetch_20newsgroups(subset="train")
test    = fetch_20newsgroups(subset="test")

X_train_full, X_dev, y_train_full, y_dev = train_test_split(
    train.data, train.target, test_size=0.20, random_state=42)


###############################################################################
# 6. TF-IDF pipeline + grid-search (unchanged)
###############################################################################
tfidf_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf",   LogisticRegression(max_iter=1000, C=1))
])
gs_tfidf = GridSearchCV(tfidf_pipe,
                        {"tfidf__max_features":[10_000, 20_000],
                         "tfidf__ngram_range":[(1,1),(1,2)]},
                        cv=3, scoring="accuracy", n_jobs=-1)
gs_tfidf.fit(X_train_full, y_train_full)
best_tfidf = gs_tfidf.best_estimator_

print("TF-IDF   dev:", accuracy_score(y_dev, best_tfidf.predict(X_dev)))
print("TF-IDF  test:", accuracy_score(test.target,
                                      best_tfidf.predict(test.data)))


###############################################################################
# 7.  Weighted-GloVe baseline  (Method 2)
###############################################################################
glove_dict = load_glove_embedding("glove.6B.300d.txt")
wglove_pipe = make_pipeline(
    WeightedGloveVectorizer(glove_dict),
    LogisticRegression(max_iter=1000, C=1)
)
wglove_pipe.fit(X_train_full, y_train_full)

print("\nW-GloVe dev:", accuracy_score(y_dev, wglove_pipe.predict(X_dev)))
print("W-GloVe test:", accuracy_score(test.target,
                                      wglove_pipe.predict(test.data)))


###############################################################################
# 8.  Doc2Vec baseline  + grid-search (unchanged)
###############################################################################
doc2vec_pipe = Pipeline([
    ("doc2vec", Doc2VecTransformer()),
    ("clf", LogisticRegression(max_iter=1000, C=1))
])
gs_doc2vec = GridSearchCV(doc2vec_pipe,
                          {"doc2vec__vector_size":[300,600],
                           "doc2vec__window":[5,8],
                           "doc2vec__dm":[0,1]},
                          cv=3, scoring="accuracy", n_jobs=-1)
gs_doc2vec.fit(X_train_full, y_train_full)
best_doc2vec = gs_doc2vec.best_estimator_

print("\nDoc2Vec dev:", accuracy_score(y_dev, best_doc2vec.predict(X_dev)))
print("Doc2Vec test:", accuracy_score(test.target,
                                      best_doc2vec.predict(test.data)))


###############################################################################
# 9.  Hybrid  (TF-IDF + Doc2Vec + Weighted-GloVe)
###############################################################################
tfidf_block = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=best_tfidf.named_steps["tfidf"].max_features,
        ngram_range=best_tfidf.named_steps["tfidf"].ngram_range)),
    ("svd",    TruncatedSVD(n_components=3000, random_state=42)),
    ("scaler", StandardScaler())
])

doc2vec_block = Pipeline([
    ("doc2vec", Doc2VecTransformer(
        vector_size = best_doc2vec.named_steps["doc2vec"].vector_size,
        window      = best_doc2vec.named_steps["doc2vec"].window,
        dm          = best_doc2vec.named_steps["doc2vec"].dm,
        epochs      = 20,
        min_count   = 2)),
    ("scaler", StandardScaler())
])

wglove_block = Pipeline([
    ("wglove", WeightedGloveVectorizer(glove_dict)),
    ("scaler", StandardScaler())
])

hybrid = Pipeline([
    ("features", FeatureUnion([
        ("tfidf3000", tfidf_block),
        ("doc2vec300", doc2vec_block),
        ("wglove300",  wglove_block)
    ])),
    ("clf", LogisticRegression(max_iter=1000, C=1, n_jobs=-1))
])

hybrid.fit(X_train_full, y_train_full)
print("\nHybrid dev:", accuracy_score(y_dev, hybrid.predict(X_dev)))
print("Hybrid test:", accuracy_score(test.target,
                                     hybrid.predict(test.data)))
