'''
TFIDF/GloVe/Doc2Vec individual baselines
Hybrid models (combining TFIDF/GloVe/Doc2Vec)
+ POS histogram and bigram features
+ Ner features

'''


import os, re, warnings, random, numpy as np, nltk, spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

for pkg in ["punkt", "averaged_perceptron_tagger"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt" else f"taggers/{pkg}")
    except LookupError:
        nltk.download(pkg)

# NER
try:
    _NLP = spacy.load("en_core_web_sm",disable=["tagger","parser","lemmatizer"])
except OSError:
    from spacy.cli import download; download("en_core_web_sm")
    _NLP = spacy.load("en_core_web_sm",disable=["tagger","parser","lemmatizer"])

NER_TYPES = ["PERSON","NORP","FAC","ORG","GPE","LOC",
             "PRODUCT","EVENT","DATE","TIME","CARDINAL","MONEY"]
NER2IDX = {lbl:i for i,lbl in enumerate(NER_TYPES)}

class NerHistogram(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None): return self
    def transform(self,X):
        out=np.zeros((len(X),len(NER_TYPES)),dtype=np.float32)
        for i, doc in enumerate(_NLP.pipe(X, batch_size=32)):
            for ent in doc.ents:
                j = NER2IDX.get(ent.label_)
                if j is not None:
                    out[i, j] += 1
        row_sums = out.sum(axis=1)
        nonzero = row_sums > 0
        out[nonzero] = out[nonzero] / row_sums[nonzero, np.newaxis]
        return out

ner_block = Pipeline([("hist",NerHistogram()),
                      ("scal",StandardScaler())])

# --------------------------------------------------------------------------- #
# 3.  Utility – GloVe loader                                                  #
# --------------------------------------------------------------------------- #

def load_glove_embedding(path: str = "glove.6B.300d.txt"):
    """Returns {word -> np.ndarray(dim)} for the requested GloVe file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GloVe file not found at {path}; download from https://nlp.stanford.edu/projects/glove/")
    glove = {}
    with open(path, "r", encoding="utf‑8") as f:
        for ln in f:
            w, *vec = ln.rstrip().split()
            glove[w] = np.asarray(vec, dtype="float32")
    return glove

class GloveVectorizer(BaseEstimator, TransformerMixin): 
    def __init__(self, glove_dict):
        self.glove = glove_dict
        self.dim = len(next(iter(glove_dict.values())))

    @staticmethod
    def _tokenize(s: str):
        return re.findall(r"\w+", s.lower())

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.vstack([
            np.mean([self.glove[w] for w in self._tokenize(doc) if w in self.glove], 0)
            if any(w in self.glove for w in self._tokenize(doc)) else np.zeros(self.dim)
            for doc in X
        ])

# Doc2Vec transformer

class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=300, window=5, min_count=2, epochs=20, dm=1, model_dir="doc2vec_models"):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self._model = None  

    def get_params(self, deep=True):
        return {
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "epochs": self.epochs,
            "dm": self.dm,
            "model_dir": self.model_dir,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _model_path(self):
        name = f"d2v_vs{self.vector_size}_win{self.window}_dm{self.dm}.model"
        return os.path.join(self.model_dir, name)

    def fit(self, X, y=None):
        path = self._model_path()
        if os.path.isfile(path):
            self._model = Doc2Vec.load(path)
        else:
            corpus = [TaggedDocument(simple_preprocess(d), [i]) for i, d in enumerate(X)]
            self._model = Doc2Vec(
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                dm=self.dm,
                epochs=self.epochs,
                workers=4,   
                seed=SEED,
            )
            self._model.build_vocab(corpus)
            self._model.train(corpus, total_examples=self._model.corpus_count, epochs=self.epochs)
            self._model.save(path)
        return self

    def transform(self, X):
        if self._model is None:
            raise RuntimeError("Doc2VecTransformer must be fitted before calling transform().")
        return np.vstack([
            self._model.infer_vector(simple_preprocess(d), epochs=self.epochs) for d in X
        ])


# POS‑histogram transformer 

#SELECTED_TAGS = ["NN", "NNP"]   # ["NN","NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ","JJ","JJR","JJS","RB","RBR","RBS"]

SELECTED_TAGS = [
    "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
    "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$",
    "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "WDT", "WP", "WP$", "WRB",
    "#", "$", "``", "''", ",", ".", ":", "-LRB-", "-RRB-"
]     

TAG2IDX = {t: i for i, t in enumerate(SELECTED_TAGS)}


def pos_histogram(doc: str):
    vec = np.zeros(len(SELECTED_TAGS), dtype=np.float32)
    for _, tag in nltk.pos_tag(nltk.word_tokenize(doc)):
        idx = TAG2IDX.get(tag)
        if idx is not None:
            vec[idx] += 1
    if vec.sum():
        vec /= vec.sum() 
    return vec

pos_hist_block = Pipeline([
    ("hist", FunctionTransformer(lambda docs: np.vstack([pos_histogram(d) for d in docs]), validate=False)),
    ("scal", StandardScaler()),
])

from nltk import word_tokenize, pos_tag
from nltk.util import ngrams
def pos_ngrams(doc, n=2):
    tags=[t for _,t in pos_tag(word_tokenize(doc))]
    return ["_".join(g) for g in ngrams(tags,n)]

pos_bigram = Pipeline([
    ("vect", TfidfVectorizer(
        analyzer="word",
        tokenizer=pos_ngrams,
        ngram_range=(1,1),       
        max_features=5000)),
    ("svd", TruncatedSVD(n_components=1500, random_state=SEED)),
    ("scal", StandardScaler(with_mean=False))
])


# 20-newsgroups Dataset
train = fetch_20newsgroups(subset="train")
_test = fetch_20newsgroups(subset="test")
X_train, X_dev, y_train, y_dev = train_test_split(train.data, train.target, test_size=0.20, random_state=SEED)

print("Newsgroup label → category name:")
for i, name in enumerate(train.target_names):
    print(f"  {i:2d} → {name}")


# TFIDF baseline with grid‑search
TFIDF_GRID = {
    "tfidf__max_features": [10_000, 20_000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
}

tfidf_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, C=1)),
])

grid = GridSearchCV(tfidf_pipe, TFIDF_GRID, cv=3, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
best_tfidf = grid.best_estimator_
best_tfidf_params = grid.best_params_

# --------------------------------------------------------------------------- #
# 8.  GloVe baseline                                                         #
# --------------------------------------------------------------------------- #
glove = load_glove_embedding("glove.6B.300d.txt")

glove_baseline = make_pipeline(
    GloveVectorizer(glove),
    LogisticRegression(max_iter=1000, C=1),
).fit(X_train, y_train)

# --------------------------------------------------------------------------- #
# 9.  Doc2Vec baseline (with grid‑search & caching)                          #
# --------------------------------------------------------------------------- #

D2V_GRID = {
    "doc2vec__vector_size": [300, 600],
    "doc2vec__window": [5, 8],
    "doc2vec__dm": [0, 1],
}

d2v_pipe = Pipeline([
    ("doc2vec", Doc2VecTransformer()),
    ("clf", LogisticRegression(max_iter=1000, C=1)),
])

grid2 = GridSearchCV(d2v_pipe, D2V_GRID, cv=3, scoring="accuracy", n_jobs=-1)
grid2.fit(X_train, y_train)
best_d2v = grid2.best_estimator_
best_d2v_params = grid2.best_params_

# --------------------------------------------------------------------------- #
# 10.  Feature blocks for hybrid model                                       #
# --------------------------------------------------------------------------- #

tfidf_words3000 = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=best_tfidf_params["tfidf__max_features"],
        ngram_range=best_tfidf_params["tfidf__ngram_range"],
    )),
    #("svd", TruncatedSVD(n_components=3000, random_state=SEED)),
    #("scal", StandardScaler()),
    ("scal", StandardScaler(with_mean=False)),
])

doc2vec300 = Pipeline([
    ("doc2vec", Doc2VecTransformer(
        vector_size=best_d2v_params["doc2vec__vector_size"],
        window=best_d2v_params["doc2vec__window"],
        dm=best_d2v_params["doc2vec__dm"],
        epochs=20,
        min_count=2,
    )),
    ("scal", StandardScaler()),
])

glove300 = Pipeline([
    ("glove", GloveVectorizer(glove)),
    ("scal", StandardScaler()),
])

# --------------------------------------------------------------------------- #
# 11.  HYBRID (TF‑IDF + POS‑hist + D2V + GloVe)                              #
# --------------------------------------------------------------------------- #

def seq_union(name_blocks):
    """FeatureUnion that always runs sequentially (n_jobs = 1)."""
    return FeatureUnion(name_blocks, n_jobs=1)


# ---------- build the blocks that never cause trouble -------------------
core_blocks = [
    ("tfidf", tfidf_words3000),
    ("glove", glove300),
    ("d2v",   doc2vec300),
]

hybrid_hist = Pipeline([
    ("features", FeatureUnion([
        ("tfidf_words3000", tfidf_words3000),
        ("pos_hist45", pos_hist_block),
        ("doc2vec300", doc2vec300),
        ("glove300", glove300),
    ], n_jobs=-1)),
    ("clf", LogisticRegression(max_iter=1000, C=1, n_jobs=-1)),
]).fit(X_train, y_train)


hybrid_noPOS = Pipeline([
    ("features", FeatureUnion([
        ("tfidf_words3000", tfidf_words3000),
        ("doc2vec300", doc2vec300),
        ("glove300", glove300),
    ], n_jobs=-1)),
    ("clf", LogisticRegression(max_iter=1000, C=1, n_jobs=-1)),
]).fit(X_train, y_train)


hybrid_bigram = Pipeline([
    ("features", FeatureUnion([
        ("tfidf_words3000", tfidf_words3000),
        ("pos_bigram", pos_bigram),
        ("doc2vec300", doc2vec300),
        ("glove300", glove300),
    ], n_jobs=-1)),
    ("clf", LogisticRegression(max_iter=1000, C=1, n_jobs=-1)),
]).fit(X_train, y_train)

hybrid_both = Pipeline([
    ("features", FeatureUnion([
        ("tfidf_words3000", tfidf_words3000),
        ("pos_hist45", pos_hist_block),
        ("pos_bigram", pos_bigram),
        ("doc2vec300", doc2vec300),
        ("glove300", glove300),
    ], n_jobs=-1)),
    ("clf", LogisticRegression(max_iter=1000, C=1, n_jobs=-1)),
]).fit(X_train, y_train)

hybrid_ner = Pipeline([
    ("features", seq_union(core_blocks + [("ner", ner_block)])),
    ("clf",      LogisticRegression(max_iter=1000, C=1, n_jobs=-1)),
]).fit(X_train, y_train)


hybrid_pos_ner = Pipeline([
    ("features", seq_union(core_blocks + [
        ("pos_hist", pos_hist_block),
        ("pos_bi",   pos_bigram),
        ("ner",      ner_block)
    ])),
    ("clf",      LogisticRegression(max_iter=1000, C=1, n_jobs=-1)),
]).fit(X_train, y_train)


# --------------------------------------------------------------------------- #
# 12.  Evaluation helper                                                     #
# --------------------------------------------------------------------------- #

def report(name: str, model):
    print(f"\n{name} Dev acc: {accuracy_score(y_dev, model.predict(X_dev)):.4f}")
    print(classification_report(y_dev, model.predict(X_dev)))

    y_test_pred = model.predict(_test.data)
    print(f"{name} Test acc: {accuracy_score(_test.target, y_test_pred):.4f}")
    print(classification_report(_test.target, y_test_pred))

    p, r, f, _ = precision_recall_fscore_support(_test.target, y_test_pred, average="micro")
    print(f"micro avg (test)  precision: {p:.2f}  recall: {r:.2f}  f1: {f:.2f}")


report("TF‑IDF", best_tfidf)
report("GloVe", glove_baseline)
report("Doc2Vec", best_d2v)
report("Hybrid_NoPOS", hybrid_noPOS)
report("Hybrid", hybrid_hist)
report("Hybrid-Bigram", hybrid_bigram)
report("Hybrid-both", hybrid_both)
report("Hybrid + NER",      hybrid_ner)
report("Hybrid + POS+NER",  hybrid_pos_ner)





