'''
TFIDF individual baseline 
+ POS histogram and bigram features
+ Ner features
'''
import warnings, random, numpy as np, nltk, spacy
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline  import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics      import accuracy_score, classification_report
from nltk import word_tokenize, pos_tag
from nltk.util import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
random.seed(SEED);  np.random.seed(SEED)

for pkg in ("punkt", "averaged_perceptron_tagger"):
    try: nltk.data.find(f"tokenizers/{pkg}" if pkg=="punkt" else f"taggers/{pkg}")
    except LookupError: nltk.download(pkg)

# POS-histogram 
SELECTED_TAGS = [
    "CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS",
    "PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH",
    "VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB"
]
TAG2IDX = {t:i for i,t in enumerate(SELECTED_TAGS)}

def pos_histogram(doc:str) -> np.ndarray:
    vec = np.zeros(len(SELECTED_TAGS), dtype=np.float32)
    for _, tag in nltk.pos_tag(nltk.word_tokenize(doc)):
        idx = TAG2IDX.get(tag)
        if idx is not None:
            vec[idx] += 1
    s = vec.sum();  return vec if s == 0 else vec / s

pos_hist_block = Pipeline([
    ("hist", FunctionTransformer(
        lambda D: np.vstack([pos_histogram(d) for d in D]), validate=False)),
    ("scal", StandardScaler())    
])

# POS bigram TF-IDF block
def pos_ngrams(doc, n=2):
    tags = [t for _, t in pos_tag(word_tokenize(doc))]
    return ["_".join(g) for g in ngrams(tags, n)]

pos_bigram_block = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="word",
                              tokenizer=pos_ngrams,
                              token_pattern=None,
                              max_features=5_000)),
    ("svd",  TruncatedSVD(n_components=1_500, random_state=SEED)),
    ("scal", StandardScaler(with_mean=False))     # sparse in → keep sparse
])

# NER histogram block
try:
    _NLP = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
except OSError:
    from spacy.cli import download; download("en_core_web_sm")
    _NLP = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

NER_TYPES = ["PERSON","NORP","FAC","ORG","GPE","LOC",
             "PRODUCT","EVENT","DATE","TIME","CARDINAL","MONEY"]
NER2IDX = {lbl:i for i,lbl in enumerate(NER_TYPES)}

class NerHistogram(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = np.zeros((len(X), len(NER_TYPES)), dtype=np.float32)
        for i, doc in enumerate(_NLP.pipe(X, batch_size=32)):
            for ent in doc.ents:
                j = NER2IDX.get(ent.label_)
                if j is not None:
                    out[i, j] += 1
        #row_sums = out.sum(1, keepdims=True)
        #out[row_sums != 0] /= row_sums[row_sums != 0]
        row_sums = out.sum(axis=1)
        nonzero = row_sums > 0
        out[nonzero] = out[nonzero] / row_sums[nonzero, np.newaxis]
        return out

ner_block = Pipeline([
    ("hist", NerHistogram()),
    ("scal", StandardScaler())     
])

# 20 Newsgroups Dataset
train = fetch_20newsgroups(subset="train");  test = fetch_20newsgroups(subset="test")
X_tr, X_dev, y_tr, y_dev = train_test_split(train.data, train.target,
                                            test_size=0.2, random_state=SEED)

# baseline TF-IDF with grid search
grid = GridSearchCV(
    Pipeline([("tfidf", TfidfVectorizer(stop_words="english")),
              ("clf",   LogisticRegression(max_iter=1_000, C=1))]),
    {"tfidf__max_features":[10_000, 20_000],
     "tfidf__ngram_range" :[(1,1), (1,2)]},
    cv=3, scoring="accuracy", n_jobs=-1
)
grid.fit(X_tr, y_tr)
best_tfidf_params = grid.best_params_
baseline = grid.best_estimator_

tfidf_block = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english",
                              max_features=best_tfidf_params["tfidf__max_features"],
                              ngram_range = best_tfidf_params["tfidf__ngram_range"])),
    ("scal",  StandardScaler(with_mean=False))
])

def lr(): return LogisticRegression(max_iter=1_000, C=1, n_jobs=-1)

hybrid_hist = Pipeline([
    ("features", FeatureUnion([("tfidf",tfidf_block),
                               ("pos_hist",pos_hist_block)])),
    ("clf", lr())
]).fit(X_tr, y_tr)

hybrid_bi   = Pipeline([
    ("features", FeatureUnion([("tfidf",tfidf_block),
                               ("pos_bi",pos_bigram_block)])),
    ("clf", lr())
]).fit(X_tr, y_tr)

hybrid_both = Pipeline([
    ("features", FeatureUnion([("tfidf",tfidf_block),
                               ("pos_hist",pos_hist_block),
                               ("pos_bi",pos_bigram_block)])),
    ("clf", lr())
]).fit(X_tr, y_tr)

hybrid_ner  = Pipeline([
    ("features", FeatureUnion([("tfidf",tfidf_block),
                               ("ner",   ner_block)])),
    ("clf", lr())
]).fit(X_tr, y_tr)

hybrid_all  = Pipeline([
    ("features", FeatureUnion([("tfidf",tfidf_block),
                               ("pos_hist",pos_hist_block),
                               ("pos_bi",  pos_bigram_block),
                               ("ner",     ner_block)])),
    ("clf", lr())
]).fit(X_tr, y_tr)

# Evaluation 
def report(name, model):
    dev_acc  = accuracy_score(y_dev, model.predict(X_dev))
    y_pred   = model.predict(test.data)
    test_acc = accuracy_score(test.target, y_pred)
    print(f"\n{name:18s}  Dev acc: {dev_acc:.4f}   Test acc: {test_acc:.4f}")
    print(classification_report(test.target, y_pred, digits=3))

for lbl, mdl in [("TF-IDF baseline", baseline),
                 ("+ POS-hist",      hybrid_hist),
                 ("+ POS-bigram",    hybrid_bi),
                 ("+ POS hist+bi",   hybrid_both),
                 ("+ NER",           hybrid_ner),
                 ("+ POS+NER",       hybrid_all)]:
    report(lbl, mdl)




