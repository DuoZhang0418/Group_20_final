import numpy as np
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Gensim imports for Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

###############################################################################
# 1. Load GloVe vectors from a local file (download from Stanford if needed).
###############################################################################
def load_glove_embedding(file_path):
    """
    Loads GloVe vectors from a file and returns a dictionary
    mapping words -> np.array(dim).
    """
    glove_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            if len(values) < 2:
                continue
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_dict[word] = vector
    return glove_dict

###############################################################################
# 2. GloVeVectorizer: Converts raw text -> average GloVe embedding
###############################################################################
class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, glove_dict):
        """
        glove_dict: dictionary mapping words to GloVe vectors.
        """
        self.glove_dict = glove_dict
        # Infer vector dimension from one entry in the dictionary.
        if len(glove_dict) > 0:
            self.dim = len(next(iter(glove_dict.values())))
        else:
            self.dim = 300  # Default if glove_dict is empty.

    def _clean_text(self, text):
        # Simple tokenization
        tokens = re.findall(r"\w+", text.lower())
        return tokens

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_emb = []
        for doc in X:
            tokens = self._clean_text(doc)
            valid_vectors = [self.glove_dict[t] for t in tokens if t in self.glove_dict]
            if valid_vectors:
                doc_vec = np.mean(valid_vectors, axis=0)
            else:
                doc_vec = np.zeros(self.dim)
            X_emb.append(doc_vec)
        return np.array(X_emb)

###############################################################################
# 3. Doc2VecTransformer: Learns a Doc2Vec model over the training corpus, then
#    infers vectors for any new documents.
###############################################################################
class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 vector_size=100,
                 window=5,
                 min_count=2,
                 epochs=20,
                 dm=1): 
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm

    def fit(self, X, y=None):
        # Convert each doc to a TaggedDocument
        self.train_corpus = [TaggedDocument(simple_preprocess(doc), [i]) for i, doc in enumerate(X)]

        # Build the Doc2Vec model
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            dm=self.dm,
            epochs=self.epochs,
            workers=4
        )
        self.model.build_vocab(self.train_corpus)
        self.model.train(
            self.train_corpus,
            total_examples=self.model.corpus_count,
            epochs=self.epochs
        )
        return self

    def transform(self, X):
        # Infer a vector for each document
        vectors = []
        for doc in X:
            tokens = simple_preprocess(doc)
            vec = self.model.infer_vector(tokens, epochs=self.epochs)
            vectors.append(vec)
        return np.array(vectors)

###############################################################################
# 4. Load the 20 Newsgroups dataset and create train/dev/test splits
###############################################################################
train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')

#print(f"Number of training documents: {len(train_data.data)}")
#print(f"Number of testing documents:  {len(test_data.data)}")

# Create a dev set (20% of the training set).
X_train_full, X_dev, y_train_full, y_dev = train_test_split(
    train_data.data, 
    train_data.target, 
    test_size=0.2,  # 20% dev set
    random_state=42
)
#print(f"Training size:   {len(X_train_full)}")
#print(f"Development size:{len(X_dev)}")

###############################################################################
# 5. TF–IDF Pipeline Baseline with Fixed Classifier Parameter (C=1)
#    We only tune the TfidfVectorizer hyperparams for fairness.
###############################################################################
tfidf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, C=1))  # Fixed C=1
])

# Parameter grid for TF–IDF only
param_grid_tfidf = {
    'tfidf__max_features': [10000, 20000],
    'tfidf__ngram_range': [(1, 1), (1, 2)]
}

grid_search_tfidf = GridSearchCV(
    tfidf_pipeline,
    param_grid_tfidf,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search_tfidf.fit(X_train_full, y_train_full)
print("Best params (TF–IDF):", grid_search_tfidf.best_params_)
best_tfidf_model = grid_search_tfidf.best_estimator_

# Evaluate TF–IDF on Dev
X_dev_transformed = best_tfidf_model.named_steps['tfidf'].transform(X_dev)
y_dev_pred = best_tfidf_model.named_steps['clf'].predict(X_dev_transformed)
print("\n[TF–IDF Baseline] Dev Accuracy:", accuracy_score(y_dev, y_dev_pred))
print("Classification Report (TF–IDF - Dev):")
print(classification_report(y_dev, y_dev_pred))

# Evaluate TF–IDF on Test
X_test_transformed = best_tfidf_model.named_steps['tfidf'].transform(test_data.data)
y_test_pred = best_tfidf_model.named_steps['clf'].predict(X_test_transformed)
print("[TF–IDF Baseline] Test Accuracy:", accuracy_score(test_data.target, y_test_pred))
print("Classification Report (TF–IDF - Test):")
print(classification_report(test_data.target, y_test_pred))

###############################################################################
# 6. GloVe Pipeline Baseline with Fixed Classifier Parameter (C=1)
###############################################################################
glove_dict = load_glove_embedding("glove.6B.300d.txt")  # path to your GloVe file

glove_pipeline = make_pipeline(
    GloveVectorizer(glove_dict),
    LogisticRegression(max_iter=1000, C=1)  # Fixed C=1
)

# Fit GloVe pipeline on training data
glove_pipeline.fit(X_train_full, y_train_full)

# Evaluate GloVe on Dev
y_dev_pred_glove = glove_pipeline.predict(X_dev)
print("\n[GloVe Baseline] Dev Accuracy:", accuracy_score(y_dev, y_dev_pred_glove))
print("Classification Report (GloVe - Dev):")
print(classification_report(y_dev, y_dev_pred_glove))

# Evaluate GloVe on Test
y_test_pred_glove = glove_pipeline.predict(test_data.data)
print("[GloVe Baseline] Test Accuracy:", accuracy_score(test_data.target, y_test_pred_glove))
print("Classification Report (GloVe - Test):")
print(classification_report(test_data.target, y_test_pred_glove))

###############################################################################
# 7. Doc2Vec Pipeline Baseline with Fixed Classifier Parameter (C=1)
#    Demonstrates minimal tuning of Doc2Vec hyperparameters if desired.
###############################################################################
doc2vec_pipeline = Pipeline([
    ('doc2vec', Doc2VecTransformer()),  # The custom transformer
    ('clf', LogisticRegression(max_iter=1000, C=1))  # Fixed C=1
])

# Example parameter grid for Doc2Vec (tweak if you like).
# doc2vec__dm: 1 => 'Distributed Memory'; 0 => 'DBOW'.
param_grid_doc2vec = {
    'doc2vec__vector_size': [300, 600],
    'doc2vec__window': [5, 8],
    'doc2vec__dm': [0, 1],
    # 'doc2vec__epochs': [20], 
}

grid_search_doc2vec = GridSearchCV(
    doc2vec_pipeline,
    param_grid_doc2vec,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid_search_doc2vec.fit(X_train_full, y_train_full)
print("\nBest params (Doc2Vec):", grid_search_doc2vec.best_params_)

best_doc2vec_model = grid_search_doc2vec.best_estimator_

# Evaluate Doc2Vec on Dev
y_dev_pred_doc2vec = best_doc2vec_model.predict(X_dev)
print("\n[Doc2Vec Baseline] Dev Accuracy:", accuracy_score(y_dev, y_dev_pred_doc2vec))
print("Classification Report (Doc2Vec - Dev):")
print(classification_report(y_dev, y_dev_pred_doc2vec))

# Evaluate Doc2Vec on Test
y_test_pred_doc2vec = best_doc2vec_model.predict(test_data.data)
print("[Doc2Vec Baseline] Test Accuracy:", accuracy_score(test_data.target, y_test_pred_doc2vec))
print("Classification Report (Doc2Vec - Test):")
print(classification_report(test_data.target, y_test_pred_doc2vec))


###############################################################################
# 8. Hybrid Model with Pre-Tuned Parameters (No Grid Search)
###############################################################################
from sklearn.pipeline import FeatureUnion

# Get best parameters from individual models
best_tfidf_params = grid_search_tfidf.best_params_
best_doc2vec_params = grid_search_doc2vec.best_params_


# Normalized concatenation
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

# --- 9‑A. 300‑D, unit‑variance TF‑IDF block ----------------------------------
tfidf_300d_scaled = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=best_tfidf_params['tfidf__max_features'],
        ngram_range=best_tfidf_params['tfidf__ngram_range']
    )),
    ('svd', TruncatedSVD(n_components=3000, random_state=42)),
    ('scaler', StandardScaler())             
])

# --- 9‑B. 300‑D, unit‑variance Doc2Vec block ---------------------------------
doc2vec_300d_scaled = Pipeline([
    ('doc2vec', Doc2VecTransformer(
        vector_size=300,
        dm=best_doc2vec_params['doc2vec__dm'],
        window=best_doc2vec_params.get('doc2vec__window', 5),
        epochs=best_doc2vec_params.get('doc2vec__epochs', 20),
        min_count=best_doc2vec_params.get('doc2vec__min_count', 2)
    )), 
    ('scaler', StandardScaler())                # mean‑0, var‑1
])

# --- 9‑C. 300‑D, unit‑variance GloVe block -----------------------------------
glove_300d_scaled = Pipeline([
    ('glove', GloveVectorizer(glove_dict)),     # already 300‑D
    ('scaler', StandardScaler())                # mean‑0, var‑1
])

# --- 9‑D. Concatenate blocks and train single LR -----------------------------
hybrid_concat_pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf300', tfidf_300d_scaled),
        ('doc2vec300', doc2vec_300d_scaled),
        ('glove300', glove_300d_scaled),
    ])),
    ('clf', LogisticRegression(max_iter=1000, C=1, n_jobs=-1))
])

# --- 9‑E. Fit on training data ----------------------------------------------
hybrid_concat_pipeline.fit(X_train_full, y_train_full)

# --- 9‑F. Evaluate on dev set ------------------------------------------------
y_dev_pred = hybrid_concat_pipeline.predict(X_dev)
print("\n[Hybrid Concat] Dev Accuracy:",
      accuracy_score(y_dev, y_dev_pred))
print("Classification Report (Hybrid Concat – Dev):")
print(classification_report(y_dev, y_dev_pred))

# --- 9‑G. Evaluate on test set ----------------------------------------------
y_test_pred = hybrid_concat_pipeline.predict(test_data.data)
print("[Hybrid Concat] Test Accuracy:",
      accuracy_score(test_data.target, y_test_pred))
print("Classification Report (Hybrid Concat – Test):")
print(classification_report(test_data.target, y_test_pred))

