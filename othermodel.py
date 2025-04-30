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
        """
        Args:
            vector_size (int): Dimensionality of the feature vectors.
            window (int): The maximum distance between the current and predicted word.
            min_count (int): Ignores all words with total frequency lower than this.
            epochs (int): Number of training epochs.
            dm (int): Defines the training algorithm. 
                      dm=1 uses 'Distributed Memory' (word context + doc vector).
                      dm=0 uses 'DBOW' (distributed bag-of-words).
        """
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
    'doc2vec__vector_size': [100, 300],
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

# Build hybrid pipeline with best params


# TFIDF+Doc
'''
hybrid_pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=best_tfidf_params['tfidf__max_features'],
            ngram_range=best_tfidf_params['tfidf__ngram_range']
        )),
        ('doc2vec', Doc2VecTransformer(
            vector_size=best_doc2vec_params['doc2vec__vector_size'],
            dm=best_doc2vec_params['doc2vec__dm'],
            window=best_doc2vec_params.get('doc2vec__window', 5),
            epochs=best_doc2vec_params.get('doc2vec__epochs', 20),
            min_count=best_doc2vec_params.get('doc2vec__min_count', 2)
        ))
    ])),
    ('clf', LogisticRegression(max_iter=1000, C=1))
])

# Train and evaluate
hybrid_pipeline.fit(X_train_full, y_train_full)

# Dev set
y_dev_pred = hybrid_pipeline.predict(X_dev)
print("\n[Hybrid Model] Dev Accuracy:", accuracy_score(y_dev, y_dev_pred))
print("Classification Report (Hybrid - Dev):")
print(classification_report(y_dev, y_dev_pred))

# Test set
y_test_pred = hybrid_pipeline.predict(test_data.data)
print("[Hybrid Model] Test Accuracy:", accuracy_score(test_data.target, y_test_pred))
print("Classification Report (Hybrid - Test):")
print(classification_report(test_data.target, y_test_pred))


'''








# 3 models no compress
'''

# Modify the FeatureUnion to include GloVe
hybrid_pipeline = Pipeline([
    ('features', FeatureUnion([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=best_tfidf_params['tfidf__max_features'],
            ngram_range=best_tfidf_params['tfidf__ngram_range']
        )),
        ('doc2vec', Doc2VecTransformer(
            vector_size=best_doc2vec_params['doc2vec__vector_size'],
            dm=best_doc2vec_params['doc2vec__dm'],
            window=best_doc2vec_params.get('doc2vec__window', 5),
            epochs=best_doc2vec_params.get('doc2vec__epochs', 20),
            min_count=best_doc2vec_params.get('doc2vec__min_count', 2)
        )),
        ('glove', GloveVectorizer(glove_dict))  # Add GloVe
    ])),
    ('clf', LogisticRegression(max_iter=1000, C=1))
])

# Train the updated hybrid model
hybrid_pipeline.fit(X_train_full, y_train_full)

# Evaluate on Dev
y_dev_pred = hybrid_pipeline.predict(X_dev)
print("\n[Hybrid Model (TF-IDF + Doc2Vec + GloVe)] Dev Accuracy:", accuracy_score(y_dev, y_dev_pred))
print("Classification Report (Hybrid - Dev):")
print(classification_report(y_dev, y_dev_pred))

# Evaluate on Test
y_test_pred = hybrid_pipeline.predict(test_data.data)
print("[Hybrid Model (TF-IDF + Doc2Vec + GloVe)] Test Accuracy:", accuracy_score(test_data.target, y_test_pred))
print("Classification Report (Hybrid - Test):")
print(classification_report(test_data.target, y_test_pred))

'''


# 3 model with compress
'''
###############################################################################
# 8. Hybrid Model with Compressed TF-IDF + Doc2Vec + GloVe
###############################################################################
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD  # NEW

# Get best parameters from individual models
best_tfidf_params = grid_search_tfidf.best_params_
best_doc2vec_params = grid_search_doc2vec.best_params_

# Build hybrid pipeline with TF-IDF compression
hybrid_pipeline = Pipeline([
    ('features', FeatureUnion([
        # TF-IDF with dimensionality reduction
        ('tfidf', Pipeline([
            ('vectorizer', TfidfVectorizer(
                stop_words='english',
                max_features=best_tfidf_params['tfidf__max_features'],
                ngram_range=best_tfidf_params['tfidf__ngram_range']
            )),
            ('svd', TruncatedSVD(n_components=8000))  # Compress to nD
        ])),
        ('doc2vec', Doc2VecTransformer(
            vector_size=best_doc2vec_params['doc2vec__vector_size'],
            dm=best_doc2vec_params['doc2vec__dm'],
            window=best_doc2vec_params.get('doc2vec__window', 5),
            epochs=best_doc2vec_params.get('doc2vec__epochs', 20),
            min_count=best_doc2vec_params.get('doc2vec__min_count', 2)
        )),
        ('glove', GloveVectorizer(glove_dict))
    ])),
    ('clf', LogisticRegression(max_iter=1000, C=1))
])

# Train and evaluate
hybrid_pipeline.fit(X_train_full, y_train_full)

# Dev set
y_dev_pred = hybrid_pipeline.predict(X_dev)
print("\n[Hybrid Model] Dev Accuracy:", accuracy_score(y_dev, y_dev_pred))
print("Classification Report (Hybrid - Dev):")
print(classification_report(y_dev, y_dev_pred))

# Test set
y_test_pred = hybrid_pipeline.predict(test_data.data)
print("[Hybrid Model] Test Accuracy:", accuracy_score(test_data.target, y_test_pred))
print("Classification Report (Hybrid - Test):")
print(classification_report(test_data.target, y_test_pred))
'''




# Sum up 3 models in 300D
'''
from sklearn.decomposition import TruncatedSVD

# 1. Ensure all vectors are 300D by adjusting components
best_tfidf_params = grid_search_tfidf.best_params_
best_doc2vec_params = grid_search_doc2vec.best_params_

# Create dimension-matched components
tfidf_300d = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=best_tfidf_params['tfidf__max_features'],
        ngram_range=best_tfidf_params['tfidf__ngram_range']
    )),
    ('svd', TruncatedSVD(n_components=300))  # Force 300D
])

doc2vec_300d = Doc2VecTransformer(
    vector_size=300,  # Force 300D regardless of best params
    dm=best_doc2vec_params['doc2vec__dm'],
    window=best_doc2vec_params.get('doc2vec__window', 5),
    epochs=best_doc2vec_params.get('doc2vec__epochs', 20),
    min_count=best_doc2vec_params.get('doc2vec__min_count', 2)
)

glove_300d = GloveVectorizer(glove_dict)  # Already 300D

# 2. Custom transformer for weighted summation
class VectorSumTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizers):
        self.vectorizers = vectorizers
        
    def fit(self, X, y=None):
        for vec in self.vectorizers:
            vec.fit(X, y)
        return self
    
    def transform(self, X):
        vectors = [vec.transform(X) for vec in self.vectorizers]
        
        # Validate dimensions
        shapes = [v.shape[1] for v in vectors]
        if len(set(shapes)) != 1:
            raise ValueError(f"Vectors must have same dimension. Got dimensions: {shapes}")
            
        return np.sum(vectors, axis=0)  # Equal weighting

# 3. Create hybrid pipeline
hybrid_sum_pipeline = Pipeline([
    ('sum', VectorSumTransformer([tfidf_300d, doc2vec_300d, glove_300d])),
    ('clf', LogisticRegression(max_iter=1000, C=1))
])

# 4. Train and evaluate
hybrid_sum_pipeline.fit(X_train_full, y_train_full)

# Dev set evaluation
y_dev_pred = hybrid_sum_pipeline.predict(X_dev)
print("\n[Hybrid Sum] Dev Accuracy:", accuracy_score(y_dev, y_dev_pred))
print("Classification Report (Hybrid Sum - Dev):")
print(classification_report(y_dev, y_dev_pred))

# Test set evaluation
y_test_pred = hybrid_sum_pipeline.predict(test_data.data)
print("[Hybrid Sum] Test Accuracy:", accuracy_score(test_data.target, y_test_pred))
print("Classification Report (Hybrid Sum - Test):")
print(classification_report(test_data.target, y_test_pred))

'''


###############################################################################


'''
# 9. Hybrid Model: Weighted Average of TF-IDF (300D) + Doc2Vec (300D) + GloVe (300D)
#    with Grid Search over weights
###############################################################################
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

# --- Step 1: Ensure all representations are 300D ---

# Use best TFIDF parameters from grid_search_tfidf (assumed defined earlier)
best_tfidf_params = grid_search_tfidf.best_params_

# Use best Doc2Vec parameters from grid_search_doc2vec (assumed defined earlier)
best_doc2vec_params = grid_search_doc2vec.best_params_

# Create a TFIDF pipeline that outputs 300D using TruncatedSVD
tfidf_300d = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=best_tfidf_params['tfidf__max_features'],
        ngram_range=best_tfidf_params['tfidf__ngram_range']
    )),
    ('svd', TruncatedSVD(n_components=300))
])

# Create a Doc2Vec transformer forced to output 300D
doc2vec_300d = Doc2VecTransformer(
    vector_size=300,  # Force 300D regardless of best params
    dm=best_doc2vec_params['doc2vec__dm'],
    window=best_doc2vec_params.get('doc2vec__window', 5),
    epochs=best_doc2vec_params.get('doc2vec__epochs', 20),
    min_count=best_doc2vec_params.get('doc2vec__min_count', 2)
)

# GloVe transformer should already produce 300D (assuming you loaded the 300D file)
glove_300d = GloveVectorizer(glove_dict)  # glove_dict is assumed defined earlier

# --- Step 2: Define a custom transformer for a weighted average ---
class WeightedSumTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizers, weights=None, average=True):
        """
        vectorizers: List of transformers (each must produce the same output dimension).
        weights: List of floats representing the relative weights.
        average: If True, returns a weighted average (dividing by the sum of weights);
                 otherwise returns the weighted sum.
        """
        self.vectorizers = vectorizers
        # Default to equal weights if none specified.
        self.weights = weights if weights is not None else [1.0] * len(vectorizers)
        self.average = average

    def fit(self, X, y=None):
        for vec in self.vectorizers:
            vec.fit(X, y)
        return self

    def transform(self, X):
        # Transform X with each vectorizer
        vectors_list = [vec.transform(X) for vec in self.vectorizers]
        
        # Ensure all vectors have the same dimension
        dims = [v.shape[1] for v in vectors_list]
        if len(set(dims)) != 1:
            raise ValueError(f"All vectors must have the same dimension, got: {dims}")
        
        # Multiply each vector by its corresponding weight and sum them
        weighted_vectors = [w * v for w, v in zip(self.weights, vectors_list)]
        fused = np.sum(weighted_vectors, axis=0)
        
        # Return weighted average if desired
        if self.average:
            fused = fused / np.sum(self.weights)
        return fused

# --- Step 3: Create the hybrid pipeline ---
hybrid_weighted_pipeline = Pipeline([
    ('weighted_sum', WeightedSumTransformer([tfidf_300d, doc2vec_300d, glove_300d], average=True)),
    ('clf', LogisticRegression(max_iter=1000, C=1))
])

# --- Step 4: Define a parameter grid for different weight combinations ---


param_grid_weights = {
    'weighted_sum__weights': [
        # — balanced three‑way —
        [1, 1, 1],

        # — two‑way hard combinations —
        [1, 1, 0],  # TF–IDF + Doc2Vec
        [1, 0, 1],  # TF–IDF + GloVe
        [0, 1, 1],  # Doc2Vec + GloVe

        # — two‑way soft combinations (sum=1) —
        [0.5, 0.5, 0.0],
        [0.6, 0.4, 0.0],
        [0.4, 0.6, 0.0],

        [0.5, 0.0, 0.5],
        [0.6, 0.0, 0.4],
        [0.4, 0.0, 0.6],

        [0.0, 0.5, 0.5],
        [0.0, 0.6, 0.4],
        [0.0, 0.4, 0.6],

        # — three‑way soft distributions —
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],

        # — moderate three‑way emphasis —
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6],

        # — strong three‑way emphasis —
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],

        # — your best‑performing floats —
        [0.3, 0.3, 0.4],
        [0.3, 0.4, 0.3],
        [0.4, 0.3, 0.3],
    ]
}



# --- Step 5: Run Grid Search to find the best weight configuration on the dev set ---
grid_search_weighted = GridSearchCV(
    hybrid_weighted_pipeline,
    param_grid_weights,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_weighted.fit(X_train_full, y_train_full)
print("Best weights found:", grid_search_weighted.best_params_)

best_hybrid_weighted_model = grid_search_weighted.best_estimator_

# --- Step 6: Evaluate on the dev set ---
y_dev_pred = best_hybrid_weighted_model.predict(X_dev)
print("\n[Hybrid Weighted Average] Dev Accuracy:", accuracy_score(y_dev, y_dev_pred))
print("Classification Report (Hybrid Weighted Average - Dev):")
print(classification_report(y_dev, y_dev_pred))

# --- Step 7: Evaluate on the test set ---
y_test_pred = best_hybrid_weighted_model.predict(test_data.data)
print("[Hybrid Weighted Average] Test Accuracy:", accuracy_score(test_data.target, y_test_pred))
print("Classification Report (Hybrid Weighted Average - Test):")
print(classification_report(test_data.target, y_test_pred))



'''



# Weighted Average (or Sum) of TF-IDF (300D) + Doc2Vec (300D) + GloVe (300D) with Grid Search over weights
'''

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

# --- Step 1: Ensure all representations are 300D ---

# Use best TFIDF parameters from grid_search_tfidf (assumed to be defined earlier)
best_tfidf_params = grid_search_tfidf.best_params_

# Use best Doc2Vec parameters from grid_search_doc2vec (assumed to be defined earlier)
best_doc2vec_params = grid_search_doc2vec.best_params_

# Create a TFIDF pipeline that outputs 300D using TruncatedSVD
tfidf_300d = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=best_tfidf_params['tfidf__max_features'],
        ngram_range=best_tfidf_params['tfidf__ngram_range']
    )),
    ('svd', TruncatedSVD(n_components=300))
])

# Create a Doc2Vec transformer forced to output 300D
doc2vec_300d = Doc2VecTransformer(
    vector_size=300,  # Force 300D regardless of best params
    dm=best_doc2vec_params['doc2vec__dm'],
    window=best_doc2vec_params.get('doc2vec__window', 5),
    epochs=best_doc2vec_params.get('doc2vec__epochs', 20),
    min_count=best_doc2vec_params.get('doc2vec__min_count', 2)
)

# GloVe transformer should already produce 300D (assuming you loaded the 300D file)
glove_300d = GloveVectorizer(glove_dict)  # glove_dict is assumed defined earlier

# --- Step 2: Define a custom transformer for a weighted sum/average ---
class WeightedSumTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizers, weights=None, average=False):
        """
        vectorizers: List of transformers (each must produce the same output dimension).
        weights: List of floats representing relative weights.
        average: If True, returns a weighted average (dividing by sum of weights),
                 otherwise returns the weighted sum.
        """
        self.vectorizers = vectorizers
        # Default to 1.0 each if no weights specified.
        self.weights = weights if weights is not None else [1.0] * len(vectorizers)
        self.average = average

    def fit(self, X, y=None):
        for vec in self.vectorizers:
            vec.fit(X, y)
        return self

    def transform(self, X):
        # Transform X with each vectorizer
        vectors_list = [vec.transform(X) for vec in self.vectorizers]
        
        # Ensure all vectors have the same dimension
        dims = [v.shape[1] for v in vectors_list]
        if len(set(dims)) != 1:
            raise ValueError(f"All vectors must have the same dimension, got: {dims}")
        
        # Multiply each vector by its weight and sum them
        weighted_vectors = [w * v for w, v in zip(self.weights, vectors_list)]
        fused = np.sum(weighted_vectors, axis=0)
        
        if self.average:
            fused = fused / np.sum(self.weights)
        return fused

# --- Step 3: Create the hybrid pipeline ---
hybrid_weighted_pipeline = Pipeline([
    ('weighted_sum', WeightedSumTransformer([tfidf_300d, doc2vec_300d, glove_300d])),
    ('clf', LogisticRegression(max_iter=1000, C=1))
])

# --- Step 4: Define a parameter grid for weight combinations ---
# These weight triples are examples; adjust as needed.
param_grid_weights = {
    'weighted_sum__weights': [
        [1.0, 1.0, 1.0],       # All equal: weighted sum.
        [0.5, 1.0, 1.0],       # Slight emphasis on doc2vec + glove.
        [1.0, 0.5, 1.5],       # Emphasis on GloVe.
        [2.0, 1.0, 1.0],       # Emphasis on TF-IDF.
        [0.5, 0.5, 2.0],       # Strong emphasis on GloVe.
    ],
    # If you want a weighted average instead of sum, you can add:
    #'weighted_sum__average': [True, False]
}


# --- Step 5: Run Grid Search on the dev set ---
grid_search_weighted = GridSearchCV(
    hybrid_weighted_pipeline,
    param_grid_weights,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_weighted.fit(X_train_full, y_train_full)
print("Best weights found:", grid_search_weighted.best_params_)

best_hybrid_weighted_model = grid_search_weighted.best_estimator_

# --- Step 6: Evaluate on the dev set ---
y_dev_pred = best_hybrid_weighted_model.predict(X_dev)
print("\n[Hybrid Weighted] Dev Accuracy:", accuracy_score(y_dev, y_dev_pred))
print("Classification Report (Hybrid Weighted - Dev):")
print(classification_report(y_dev, y_dev_pred))

# --- Step 7: Evaluate on the test set ---
y_test_pred = best_hybrid_weighted_model.predict(test_data.data)
print("[Hybrid Weighted] Test Accuracy:", accuracy_score(test_data.target, y_test_pred))
print("Classification Report (Hybrid Weighted - Test):")
print(classification_report(test_data.target, y_test_pred))


'''

