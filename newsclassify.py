from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load train and test data
train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')

# Check how many samples in each set
n_train = len(train_data.data)
n_test = len(test_data.data)
print(f"Number of training documents: {n_train}")
print(f"Number of testing documents:  {n_test}")

# 2. Split off a development set from the original training data
#    (e.g., 80% for actual training, 20% for dev)
X_train_full, X_dev, y_train_full, y_dev = train_test_split(
    train_data.data, 
    train_data.target, 
    test_size=0.2,   # 20% of training data becomes the dev set
    random_state=42
)

print(f"New training set size: {len(X_train_full)}")
print(f"Development set size:  {len(X_dev)}")

# 3. Build a pipeline (TF–IDF -> Logistic Regression) and set up a simple grid search
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),  # We will tune some TF–IDF params
    ('clf', LogisticRegression(max_iter=1000))         # And the regularization parameter C
])

param_grid = {
    # Try different max_features and ngram ranges for the TF–IDF vectorizer
    'tfidf__max_features': [10000, 20000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],

    # Try different regularization strengths for Logistic Regression
    'clf__C': [0.1, 1, 10]
}

# We use cross-validation on our new training set (X_train_full) to find the best parameters
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,               # 3-fold cross-validation
    scoring='accuracy', # Optimize for accuracy
    n_jobs=-1           # Use all available CPU cores; optional
)

# Perform the grid search
grid_search.fit(X_train_full, y_train_full)

# Print the best parameters found
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Evaluate the best model on the dev set
best_model = grid_search.best_estimator_
X_dev_transformed = best_model.named_steps['tfidf'].transform(X_dev)  # Just to illustrate
y_dev_pred = best_model.named_steps['clf'].predict(X_dev_transformed)

dev_accuracy = accuracy_score(y_dev, y_dev_pred)
print(f"Dev Accuracy: {dev_accuracy:.4f}")
print("Classification Report (Dev Set):")
print(classification_report(y_dev, y_dev_pred))

# 4. Final evaluation on the official test set
X_test_transformed = best_model.named_steps['tfidf'].transform(test_data.data)
y_test_pred = best_model.named_steps['clf'].predict(X_test_transformed)

test_accuracy = accuracy_score(test_data.target, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Classification Report (Test Set):")
print(classification_report(test_data.target, y_test_pred))
