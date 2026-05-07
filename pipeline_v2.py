"""
pipeline_v2.py — Improved Classical Baseline
Food Hazard Detection (SemEval-2025 Task 9, ST1)
NLP 053 - CSE UOI 2026

Improvements over v1:
  - Combines title + text into a richer input field
  - Adds country and year as categorical features
  - Text preprocessing (lowercase, punctuation, stopwords via TF-IDF's built-in tools)
  - Per-class F1 report for the report/presentation
  - Confusion matrix export
  - Cleaner score reporting
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.model_selection import cross_val_predict, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")
test  = pd.read_csv("test.csv")

print(f"Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")

# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────

def build_text_field(df):
    """
    Combine title and full text.
    Title is repeated 3× to give it higher TF-IDF weight
    (titles are short but highly informative).
    """
    title = df['title'].fillna('') if 'title' in df.columns else pd.Series([''] * len(df))
    text  = df['text'].fillna('')  if 'text'  in df.columns else pd.Series([''] * len(df))
    return (title + ' ') * 3 + text

train['input'] = build_text_field(train)
valid['input'] = build_text_field(valid)
test['input']  = build_text_field(test)

# ─────────────────────────────────────────────
# 3. TF-IDF (unigrams + bigrams, with stopword removal)
# ─────────────────────────────────────────────
print("Vectorizing text (unigrams + bigrams, max 30k features)...")
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    sublinear_tf=True,       # log-scaling of TF — helps with long texts
    strip_accents='unicode',
    analyzer='word',
    stop_words='english',
    min_df=2                  # ignore tokens that appear in only 1 document
)
train_tfidf = tfidf.fit_transform(train['input'])
valid_tfidf = tfidf.transform(valid['input'])
test_tfidf  = tfidf.transform(test['input'])

# ─────────────────────────────────────────────
# 4. Optional: add country as a categorical feature
# ─────────────────────────────────────────────
meta_cols = [c for c in ['country', 'year'] if c in train.columns]

if meta_cols:
    print(f"Adding metadata features: {meta_cols}")
    meta_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    train_meta = meta_enc.fit_transform(train[meta_cols].fillna('unknown'))
    valid_meta = meta_enc.transform(valid[meta_cols].fillna('unknown'))
    test_meta  = meta_enc.transform(test[meta_cols].fillna('unknown'))

    train_tfidf = hstack([train_tfidf, train_meta])
    valid_tfidf = hstack([valid_tfidf, valid_meta])
    test_tfidf  = hstack([test_tfidf,  test_meta])

# ─────────────────────────────────────────────
# 5. Stage 1 — Hazard-category classifier
# ─────────────────────────────────────────────
# Add this before training to find the best C
print("\nFinding best C for Hazard-category...")
param_grid = {'C': [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]}
grid = GridSearchCV(
    LinearSVC(class_weight='balanced', max_iter=3000),
    param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
)
grid.fit(train_tfidf, train['hazard-category'])
best_c = grid.best_params_['C']
print(f"Best C: {best_c}")
print(f"Best CV score: {grid.best_score_:.4f}")

print("\nTraining Stage 1: Hazard-category (LinearSVC) with best C...")
hazard_model = LinearSVC(class_weight='balanced', C=best_c, max_iter=3000)
hazard_model.fit(train_tfidf, train['hazard-category'])

valid_pred_hazard = hazard_model.predict(valid_tfidf)

# Cross-val on train to avoid leakage for Stage 2 input
print("Cross-validating hazard predictions on train (5-fold)...")
train_pred_hazard = cross_val_predict(
    LinearSVC(class_weight='balanced', C=best_c, max_iter=3000),
    train_tfidf,
    train['hazard-category'],
    cv=5,
    n_jobs=-1
)

# ─────────────────────────────────────────────
# 6. Stage 2 — Product-category classifier
# ─────────────────────────────────────────────
print("Building Stage 2 features (TF-IDF + hazard prediction)...")
ohe_hazard = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
train_hazard_ohe = ohe_hazard.fit_transform(train_pred_hazard.reshape(-1, 1)).multiply(5.0)
valid_hazard_ohe = ohe_hazard.transform(valid_pred_hazard.reshape(-1, 1)).multiply(5.0)

train_prod_feats = hstack([train_tfidf, train_hazard_ohe])
valid_prod_feats = hstack([valid_tfidf, valid_hazard_ohe])

print("Training Stage 2: Product-category (LinearSVC) with best C...")
product_model = LinearSVC(class_weight='balanced', C=best_c, max_iter=3000)
product_model.fit(train_prod_feats, train['product-category'])

valid_pred_product = product_model.predict(valid_prod_feats)

# ─────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("VALIDATION RESULTS")
print("="*55)

hazard_macro_f1 = f1_score(valid['hazard-category'], valid_pred_hazard, average='macro')
print(f"\nHazard Macro-F1:  {hazard_macro_f1:.4f}")

correct_mask = valid['hazard-category'] == valid_pred_hazard
print(f"Hazard correct:   {correct_mask.sum()}/{len(valid)} ({correct_mask.mean()*100:.1f}%)")

if correct_mask.sum() > 0:
    product_macro_f1 = f1_score(
        valid.loc[correct_mask, 'product-category'],
        valid_pred_product[correct_mask],
        average='macro'
    )
else:
    product_macro_f1 = 0.0

print(f"Product Macro-F1 (on correct hazard): {product_macro_f1:.4f}")
official_score = (hazard_macro_f1 + product_macro_f1) / 2
print(f"\n★ Official SemEval Score: {official_score:.4f}")
print("="*55)

# Per-class breakdown (important for the report!)
print("\n--- Hazard-category per-class F1 ---")
print(classification_report(valid['hazard-category'], valid_pred_hazard, digits=3))

print("\n--- Product-category per-class F1 (all valid examples) ---")
print(classification_report(valid['product-category'], valid_pred_product, digits=3))

# ─────────────────────────────────────────────
# 8. Confusion matrix plots (save for report)
# ─────────────────────────────────────────────
def plot_confusion(y_true, y_pred, title, filename):
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels)-2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

plot_confusion(valid['hazard-category'], valid_pred_hazard,
               'Hazard-Category Confusion Matrix (Validation)',
               'confusion_hazard.png')

plot_confusion(valid['product-category'], valid_pred_product,
               'Product-Category Confusion Matrix (Validation)',
               'confusion_product.png')

# ─────────────────────────────────────────────
# 9. Kaggle submission
# ─────────────────────────────────────────────
print("\nGenerating Kaggle submission...")
test_pred_hazard = hazard_model.predict(test_tfidf)
test_hazard_ohe  = ohe_hazard.transform(test_pred_hazard.reshape(-1, 1)).multiply(5.0)
test_prod_feats  = hstack([test_tfidf, test_hazard_ohe])
test_pred_product = product_model.predict(test_prod_feats)

submission = pd.DataFrame({
    'id': test['id'],
    'hazard-category': test_pred_hazard,
    'product-category': test_pred_product
})
submission.to_csv('submission_v2.csv', index=False)
print("Saved → submission_v2.csv")
print("\nDone!")
