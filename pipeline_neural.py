"""
Neural Baseline Pipeline - Food Hazard Detection (SemEval-2025 Task 9, ST1)
Uses sentence-transformers embeddings + 2-stage LinearSVC classifier.
 
Requirements:
    pip install sentence-transformers scikit-learn pandas scipy
 
Usage:
    python pipeline_neural.py
"""
 
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict
from scipy.sparse import hstack, csr_matrix
 
# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")
test  = pd.read_csv("test.csv")
 
for df in [train, valid, test]:
    df['text'] = df['text'].fillna('')
 
# ──────────────────────────────────────────────
# 2. Sentence-Transformer embeddings
#
# Model options (trade-off speed vs quality):
#   Fast  : "all-MiniLM-L6-v2"      (384-dim,  ~80 MB, no GPU needed)
#   Better: "all-mpnet-base-v2"     (768-dim, ~420 MB, CPU OK)
#   Best  : "intfloat/e5-large-v2"  (1024-dim, needs GPU for speed)
#
# Change MODEL_NAME to experiment — this counts as a methodological comparison!
# ──────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Encoding texts with '{MODEL_NAME}'...")
 
encoder = SentenceTransformer(MODEL_NAME)
 
# encode() returns a numpy float32 array of shape (n_samples, embedding_dim)
train_emb = encoder.encode(train['text'].tolist(), batch_size=64, show_progress_bar=True)
valid_emb = encoder.encode(valid['text'].tolist(), batch_size=64, show_progress_bar=True)
test_emb  = encoder.encode(test['text'].tolist(),  batch_size=64, show_progress_bar=True)

import numpy as np
np.save("train_emb.npy", train_emb)
np.save("valid_emb.npy", valid_emb)
np.save("test_emb.npy", test_emb)
 
# Convert to sparse so we can hstack with the one-hot features later
train_emb_sp = csr_matrix(train_emb)
valid_emb_sp = csr_matrix(valid_emb)
test_emb_sp  = csr_matrix(test_emb)
 
# ──────────────────────────────────────────────
# 3. Stage 1 — Hazard-category classifier
# ──────────────────────────────────────────────
print("\nTraining Stage 1: Hazard classifier...")
hazard_model = LinearSVC(class_weight='balanced', max_iter=3000, C=1.0)
hazard_model.fit(train_emb_sp, train['hazard-category'])
valid_pred_hazard = hazard_model.predict(valid_emb_sp)
 
# Cross-val on train to build unbiased features for Stage 2
print("Cross-validating Stage 1 on training set (5-fold)...")
train_pred_hazard = cross_val_predict(
    LinearSVC(class_weight='balanced', max_iter=3000, C=1.0),
    train_emb_sp,
    train['hazard-category'],
    cv=5,
    n_jobs=-1,
)
 
# ──────────────────────────────────────────────
# 4. Stage 2 — Product-category classifier
#    Features = embeddings + weighted one-hot(hazard prediction)
# ──────────────────────────────────────────────
print("\nBuilding Stage 2 features...")
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
train_hazard_ohe = ohe.fit_transform(train_pred_hazard.reshape(-1, 1)).multiply(5.0)
valid_hazard_ohe = ohe.transform(valid_pred_hazard.reshape(-1, 1)).multiply(5.0)
 
train_product_X = hstack([train_emb_sp, train_hazard_ohe])
valid_product_X = hstack([valid_emb_sp, valid_hazard_ohe])
 
print("Training Stage 2: Product classifier...")
product_model = LinearSVC(class_weight='balanced', max_iter=3000, C=1.0)
product_model.fit(train_product_X, train['product-category'])
valid_pred_product = product_model.predict(valid_product_X)
 
# ──────────────────────────────────────────────
# 5. Evaluation (official SemEval metric)
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("VALIDATION RESULTS")
print("="*50)
 
hazard_f1 = f1_score(valid['hazard-category'], valid_pred_hazard, average='macro')
print(f"\nHazard Macro-F1 : {hazard_f1:.4f}")
 
# Detailed per-class report
print("\n--- Hazard per-class F1 ---")
print(classification_report(valid['hazard-category'], valid_pred_hazard, digits=3))
 
correct_hazard_mask = (valid['hazard-category'] == valid_pred_hazard)
if correct_hazard_mask.sum() > 0:
    product_f1 = f1_score(
        valid.loc[correct_hazard_mask, 'product-category'],
        valid_pred_product[correct_hazard_mask],
        average='macro',
    )
else:
    product_f1 = 0.0
 
print(f"Product Macro-F1 (where hazard correct): {product_f1:.4f}")
 
official_score = (hazard_f1 + product_f1) / 2
print(f"\n>>> Official SemEval Score : {official_score:.4f} <<<")
 
print("\n--- Product per-class F1 ---")
print(classification_report(
    valid.loc[correct_hazard_mask, 'product-category'],
    valid_pred_product[correct_hazard_mask],
    digits=3,
))
 
# ──────────────────────────────────────────────
# 6. Generate Kaggle submission
# ──────────────────────────────────────────────
print("\nGenerating Kaggle submission...")
test_pred_hazard  = hazard_model.predict(test_emb_sp)
test_hazard_ohe   = ohe.transform(test_pred_hazard.reshape(-1, 1)).multiply(5.0)
test_product_X    = hstack([test_emb_sp, test_hazard_ohe])
test_pred_product = product_model.predict(test_product_X)
 
submission = pd.DataFrame({
    'id': test['id'],
    'hazard-category':  test_pred_hazard,
    'product-category': test_pred_product,
})
submission.to_csv('submission_neural.csv', index=False)
print("Saved -> submission_neural.csv")
 
# ──────────────────────────────────────────────
# 7. Quick ablation summary (for the report)
# ──────────────────────────────────────────────
print("\n" + "="*50)
print("ABLATION — Stage 2 WITHOUT hazard feature")
print("="*50)
product_model_ablation = LinearSVC(class_weight='balanced', max_iter=3000, C=1.0)
product_model_ablation.fit(train_emb_sp, train['product-category'])
valid_pred_product_ablation = product_model_ablation.predict(valid_emb_sp)
 
correct_mask_abl = (valid['hazard-category'] == valid_pred_hazard)
if correct_mask_abl.sum() > 0:
    product_f1_abl = f1_score(
        valid.loc[correct_mask_abl, 'product-category'],
        valid_pred_product_ablation[correct_mask_abl],
        average='macro',
    )
else:
    product_f1_abl = 0.0
 
official_abl = (hazard_f1 + product_f1_abl) / 2
print(f"Product Macro-F1 (no hazard feature): {product_f1_abl:.4f}")
print(f"Official Score   (no hazard feature): {official_abl:.4f}")
print(f"\nGain from 2-stage cascade: {official_score - official_abl:+.4f}")

# ──────────────────────────────────────────────
# 8. Confusion Matrices
# ──────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("\nGenerating Confusion Matrices...")
def plot_cm(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Hazard CM
hazard_labels = sorted(valid['hazard-category'].unique())
plot_cm(valid['hazard-category'], valid_pred_hazard, hazard_labels, "Hazard Classification Confusion Matrix", "confusion_hazard.png")

# Product CM (only where hazard is valid, to align with metric, or just overall)
# Let's plot overall product CM
product_labels = sorted(valid['product-category'].unique())
plot_cm(valid['product-category'], valid_pred_product, product_labels, "Product Classification Confusion Matrix", "confusion_product.png")
print("Saved -> confusion_hazard.png and confusion_product.png")