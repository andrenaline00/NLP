"""
pipeline_v3.py — BERT Embeddings + LinearSVC
Food Hazard Detection (SemEval-2025 Task 9, ST1)

Αναβάθμιση από v2:
  - Αντί TF-IDF → frozen DistilBERT [CLS] embeddings (768-dim)
  - Ίδιο 2-Stage concept (Hazard → Product)
  - Σύγκριση score με v2 στο τέλος

Requirements:
  pip install transformers torch scikit-learn pandas numpy
"""

import torch
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 32
MAX_LEN    = 128

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}")

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("\nLoading data...")
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")
test  = pd.read_csv("test.csv")
print(f"Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")

def build_text(df):
    """Ίδια λογική με v2: title x3 + text"""
    title = df['title'].fillna('') if 'title' in df.columns else pd.Series([''] * len(df))
    text  = df['text'].fillna('')
    return ((title + ' ') * 3 + text).tolist()

train_texts = build_text(train)
valid_texts = build_text(valid)
test_texts  = build_text(test)

# ─────────────────────────────────────────────
# 2. BERT Embeddings
# ─────────────────────────────────────────────
print("\nLoading BERT model...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert_model.eval()

def get_embeddings(texts, batch_size=BATCH_SIZE):
    """
    Εξάγει [CLS] embeddings από κάθε κείμενο.
    Output shape: [N, 768]
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        # Μεταφέρουμε στο GPU αν υπάρχει
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            output = bert_model(**encoded)

        # [CLS] token = index 0
        cls_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_emb)

        if (i // batch_size) % 5 == 0:
            print(f"  {min(i+batch_size, len(texts))}/{len(texts)} texts processed...")

    return np.vstack(all_embeddings)

print("\nExtracting train embeddings (αυτό παίρνει λίγο χρόνο)...")
X_train = get_embeddings(train_texts)

print("Extracting valid embeddings...")
X_valid = get_embeddings(valid_texts)

print("Extracting test embeddings...")
X_test  = get_embeddings(test_texts)

print(f"\nEmbedding shape: {X_train.shape}")  # (N, 768)

# ─────────────────────────────────────────────
# 3. Labels
# ─────────────────────────────────────────────
yh_train = train['hazard-category'].values
yp_train = train['product-category'].values
yh_valid = valid['hazard-category'].values
yp_valid = valid['product-category'].values

# ─────────────────────────────────────────────
# 4. Stage 1 — Hazard Classifier
# ─────────────────────────────────────────────
print("\nTraining Stage 1: Hazard (LinearSVC on BERT embeddings)...")
hazard_clf = LinearSVC(class_weight='balanced', C=1.0, max_iter=3000)
hazard_clf.fit(X_train, yh_train)

valid_pred_hazard = hazard_clf.predict(X_valid)
test_pred_hazard  = hazard_clf.predict(X_test)

f1_haz = f1_score(yh_valid, valid_pred_hazard, average='macro', zero_division=0)
print(f"  F1 Hazard (valid): {f1_haz:.4f}")

# Cross-val predictions για Stage 2 training (αποφυγή leakage)
from sklearn.model_selection import cross_val_predict
print("Cross-validating hazard predictions (5-fold)...")
train_pred_hazard_cv = cross_val_predict(
    LinearSVC(class_weight='balanced', C=1.0, max_iter=3000),
    X_train, yh_train, cv=5, n_jobs=-1
)

# ─────────────────────────────────────────────
# 5. Stage 2 — Product Classifier
#    Input = BERT embedding (768) + hazard one-hot (10)
# ─────────────────────────────────────────────
print("\nBuilding Stage 2 features...")
n_hazard_classes = len(np.unique(yh_train))
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Training: χρησιμοποιούμε cross-val predictions (όχι ground truth)
train_haz_ohe = ohe.fit_transform(train_pred_hazard_cv.reshape(-1, 1))
valid_haz_ohe = ohe.transform(valid_pred_hazard.reshape(-1, 1))
test_haz_ohe  = ohe.transform(test_pred_hazard.reshape(-1, 1))

# Multiply by 5 για να δώσουμε βάρος στο hazard signal (ίδιο με v2)
train_stage2 = np.hstack([X_train, train_haz_ohe * 5.0])
valid_stage2 = np.hstack([X_valid, valid_haz_ohe * 5.0])
test_stage2  = np.hstack([X_test,  test_haz_ohe  * 5.0])

print("Training Stage 2: Product (LinearSVC)...")
product_clf = LinearSVC(class_weight='balanced', C=1.0, max_iter=3000)
product_clf.fit(train_stage2, yp_train)

valid_pred_product = product_clf.predict(valid_stage2)
test_pred_product  = product_clf.predict(test_stage2)

# ─────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────
def st1_score(yh_true, yh_pred, yp_true, yp_pred, label=""):
    f1_haz  = f1_score(yh_true, yh_pred, average='macro', zero_division=0)
    mask    = (np.array(yh_true) == np.array(yh_pred))
    f1_prod = f1_score(
        np.array(yp_true)[mask], np.array(yp_pred)[mask],
        average='macro', zero_division=0
    ) if mask.sum() > 0 else 0.0
    score = (f1_haz + f1_prod) / 2
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  Hazard correct: {mask.sum()}/{len(mask)} ({mask.mean()*100:.1f}%)")
    print(f"  F1 Hazard:      {f1_haz:.4f}")
    print(f"  F1 Product:     {f1_prod:.4f}")
    print(f"  ST1 Score:      {score:.4f}")
    print(f"{'='*55}")
    return {"f1_hazard": round(f1_haz,4), "f1_product": round(f1_prod,4), "score": round(score,4)}

scores_v3 = st1_score(
    yh_valid, valid_pred_hazard,
    yp_valid, valid_pred_product,
    label="pipeline_v3 (BERT + LinearSVC)"
)

print("\n--- Hazard per-class F1 ---")
print(classification_report(yh_valid, valid_pred_hazard, digits=3))
print("\n--- Product per-class F1 ---")
print(classification_report(yp_valid, valid_pred_product, digits=3))

# ─────────────────────────────────────────────
# 7. Σύγκριση με v2
# ─────────────────────────────────────────────
try:
    with open('scores.json', 'r') as f:
        all_scores = json.load(f)
except FileNotFoundError:
    all_scores = {}

all_scores['v3'] = scores_v3

print("\n" + "="*55)
print("  MODEL COMPARISON")
print("="*55)
for version, s in all_scores.items():
    print(f"  {version}: F1_haz={s['f1_hazard']:.4f} | "
          f"F1_prod={s['f1_product']:.4f} | Score={s['score']:.4f}")
print("="*55)

with open('scores.json', 'w') as f:
    json.dump(all_scores, f, indent=2)

# ─────────────────────────────────────────────
# 8. Submission
# ─────────────────────────────────────────────
submission = pd.DataFrame({
    'id':               test['id'],
    'hazard-category':  test_pred_hazard,
    'product-category': test_pred_product
})
submission.to_csv('submission_v3.csv', index=False)
print("\nSaved → submission_v3.csv")
print("Scores saved → scores.json")
print("\nDone! Επόμενο βήμα: pipeline_v4.py (Fine-tuned BERT)")