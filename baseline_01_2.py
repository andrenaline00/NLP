# ============================================================
# ST1 - Food Hazard Text Classification - Starter Template
# ============================================================

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report




train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")
test_df  = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Valid shape:", valid_df.shape)
print("Test  shape:", test_df.shape)


print("\n--- Hazard Category Distribution (train) ---")
print(train_df["hazard-category"].value_counts())

print("\n--- Product Category Distribution (train) ---")
print(train_df["product-category"].value_counts())

print("\n--- Text Length Stats (train) ---")
print(train_df["text"].str.len().describe())


# ============================================================
# 3. PREPARE FEATURES & LABELS
# ============================================================

# Train on full train.csv
X_train   = train_df["text"]
yh_train  = train_df["hazard-category"]
yp_train  = train_df["product-category"]

# Validate on valid.csv (official split)
X_val     = valid_df["text"]
yh_val    = valid_df["hazard-category"]
yp_val    = valid_df["product-category"]

# Test (no labels)
X_test    = test_df["text"]


# ============================================================
# 4. CUSTOM SCORING FUNCTION
# ============================================================

def st1_score(yh_true, yh_pred, yp_true, yp_pred):
    # F1 on all hazard predictions
    f1_hazard = f1_score(yh_true, yh_pred, average="macro", zero_division=0)

    # F1 on product — only where hazard is correct
    mask = (np.array(yh_true) == np.array(yh_pred))
    if mask.sum() == 0:
        f1_product = 0.0
    else:
        f1_product = f1_score(
            np.array(yp_true)[mask],
            np.array(yp_pred)[mask],
            average="macro",
            zero_division=0
        )

    score = (f1_hazard + f1_product) / 2
    print(f"  F1 Hazard:   {f1_hazard:.4f}")
    print(f"  F1 Product:  {f1_product:.4f}")
    print(f"  ST1 Score:   {score:.4f}")
    return score


# ============================================================
# 5. BUILD MODELS (TF-IDF + Logistic Regression)
# ============================================================

# --- Hazard Model ---
hazard_model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
])

# --- Product Model ---
product_model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))
])


# ============================================================
# 6. TRAIN
# ============================================================

print("\nTraining hazard model...")
hazard_model.fit(X_train, yh_train)

print("Training product model...")
product_model.fit(X_train, yp_train)


# ============================================================
# 7. EVALUATE ON VALIDATION SET
# ============================================================

yh_pred_val = hazard_model.predict(X_val)
yp_pred_val = product_model.predict(X_val)

print("\n--- Validation Results ---")
st1_score(yh_val, yh_pred_val, yp_val, yp_pred_val)

print("\n--- Hazard Classification Report ---")
print(classification_report(yh_val, yh_pred_val, zero_division=0))

print("\n--- Product Classification Report ---")
print(classification_report(yp_val, yp_pred_val, zero_division=0))


# ============================================================
# 8. PREDICT ON TEST SET
# ============================================================

hazard_preds  = hazard_model.predict(X_test)
product_preds = product_model.predict(X_test)


# ============================================================
# 9. CREATE SUBMISSION FILE
# ============================================================

submission = pd.DataFrame({
    "id":               test_df["id"],
    "hazard-category":  hazard_preds,
    "product-category": product_preds
})

submission.to_csv("submission_a.csv", index=False)
print("\nsubmission.csv saved!")
print(submission.head())