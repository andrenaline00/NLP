import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.model_selection import cross_val_predict

print("Loading data...")
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")
test = pd.read_csv("test.csv")

# Ensure text has no NaNs
train['text'] = train['text'].fillna('')
valid['text'] = valid['text'].fillna('')
test['text'] = test['text'].fillna('')

print("Vectorizing text with Bigrams...")
# Adding Bigrams (1,2) and expanding features to 20,000 for better representation
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(train['text'])
valid_tfidf = tfidf.transform(valid['text'])
test_tfidf = tfidf.transform(test['text'])

print("Training Hazard Model (Stage 1) using LinearSVC...")
# Switching to Linear Support Vector Machines (often much stronger for text!)
hazard_model = LinearSVC(class_weight='balanced', max_iter=2000)
hazard_model.fit(train_tfidf, train['hazard-category'])
valid_pred_hazard = hazard_model.predict(valid_tfidf)

print("Cross-validating Hazard Model on Training Set to avoid leakage...")
train_pred_hazard = cross_val_predict(
    LinearSVC(class_weight='balanced', max_iter=2000),
    train_tfidf,
    train['hazard-category'],
    cv=5,
    n_jobs=-1
)

print("Building 2-stage features for Product Model...")
ohe = OneHotEncoder(handle_unknown='ignore')
train_hazard_features = ohe.fit_transform(train_pred_hazard.reshape(-1, 1))
valid_hazard_features = ohe.transform(valid_pred_hazard.reshape(-1, 1))

# TRICK: Multiply the 1-hot features by 5 so the LinearSVC gives them more "attention"
# (otherwise they get drowned out by the 20000 TFIDF features)
train_hazard_features = train_hazard_features.multiply(5.0)
valid_hazard_features = valid_hazard_features.multiply(5.0)

train_product_features = hstack([train_tfidf, train_hazard_features])
valid_product_features = hstack([valid_tfidf, valid_hazard_features])

print("Training Product Model (Stage 2) using LinearSVC...")
product_model = LinearSVC(class_weight='balanced', max_iter=2000)
product_model.fit(train_product_features, train['product-category'])

print("Predicting on Validation set...")
valid_pred_product = product_model.predict(valid_product_features)

print(f"\n--- Validation Results ---")
hazard_macro_f1 = f1_score(valid['hazard-category'], valid_pred_hazard, average='macro')
print(f"Hazard Macro-F1: {hazard_macro_f1:.4f}")

correct_hazard_mask = (valid['hazard-category'] == valid_pred_hazard)
if correct_hazard_mask.sum() > 0:
    product_macro_f1_correct_hazard = f1_score(
        valid.loc[correct_hazard_mask, 'product-category'], 
        valid_pred_product[correct_hazard_mask], 
        average='macro'
    )
else:
    product_macro_f1_correct_hazard = 0.0

print(f"Product Macro-F1 (only where hazard is correct): {product_macro_f1_correct_hazard:.4f}")

official_score = (hazard_macro_f1 + product_macro_f1_correct_hazard) / 2
print(f"Official SemEval Score: {official_score:.4f}")

print("\n--- Kaggle Submission ---")
print("Predicting on Test set...")
test_pred_hazard = hazard_model.predict(test_tfidf)
test_hazard_features = ohe.transform(test_pred_hazard.reshape(-1, 1)).multiply(5.0)
test_product_features = hstack([test_tfidf, test_hazard_features])
test_pred_product = product_model.predict(test_product_features)

submission = pd.DataFrame({
    'id': test['id'],
    'hazard-category': test_pred_hazard,
    'product-category': test_pred_product
})

submission.to_csv('submission.csv', index=False)
print("Saved predictions to submission.csv! Upload.")
