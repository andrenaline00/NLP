"""
Part 2 — Neural Network Pipeline for SemEval-2025 Task 9: Food Hazard Detection
Architecture: Frozen BERT (bert-base-uncased) → MLP classification heads
             2-Stage Hierarchical: Hazard → Product
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# ─── Configuration ───────────────────────────────────────────────────
BERT_MODEL = "bert-base-uncased"
MAX_LEN = 256          # Token limit (covers ~95% of texts)
BATCH_SIZE = 32        # For BERT embedding extraction
MLP_EPOCHS = 50        # Max epochs for MLP training
PATIENCE = 5           # Early stopping patience
LR = 1e-3              # Learning rate for MLP heads
HIDDEN_DIM = 256       # Hidden layer size in MLP
DEVICE = torch.device("cpu")
CV_FOLDS = 5           # Cross-validation folds for leak-free train predictions

# ─── 1. Load Data ────────────────────────────────────────────────────
print("=" * 60)
print("PART 2: Neural Network Pipeline (Frozen BERT + MLP)")
print("=" * 60)

print("\n[1/6] Loading data...")
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")
test = pd.read_csv("test.csv")

train['text'] = train['text'].fillna('')
valid['text'] = valid['text'].fillna('')
test['text'] = test['text'].fillna('')

# Build label encoders
hazard_classes = sorted(train['hazard-category'].unique())
product_classes = sorted(train['product-category'].unique())
haz2idx = {c: i for i, c in enumerate(hazard_classes)}
prod2idx = {c: i for i, c in enumerate(product_classes)}
idx2haz = {i: c for c, i in haz2idx.items()}
idx2prod = {i: c for c, i in prod2idx.items()}

num_hazard = len(hazard_classes)
num_product = len(product_classes)
print(f"   Hazard classes: {num_hazard}, Product classes: {num_product}")
print(f"   Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

# ─── 2. Extract BERT Embeddings ─────────────────────────────────────
print(f"\n[2/6] Extracting BERT embeddings ({BERT_MODEL}, max_len={MAX_LEN})...")
print("   Loading tokenizer and model (this may download ~440MB on first run)...")

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
bert_model = BertModel.from_pretrained(BERT_MODEL)
bert_model.eval()
bert_model.to(DEVICE)


def extract_embeddings(texts, desc=""):
    """Extract [CLS] embeddings from frozen BERT in batches."""
    all_embeddings = []
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_texts = texts[start:end]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = bert_model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        all_embeddings.append(cls_embeddings)
        print(f"   {desc} {end}/{total} texts processed", end="\r")

    print()
    return torch.cat(all_embeddings, dim=0)


train_emb = extract_embeddings(train['text'].tolist(), desc="Train:")
valid_emb = extract_embeddings(valid['text'].tolist(), desc="Valid:")
test_emb = extract_embeddings(test['text'].tolist(), desc="Test: ")

print(f"   Embedding shape: {train_emb.shape}")  # (N, 768)

# ─── 3. Define MLP Classifier ───────────────────────────────────────


class MLPClassifier(nn.Module):
    """Simple MLP with one hidden layer + dropout."""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def compute_class_weights(labels, num_classes):
    """Compute inverse-frequency weights for balanced loss."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_mlp(model, X_train, y_train, X_valid, y_valid, num_classes,
              epochs=MLP_EPOCHS, patience=PATIENCE, lr=LR, desc=""):
    """Train MLP with early stopping on validation loss."""

    class_weights = compute_class_weights(y_train.numpy(), num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    best_val_loss = float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_valid)
            val_loss = criterion(val_logits, y_valid).item()
            val_preds = val_logits.argmax(dim=1)
            val_f1 = f1_score(y_valid.numpy(), val_preds.numpy(), average='macro')

        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"   {desc} Epoch {epoch:3d} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Macro-F1: {val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"   {desc} Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


# ─── 4. Stage 1: Train Hazard Classifier ────────────────────────────
print(f"\n[3/6] Training Stage 1: Hazard Classifier (MLP {train_emb.shape[1]}→{HIDDEN_DIM}→{num_hazard})...")

train_hazard_labels = torch.tensor([haz2idx[c] for c in train['hazard-category']], dtype=torch.long)
valid_hazard_labels = torch.tensor([haz2idx[c] for c in valid['hazard-category']], dtype=torch.long)

hazard_model = MLPClassifier(train_emb.shape[1], HIDDEN_DIM, num_hazard)
hazard_model = train_mlp(
    hazard_model, train_emb, train_hazard_labels,
    valid_emb, valid_hazard_labels, num_hazard,
    desc="Hazard"
)

# Predict on valid & test
hazard_model.eval()
with torch.no_grad():
    valid_hazard_preds = hazard_model(valid_emb).argmax(dim=1)
    test_hazard_preds = hazard_model(test_emb).argmax(dim=1)

# ─── 5. Stage 2: Cross-val predictions to avoid leakage ─────────────
print(f"\n[4/6] Cross-validating Stage 1 on train set (CV={CV_FOLDS}) to avoid data leakage...")

train_hazard_cv_preds = torch.zeros(len(train), dtype=torch.long)
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_emb), 1):
    print(f"   Fold {fold}/{CV_FOLDS}...")
    fold_model = MLPClassifier(train_emb.shape[1], HIDDEN_DIM, num_hazard)

    fold_train_X = train_emb[tr_idx]
    fold_train_y = train_hazard_labels[tr_idx]
    fold_val_X = train_emb[val_idx]
    fold_val_y = train_hazard_labels[val_idx]

    fold_model = train_mlp(
        fold_model, fold_train_X, fold_train_y,
        fold_val_X, fold_val_y, num_hazard,
        desc=f"Fold{fold}"
    )

    fold_model.eval()
    with torch.no_grad():
        train_hazard_cv_preds[val_idx] = fold_model(fold_val_X).argmax(dim=1)

# ─── 6. Stage 2: Train Product Classifier ───────────────────────────
print(f"\n[5/6] Training Stage 2: Product Classifier...")

# Build augmented features: BERT embedding + one-hot hazard prediction
def make_stage2_features(embeddings, hazard_preds, num_hazard_classes):
    """Concatenate BERT embeddings with one-hot encoded hazard predictions."""
    one_hot = torch.zeros(len(hazard_preds), num_hazard_classes)
    one_hot.scatter_(1, hazard_preds.unsqueeze(1), 1.0)
    return torch.cat([embeddings, one_hot], dim=1)


train_product_features = make_stage2_features(train_emb, train_hazard_cv_preds, num_hazard)
valid_product_features = make_stage2_features(valid_emb, valid_hazard_preds, num_hazard)
test_product_features = make_stage2_features(test_emb, test_hazard_preds, num_hazard)

product_input_dim = train_product_features.shape[1]  # 768 + num_hazard
print(f"   Stage 2 input dim: {product_input_dim} (768 BERT + {num_hazard} hazard one-hot)")

train_product_labels = torch.tensor([prod2idx[c] for c in train['product-category']], dtype=torch.long)
valid_product_labels = torch.tensor([prod2idx[c] for c in valid['product-category']], dtype=torch.long)

product_model = MLPClassifier(product_input_dim, HIDDEN_DIM, num_product)
product_model = train_mlp(
    product_model, train_product_features, train_product_labels,
    valid_product_features, valid_product_labels, num_product,
    desc="Product"
)

# Predict on valid & test
product_model.eval()
with torch.no_grad():
    valid_product_preds = product_model(valid_product_features).argmax(dim=1)
    test_product_preds = product_model(test_product_features).argmax(dim=1)

# ─── 7. Evaluation ──────────────────────────────────────────────────
print(f"\n[6/6] Evaluation")
print("=" * 60)

# Convert predictions back to class names
valid_pred_hazard_names = [idx2haz[i.item()] for i in valid_hazard_preds]
valid_pred_product_names = [idx2prod[i.item()] for i in valid_product_preds]

hazard_macro_f1 = f1_score(valid['hazard-category'], valid_pred_hazard_names, average='macro')
print(f"Hazard Macro-F1: {hazard_macro_f1:.4f}")

correct_hazard_mask = (valid['hazard-category'].values == np.array(valid_pred_hazard_names))
if correct_hazard_mask.sum() > 0:
    product_macro_f1 = f1_score(
        valid.loc[correct_hazard_mask, 'product-category'],
        np.array(valid_pred_product_names)[correct_hazard_mask],
        average='macro'
    )
else:
    product_macro_f1 = 0.0

print(f"Product Macro-F1 (where hazard correct): {product_macro_f1:.4f}")

official_score = (hazard_macro_f1 + product_macro_f1) / 2
print(f"Official SemEval Score: {official_score:.4f}")
print("=" * 60)

# ─── 8. Kaggle Submission ───────────────────────────────────────────
print("\nGenerating Kaggle submission...")

test_pred_hazard_names = [idx2haz[i.item()] for i in test_hazard_preds]
test_pred_product_names = [idx2prod[i.item()] for i in test_product_preds]

submission = pd.DataFrame({
    'id': test['id'],
    'hazard-category': test_pred_hazard_names,
    'product-category': test_pred_product_names
})

submission.to_csv('submission_nn.csv', index=False)
print("Saved submission_nn.csv — upload to Kaggle!")
