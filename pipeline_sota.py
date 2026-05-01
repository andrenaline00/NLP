"""
SOTA Pipeline for Food Hazard Detection
Methodological Improvement: SBERT Embeddings + Neural MLP + Focal Loss
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import KFold

# ────────────────────────────────────────────────────────────────
# 1. Focal Loss Definition for Extreme Class Imbalance
# ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal loss: dynamically scales cross entropy based on prediction confidence.
        Hard examples get higher weights.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha  # Tensor of class weights

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ────────────────────────────────────────────────────────────────
# 2. Setup & Configuration
# ────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = torch.device('cpu')  # Optimized for CPU
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 1e-3
HIDDEN_DIM = 256
CV_FOLDS = 5

print("="*60)
print("SOTA Pipeline: Neural MLP + Focal Loss")
print("="*60)

train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")
test  = pd.read_csv("test.csv")

for df in [train, valid, test]:
    df['text'] = df['text'].fillna('')

hazard_classes = sorted(train['hazard-category'].unique())
product_classes = sorted(train['product-category'].unique())
haz2idx = {c: i for i, c in enumerate(hazard_classes)}
prod2idx = {c: i for i, c in enumerate(product_classes)}
num_hazard = len(hazard_classes)
num_product = len(product_classes)

train_haz_labels = torch.tensor([haz2idx[c] for c in train['hazard-category']], dtype=torch.long)
valid_haz_labels = torch.tensor([haz2idx[c] for c in valid['hazard-category']], dtype=torch.long)
train_prod_labels = torch.tensor([prod2idx[c] for c in train['product-category']], dtype=torch.long)
valid_prod_labels = torch.tensor([prod2idx[c] for c in valid['product-category']], dtype=torch.long)

# ────────────────────────────────────────────────────────────────
# 3. Embedding Extraction / Loading
# ────────────────────────────────────────────────────────────────
EMB_FILES = ["train_emb.npy", "valid_emb.npy", "test_emb.npy"]

if all(os.path.exists(f) for f in EMB_FILES):
    print("Loading cached SBERT embeddings...")
    train_emb = np.load("train_emb.npy")
    valid_emb = np.load("valid_emb.npy")
    test_emb  = np.load("test_emb.npy")
else:
    print(f"Encoding texts with '{MODEL_NAME}' (no cache found)...")
    encoder = SentenceTransformer(MODEL_NAME)
    train_emb = encoder.encode(train['text'].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
    valid_emb = encoder.encode(valid['text'].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
    test_emb  = encoder.encode(test['text'].tolist(),  batch_size=BATCH_SIZE, show_progress_bar=True)
    np.save("train_emb.npy", train_emb)
    np.save("valid_emb.npy", valid_emb)
    np.save("test_emb.npy", test_emb)

train_emb = torch.tensor(train_emb, dtype=torch.float32)
valid_emb = torch.tensor(valid_emb, dtype=torch.float32)
test_emb  = torch.tensor(test_emb, dtype=torch.float32)

def get_class_weights(labels, num_classes):
    counts = np.bincount(labels.numpy(), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

# ────────────────────────────────────────────────────────────────
# 4. Neural MLP Classifier
# ────────────────────────────────────────────────────────────────
class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_model(model, X_train, y_train, X_valid, y_valid, num_classes, prefix=""):
    class_weights = get_class_weights(y_train, num_classes)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_loss = float('inf')
    best_state = None
    wait = 0
    
    for epoch in range(1, EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_valid)
            val_loss = criterion(val_out, y_valid).item()
            preds = val_out.argmax(dim=1)
            val_f1 = f1_score(y_valid.numpy(), preds.numpy(), average='macro')
            
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                if prefix: print(f"[{prefix}] Early stopping at epoch {epoch}")
                break
                
    model.load_state_dict(best_state)
    return model

# ────────────────────────────────────────────────────────────────
# 5. Stage 1: Hazard Pipeline
# ────────────────────────────────────────────────────────────────
print("\n[Stage 1] Training Hazard Network with Focal Loss...")
haz_net = ClassifierMLP(train_emb.shape[1], HIDDEN_DIM, num_hazard)
haz_net = train_model(haz_net, train_emb, train_haz_labels, valid_emb, valid_haz_labels, num_hazard, "Hazard")

with torch.no_grad():
    valid_pred_haz = haz_net(valid_emb).argmax(dim=1)
    test_pred_haz = haz_net(test_emb).argmax(dim=1)

print("CV 5-Fold on Train set to prevent leakage for Stage 2...")
train_cv_haz_preds = torch.zeros(len(train), dtype=torch.long)
kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_emb)):
    f_model = ClassifierMLP(train_emb.shape[1], HIDDEN_DIM, num_hazard)
    f_model = train_model(f_model, train_emb[tr_idx], train_haz_labels[tr_idx],
                          train_emb[val_idx], train_haz_labels[val_idx], num_hazard)
    with torch.no_grad():
        train_cv_haz_preds[val_idx] = f_model(train_emb[val_idx]).argmax(dim=1)

# ────────────────────────────────────────────────────────────────
# 6. Stage 2: Product Pipeline
# ────────────────────────────────────────────────────────────────
print("\n[Stage 2] Building features and training Product Network...")
def build_features(emb, haz_preds, num_classes):
    ohe = torch.zeros(len(haz_preds), num_classes)
    ohe.scatter_(1, haz_preds.unsqueeze(1), 1.0)
    ohe = ohe * 5.0 # attention scaling
    return torch.cat([emb, ohe], dim=1)

train_prod_X = build_features(train_emb, train_cv_haz_preds, num_hazard)
valid_prod_X = build_features(valid_emb, valid_pred_haz, num_hazard)
test_prod_X = build_features(test_emb, test_pred_haz, num_hazard)

prod_net = ClassifierMLP(train_prod_X.shape[1], HIDDEN_DIM, num_product)
prod_net = train_model(prod_net, train_prod_X, train_prod_labels, valid_prod_X, valid_prod_labels, num_product, "Product")

with torch.no_grad():
    valid_pred_prod = prod_net(valid_prod_X).argmax(dim=1)
    test_pred_prod = prod_net(test_prod_X).argmax(dim=1)

# ────────────────────────────────────────────────────────────────
# 7. Final Evaluation & Metric
# ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
haz_f1 = f1_score(valid_haz_labels.numpy(), valid_pred_haz.numpy(), average='macro')
print(f"SOTA Hazard Macro-F1 (Focal Loss): {haz_f1:.4f}")

correct_mask = (valid_haz_labels == valid_pred_haz).numpy()
if correct_mask.sum() > 0:
    prod_f1 = f1_score(
        valid_prod_labels.numpy()[correct_mask],
        valid_pred_prod.numpy()[correct_mask],
        average='macro'
    )
else:
    prod_f1 = 0.0

print(f"SOTA Product Macro-F1 (where hazard correct): {prod_f1:.4f}")

official_score = (haz_f1 + prod_f1) / 2
print(f"\n>>> SOTA Official SemEval Score : {official_score:.4f} <<<")
print("="*50)

submission = pd.DataFrame({
    'id': test['id'],
    'hazard-category': [list(idx2haz.values())[i] for i in test_pred_haz.numpy()],
    'product-category': [list(idx2prod.values())[i] for i in test_pred_prod.numpy()],
})
submission.to_csv('submission_sota.csv', index=False)
print("Saved SOTA predictions to -> submission_sota.csv")
