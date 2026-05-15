"""
pipeline_v4.py — Fine-Tuned DistilBERT + 2-Stage Neural Classifier
Food Hazard Detection (SemEval-2025 Task 9, ST1)
NLP 053 - CSE UOI 2026

Αναβάθμιση από v3:
  - Το BERT ΔΕΝ είναι frozen — εκπαιδεύεται κι αυτό
  - Neural classifier head (όχι LinearSVC)
  - Class weights για imbalance
  - Early stopping (σταματάμε πριν overfit)
  - Αποθηκεύει το καλύτερο μοντέλο αυτόματα
  - Συγκρίνει score με v2, v3

Requirements:
  pip install transformers torch scikit-learn pandas numpy
"""

# ============================================================
# ΒΗΜΑ 0: IMPORTS
# ============================================================
# torch       → PyTorch, η βιβλιοθήκη για neural networks
# transformers → HuggingFace, δίνει έτοιμα BERT μοντέλα
# Dataset     → PyTorch class για οργάνωση δεδομένων
# DataLoader  → φορτώνει δεδομένα σε batches

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# ── Ρυθμίσεις ────────────────────────────────────────────────
# DEVICE: αν έχεις GPU (cuda) χρησιμοποιείται αυτόματα
# Αλλιώς τρέχει στη CPU (πιο αργά)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"  # 66M params, γρήγορο
BATCH_SIZE = 16     # πόσα παραδείγματα επεξεργαζόμαστε μαζί
MAX_LEN    = 128    # max tokens ανά κείμενο (τα δικά μας ~88 chars → OK)
EPOCHS     = 5      # πόσες φορές βλέπουμε ΟΛΑ τα δεδομένα
LR_BERT    = 2e-5   # learning rate για BERT (πολύ μικρό — δεν "ξεχνάει")
LR_HEAD    = 1e-4   # learning rate για classifier (μεγαλύτερο — μαθαίνει γρήγορα)
PATIENCE   = 2      # early stopping: σταμάτα αν δεν βελτιώνεται 2 epochs

print(f"Device: {DEVICE}")
print(f"Model:  {MODEL_NAME}")


# ============================================================
# ΒΗΜΑ 1: ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ
# ============================================================
print("\n[1/7] Loading data...")
train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")
test  = pd.read_csv("test.csv")
print(f"  Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")

def build_text(df):
    """
    Συνδυάζει title + text.
    Title x3 γιατί είναι πιο πυκνό σε πληροφορία.
    Παράδειγμα:
      title = "Recall due to peanuts"
      text  = "The company announces..."
      → "Recall due to peanuts Recall due to peanuts Recall due to peanuts The company..."
    """
    title = df['title'].fillna('') if 'title' in df.columns else pd.Series([''] * len(df))
    text  = df['text'].fillna('')
    return ((title + ' ') * 3 + text).tolist()

train_texts = build_text(train)
valid_texts = build_text(valid)
test_texts  = build_text(test)


# ============================================================
# ΒΗΜΑ 2: LABEL ENCODING
# ============================================================
# Τα μοντέλα θέλουν αριθμούς, όχι strings
# LabelEncoder: "allergens"→0, "biological"→1, ...
# Μετά στο submission κάνουμε inverse_transform: 0→"allergens"

print("\n[2/7] Encoding labels...")
le_hazard  = LabelEncoder()
le_product = LabelEncoder()

le_hazard.fit(train['hazard-category'])
le_product.fit(train['product-category'])

yh_train = le_hazard.transform(train['hazard-category'])
yp_train = le_product.transform(train['product-category'])
yh_valid = le_hazard.transform(valid['hazard-category'])
yp_valid = le_product.transform(valid['product-category'])

NUM_HAZARD  = len(le_hazard.classes_)   # 10
NUM_PRODUCT = len(le_product.classes_)  # 22
print(f"  Hazard classes: {NUM_HAZARD}  → {list(le_hazard.classes_)}")
print(f"  Product classes: {NUM_PRODUCT}")


# ============================================================
# ΒΗΜΑ 3: DATASET CLASS
# ============================================================
# Το PyTorch Dataset είναι ένα "container" για τα δεδομένα.
# Πρέπει να έχει:
#   __len__     → πόσα παραδείγματα έχουμε
#   __getitem__ → δώσε μου το παράδειγμα [i]

print("\n[3/7] Preparing dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class FoodRecallDataset(Dataset):
    def __init__(self, texts, hazard_labels=None, product_labels=None, max_len=MAX_LEN):
        """
        texts:          λίστα string κειμένων
        hazard_labels:  numpy array με int labels (None για test)
        product_labels: numpy array με int labels (None για test)
        max_len:        max αριθμός tokens
        """
        self.texts          = texts
        self.hazard_labels  = hazard_labels
        self.product_labels = product_labels
        self.max_len        = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Tokenization:
          "product recalled due to peanuts"
          → input_ids:      [101, 2529, 9166, 102, ...]
                              CLS  prod  recal  SEP  PAD
          → attention_mask: [1,   1,    1,    1,   0  ...]
                              real tokens         padding
        """
        encoded = tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",   # γεμίζει με [PAD] μέχρι max_len
            truncation=True,        # κόβει αν είναι πολύ μακρύ
            return_tensors="pt"     # επιστρέφει PyTorch tensors
        )

        item = {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        # Αν έχουμε labels (train/valid), τα προσθέτουμε
        if self.hazard_labels is not None:
            item["hazard_label"]  = torch.tensor(self.hazard_labels[idx],  dtype=torch.long)
            item["product_label"] = torch.tensor(self.product_labels[idx], dtype=torch.long)

        return item

# Φτιάχνουμε τα datasets
train_dataset = FoodRecallDataset(train_texts, yh_train, yp_train)
valid_dataset = FoodRecallDataset(valid_texts, yh_valid, yp_valid)
test_dataset  = FoodRecallDataset(test_texts)  # χωρίς labels

# DataLoader: φορτώνει δεδομένα σε batches
# shuffle=True στο train: ανακατεύει κάθε epoch (αποφεύγει patterns)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train batches: {len(train_loader)}")
print(f"  Valid batches: {len(valid_loader)}")


# ============================================================
# ΒΗΜΑ 4: ΜΟΝΤΕΛΟ — 2-Stage Fine-tuned BERT
# ============================================================
# Αρχιτεκτονική:
#
#   Text → DistilBERT → [CLS] (768)
#                           │
#                     ┌─────┴──────┐
#                     │            │
#               Hazard Head    │
#               Dropout(0.3)   │
#               Linear(768→10) │ Stage 1
#                     │            │
#               hazard_pred (one-hot)
#                           │
#               concat [CLS + hazard_onehot] (778)
#                           │
#                    Product Head
#                    Dropout(0.3)
#                    Linear(778→256)
#                    ReLU
#                    Dropout(0.2)
#                    Linear(256→22)     Stage 2

print("\n[4/7] Building model...")

class TwoStageBERT(nn.Module):
    def __init__(self, num_hazard, num_product):
        super().__init__()

        # ── BERT backbone ─────────────────────────────────────
        # Αυτό εκπαιδεύεται κι αυτό (fine-tuning)
        # Διαφορά από v3: εδώ ΔΕΝ κάνουμε bert_model.eval()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        # ── Stage 1: Hazard Head ──────────────────────────────
        # Dropout: τυχαία "σβήνει" neurons κατά το training
        #   → αποτρέπει overfitting
        # Linear: πλήρως συνδεδεμένο layer
        #   768 εισόδους (BERT) → num_hazard εξόδους
        self.hazard_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768, num_hazard)
        )

        # ── Stage 2: Product Head ─────────────────────────────
        # Είσοδος: 768 (BERT) + num_hazard (one-hot hazard)
        # Έξοδος: num_product κλάσεις
        # ReLU: non-linear activation → μαθαίνει πολύπλοκα patterns
        self.product_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768 + num_hazard, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_product)
        )

        self.num_hazard = num_hazard

    def forward(self, input_ids, attention_mask, hazard_labels=None):
        """
        Forward pass: κείμενο → προβλέψεις

        hazard_labels:
          - Training:  δίνουμε TRUE labels (teacher forcing)
                       → σταθερό σήμα για το Stage 2
          - Inference: None → χρησιμοποιούμε predicted labels
                       → ρεαλιστικό scenario
        """
        # ── BERT: κείμενο → embeddings ───────────────────────
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # last_hidden_state shape: [batch, tokens, 768]
        # Παίρνουμε μόνο το [CLS] token (index 0)
        # → shape: [batch, 768]
        cls_emb = bert_out.last_hidden_state[:, 0, :]

        # ── Stage 1: Hazard logits ────────────────────────────
        # Logits: "ακατέργαστες" τιμές πριν το softmax
        # shape: [batch, num_hazard]
        hazard_logits = self.hazard_head(cls_emb)

        # ── One-hot encoding για Stage 2 ─────────────────────
        if hazard_labels is not None:
            # Teacher forcing: χρησιμοποιούμε τα ΣΩΣΤΑ labels
            # Παράδειγμα: label=2 → [0, 0, 1, 0, ..., 0]
            hazard_onehot = torch.zeros(
                cls_emb.size(0), self.num_hazard, device=cls_emb.device
            )
            hazard_onehot.scatter_(1, hazard_labels.unsqueeze(1), 1.0)
        else:
            # Inference: χρησιμοποιούμε τα PREDICTED labels
            hazard_pred = hazard_logits.argmax(dim=1)
            hazard_onehot = torch.zeros(
                cls_emb.size(0), self.num_hazard, device=cls_emb.device
            )
            hazard_onehot.scatter_(1, hazard_pred.unsqueeze(1), 1.0)

        # ── Stage 2: Product logits ───────────────────────────
        # Συνδυάζουμε BERT embedding + hazard one-hot
        # [batch, 768] + [batch, num_hazard] → [batch, 768+num_hazard]
        combined       = torch.cat([cls_emb, hazard_onehot], dim=1)
        product_logits = self.product_head(combined)

        return hazard_logits, product_logits


model = TwoStageBERT(NUM_HAZARD, NUM_PRODUCT).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")


# ============================================================
# ΒΗΜΑ 5: LOSS & OPTIMIZER
# ============================================================
# Class weights: δίνουμε μεγαλύτερο βάρος στις σπάνιες κλάσεις
# Παράδειγμα:
#   allergens: 1800 examples → weight = 0.3 (χαμηλό)
#   migration:    5 examples → weight = 8.2 (υψηλό)

print("\n[5/7] Setting up training...")

def compute_class_weights(labels, num_classes):
    """
    weight[i] = total / (num_classes × count[i])
    Σπάνιες κλάσεις → μεγάλο weight → μεγαλύτερη ποινή αν λάθος
    """
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    weights = len(labels) / (num_classes * counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float).to(DEVICE)

haz_weights  = compute_class_weights(yh_train, NUM_HAZARD)
prod_weights = compute_class_weights(yp_train, NUM_PRODUCT)

# CrossEntropyLoss: μετράει πόσο λάθος είναι η πρόβλεψη
# reduction='mean': παίρνει μέσο όρο loss όλου του batch
hazard_criterion  = nn.CrossEntropyLoss(weight=haz_weights)
product_criterion = nn.CrossEntropyLoss(weight=prod_weights)

# AdamW Optimizer με διαφορετικά learning rates
# BERT layers:     lr=2e-5 (μικρό → δεν "ξεχνάει" pre-training)
# Classifier head: lr=1e-4 (μεγαλύτερο → μαθαίνει γρήγορα)
optimizer = torch.optim.AdamW([
    {"params": model.bert.parameters(),         "lr": LR_BERT},
    {"params": model.hazard_head.parameters(),  "lr": LR_HEAD},
    {"params": model.product_head.parameters(), "lr": LR_HEAD},
])

# Learning Rate Scheduler: μειώνει το LR σταδιακά
# OneCycleLR: αυξάνει → μειώνει κατά τη διάρκεια training
total_steps = len(train_loader) * EPOCHS
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[LR_BERT, LR_HEAD, LR_HEAD],
    total_steps=total_steps
)


# ============================================================
# ΒΗΜΑ 6: ΣΥΝΑΡΤΗΣΕΙΣ TRAINING & EVALUATION
# ============================================================

def st1_score(yh_true, yh_pred, yp_true, yp_pred, label=""):
    """
    Επίσημο ST1 metric:
    Score = (F1_hazard + F1_product_when_hazard_correct) / 2
    """
    f1_haz  = f1_score(yh_true, yh_pred, average='macro', zero_division=0)
    mask    = (np.array(yh_true) == np.array(yh_pred))
    f1_prod = f1_score(
        np.array(yp_true)[mask],
        np.array(yp_pred)[mask],
        average='macro', zero_division=0
    ) if mask.sum() > 0 else 0.0
    score = (f1_haz + f1_prod) / 2
    print(f"\n  {'─'*45}")
    if label: print(f"  {label}")
    print(f"  Hazard correct: {mask.sum()}/{len(mask)} ({mask.mean()*100:.1f}%)")
    print(f"  F1 Hazard:      {f1_haz:.4f}")
    print(f"  F1 Product:     {f1_prod:.4f}")
    print(f"  ST1 Score:      {score:.4f}")
    print(f"  {'─'*45}")
    return score, f1_haz, f1_prod


def train_one_epoch(model, loader, optimizer, scheduler):
    """
    Training loop για 1 epoch:
    1. Forward pass: κείμενο → πρόβλεψη
    2. Loss: πόσο λάθος είμαστε
    3. Backward: υπολογισμός gradients
    4. Step: ενημέρωση weights
    5. Scheduler: ενημέρωση learning rate
    """
    model.train()  # ενεργοποιεί Dropout, BatchNorm
    total_loss = 0.0
    total_haz_loss  = 0.0
    total_prod_loss = 0.0

    for step, batch in enumerate(loader):
        # Μεταφορά στο GPU/CPU
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        hazard_labels  = batch["hazard_label"].to(DEVICE)
        product_labels = batch["product_label"].to(DEVICE)

        # Forward pass (με teacher forcing → δίνουμε true hazard labels)
        hazard_logits, product_logits = model(
            input_ids, attention_mask,
            hazard_labels=hazard_labels
        )

        # Loss για κάθε task
        loss_h = hazard_criterion(hazard_logits, hazard_labels)
        loss_p = product_criterion(product_logits, product_labels)

        # Total loss: άθροισμα (μπορεί να προσθέσεις weights π.χ. 2*loss_h + loss_p)
        loss = loss_h + loss_p

        # Backward pass: υπολογίζει gradients (πόσο αλλάζει κάθε weight)
        optimizer.zero_grad()  # μηδενίζει παλιά gradients
        loss.backward()        # backpropagation

        # Gradient clipping: αποτρέπει πολύ μεγάλα gradients → αστάθεια
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()   # ενημέρωση weights
        scheduler.step()   # ενημέρωση learning rate

        total_loss      += loss.item()
        total_haz_loss  += loss_h.item()
        total_prod_loss += loss_p.item()

        # Progress κάθε 50 steps
        if (step + 1) % 50 == 0:
            print(f"    Step {step+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} "
                  f"(haz: {loss_h.item():.4f}, prod: {loss_p.item():.4f})")

    n = len(loader)
    return total_loss/n, total_haz_loss/n, total_prod_loss/n


def evaluate(model, loader, has_labels=True):
    """
    Evaluation loop:
    - model.eval(): απενεργοποιεί Dropout
    - torch.no_grad(): δεν υπολογίζει gradients (πιο γρήγορο)
    """
    model.eval()
    all_haz_true,  all_haz_pred  = [], []
    all_prod_true, all_prod_pred = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # Inference: χωρίς teacher forcing
            hazard_logits, product_logits = model(input_ids, attention_mask)

            # argmax: παίρνει την κλάση με τη μεγαλύτερη πιθανότητα
            haz_pred  = hazard_logits.argmax(dim=1).cpu().numpy()
            prod_pred = product_logits.argmax(dim=1).cpu().numpy()

            all_haz_pred.extend(haz_pred)
            all_prod_pred.extend(prod_pred)

            if has_labels:
                all_haz_true.extend(batch["hazard_label"].numpy())
                all_prod_true.extend(batch["product_label"].numpy())

    return (
        np.array(all_haz_true),  np.array(all_haz_pred),
        np.array(all_prod_true), np.array(all_prod_pred)
    )


# ============================================================
# ΒΗΜΑ 7: TRAINING LOOP — FULL
# ============================================================
print("\n[6/7] Training...")
print(f"  Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | Patience: {PATIENCE}")
print(f"  LR BERT: {LR_BERT} | LR Head: {LR_HEAD}\n")

best_score    = 0.0
patience_ctr  = 0
history       = []

for epoch in range(EPOCHS):
    print(f"\n{'='*55}")
    print(f"  EPOCH {epoch+1}/{EPOCHS}")
    print(f"{'='*55}")

    # ── Training ──────────────────────────────────────────────
    train_loss, haz_loss, prod_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler
    )
    print(f"\n  Train Loss: {train_loss:.4f} "
          f"(hazard: {haz_loss:.4f}, product: {prod_loss:.4f})")

    # ── Validation ────────────────────────────────────────────
    yh_true, yh_pred, yp_true, yp_pred = evaluate(model, valid_loader)
    score, f1_haz, f1_prod = st1_score(
        yh_true, yh_pred, yp_true, yp_pred,
        label=f"Epoch {epoch+1} Validation"
    )

    history.append({
        "epoch": epoch+1,
        "train_loss": round(train_loss, 4),
        "f1_hazard":  round(f1_haz, 4),
        "f1_product": round(f1_prod, 4),
        "score":      round(score, 4)
    })

    # ── Early Stopping & Best Model ───────────────────────────
    # Αποθηκεύουμε το μοντέλο μόνο αν βελτιώθηκε το score
    if score > best_score:
        best_score   = score
        patience_ctr = 0
        torch.save(model.state_dict(), "best_model_v4.pt")
        print(f"\n  ✓ New best! Score={best_score:.4f} → saved best_model_v4.pt")
    else:
        patience_ctr += 1
        print(f"\n  No improvement ({patience_ctr}/{PATIENCE})")
        if patience_ctr >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

# Εκτύπωση ιστορικού
print(f"\n{'='*55}")
print("  TRAINING HISTORY")
print(f"{'='*55}")
for h in history:
    print(f"  Epoch {h['epoch']}: loss={h['train_loss']:.4f} | "
          f"F1_haz={h['f1_hazard']:.4f} | "
          f"F1_prod={h['f1_product']:.4f} | "
          f"Score={h['score']:.4f}")
print(f"\n  Best Score: {best_score:.4f}")


# ============================================================
# ΒΗΜΑ 8: ΤΕΛΙΚΗ ΑΞΙΟΛΟΓΗΣΗ & SUBMISSION
# ============================================================
print("\n[7/7] Generating submission...")

# Φορτώνουμε το καλύτερο μοντέλο
model.load_state_dict(torch.load("best_model_v4.pt"))
model.eval()

# Τελική αξιολόγηση στο validation
yh_true, yh_pred, yp_true, yp_pred = evaluate(model, valid_loader)
final_score, f1_haz, f1_prod = st1_score(
    yh_true, yh_pred, yp_true, yp_pred,
    label="FINAL — Best Model on Validation"
)

print("\n--- Hazard per-class F1 ---")
print(classification_report(
    le_hazard.inverse_transform(yh_true),
    le_hazard.inverse_transform(yh_pred),
    digits=3
))

print("\n--- Product per-class F1 ---")
print(classification_report(
    le_product.inverse_transform(yp_true),
    le_product.inverse_transform(yp_pred),
    digits=3
))

# Predictions στο test set (χωρίς labels)
_, yh_test_pred, _, yp_test_pred = evaluate(model, test_loader, has_labels=False)

submission = pd.DataFrame({
    "id":               test["id"],
    "hazard-category":  le_hazard.inverse_transform(yh_test_pred),
    "product-category": le_product.inverse_transform(yp_test_pred)
})
submission.to_csv("submission_v4.csv", index=False)
print("\nSaved → submission_v4.csv")

# ── Σύγκριση με v2, v3 ────────────────────────────────────────
scores_v4 = {
    "f1_hazard":  round(f1_haz, 4),
    "f1_product": round(f1_prod, 4),
    "score":      round(final_score, 4)
}
try:
    with open("scores.json", "r") as f:
        all_scores = json.load(f)
except FileNotFoundError:
    all_scores = {}

all_scores["v4"] = scores_v4

print(f"\n{'='*55}")
print("  MODEL COMPARISON")
print(f"{'='*55}")
labels = {"v2": "TF-IDF + LinearSVC",
          "v3": "Frozen BERT + LinearSVC",
          "v4": "Fine-tuned BERT (neural)"}
for v, s in all_scores.items():
    tag = labels.get(v, v)
    print(f"  {v} ({tag})")
    print(f"     F1_hazard={s['f1_hazard']:.4f} | "
          f"F1_product={s['f1_product']:.4f} | "
          f"Score={s['score']:.4f}")
print(f"{'='*55}")

with open("scores.json", "w") as f:
    json.dump(all_scores, f, indent=2)
print("\nScores saved → scores.json")
print("\nDone!")