# ============================================================================
# Two-Stage Cascade Intrusion Detection System (IDS)
# Master's Thesis — Farouk
#
# Stage 1: XGBoost  → Binary classification (Benign vs Attack)
# Stage 2: LightGBM → Multi-class classification (specific attack type)
# ============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# PART 1: IMPORTING LIBRARIES
# ──────────────────────────────────────────────────────────────────────────────
# pandas         → Works with tables (like Excel spreadsheets)
# numpy          → Math operations on numbers and arrays
# matplotlib     → Drawing charts and graphs
# seaborn        → Makes prettier charts (built on top of matplotlib)
# sklearn        → Machine learning tools (splitting data, scaling, encoding)
# imblearn       → Tools to fix imbalanced datasets (SMOTE, undersampling)
# xgboost        → Powerful gradient boosting model (used in Stage 1)
# lightgbm       → Fast gradient boosting model (used in Stage 2)
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ──────────────────────────────────────────────────────────────────────────────
# PART 2: LOAD AND CLEAN DATA
# ──────────────────────────────────────────────────────────────────────────────
# What happens here:
#   1. Open the CSV file containing network traffic data
#      - Each ROW = one network "flow" (a conversation between two computers)
#      - Each COLUMN = a measurement (e.g., packet size, duration, bytes sent)
#   2. Clean up column names (remove extra spaces, replace spaces with _)
#   3. Find the label column (tells us if traffic is Benign, DoS, Exploits, etc.)
#   4. Drop columns that are just identifiers (IP addresses, timestamps)
#      — they don't help detect attacks, they're like serial numbers
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("CICFlowMeter.csv")
df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False)
print("Original Shape:", df.shape)

# Find the label column automatically
label_candidates = [col for col in df.columns if 'label' in col.lower()]
if not label_candidates:
    raise ValueError("No label column found!")
label_col = label_candidates[0]
print("Label distribution:\n", df[label_col].value_counts())

# Remove identifier columns — they don't help detect attacks
drop_cols = ['Flow_ID', 'Src_IP', 'Dst_IP', 'Timestamp']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# ──────────────────────────────────────────────────────────────────────────────
# PART 3: PREPROCESSING (PREPARING THE DATA)
# ──────────────────────────────────────────────────────────────────────────────
# What happens here:
#   1. Replace broken values (infinity) with NaN, then fill NaN with the
#      column's median (middle value) — like giving a missing test the class average
#   2. Convert all columns to numbers (some might be text by accident)
#   3. Encode labels: convert text labels to numbers the model can understand
#      - Multi-class: "Benign"=2, "DoS"=3, etc. (one number per class)
#      - Binary: "Benign"=0, any attack=1 (just two categories)
# ──────────────────────────────────────────────────────────────────────────────

df.replace([np.inf, -np.inf], np.nan, inplace=True)
y_raw    = df[label_col]
X        = df.drop(label_col, axis=1)
X        = X.fillna(X.median(numeric_only=True))
X        = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# LabelEncoder turns text labels into numbers
le       = LabelEncoder()
y_multi  = le.fit_transform(y_raw)          # e.g., "DoS" → 3, "Benign" → 2
y_binary = (y_raw != 'Benign').astype(int).values  # 0 = Benign, 1 = Attack
benign_idx = int(le.transform(['Benign'])[0])
print("\nLabel mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print(f"Benign class index: {benign_idx}")

# ──────────────────────────────────────────────────────────────────────────────
# PART 4: REMOVE RARE CLASSES + TRAIN/TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────
# What happens here:
#   1. Remove classes with fewer than 5 samples (too rare to learn from)
#   2. Split all data into 80% training and 20% testing
#      - Training: the model learns patterns from this data
#      - Testing: we check how well it learned on data it has NEVER seen
#      - stratify=y_multi ensures each class keeps its proportion in both sets
# ──────────────────────────────────────────────────────────────────────────────



df_temp = X.copy()
df_temp['__label__'] = y_multi
counts  = df_temp['__label__'].value_counts()
mask    = df_temp['__label__'].isin(counts[counts >= 5].index)
X        = X[mask].reset_index(drop=True)
y_multi  = y_multi[mask.values]
y_binary = y_binary[mask.values]
print(f"\nTotal samples: {len(y_multi):,}")

(X_train, X_test,
 y_train_multi, y_test_multi,
 y_train_bin,   y_test_bin) = train_test_split(
    X, y_multi, y_binary,
    test_size=0.2, random_state=42, stratify=y_multi
)

# ──────────────────────────────────────────────────────────────────────────────
# PART 5: SCALE + FEATURE SELECTION
# ──────────────────────────────────────────────────────────────────────────────
# What happens here:
#   1. SCALING (RobustScaler): makes all numbers comparable
#      - Without scaling: a column with values 0-1,000,000 would dominate
#        a column with values 0-1, even if the small column is more important
#      - RobustScaler uses median & IQR, so outliers don't distort the scaling
#
#   2. FEATURE SELECTION (SelectKBest with mutual information):
#      - Out of ALL columns, picks the 50 most useful ones
#      - "Mutual information" measures how much knowing a feature's value
#        helps you predict the label — higher = more useful
#      - Uses a random sample of 50k rows for speed
# ──────────────────────────────────────────────────────────────────────────────

print("\n[1/5] Scaling and selecting features...")
scaler    = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Use a 50k sample for faster feature selection
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train_s), size=min(50_000, len(X_train_s)), replace=False)
selector  = SelectKBest(score_func=mutual_info_classif, k=min(50, X_train_s.shape[1]))
selector.fit(X_train_s[idx], y_train_multi[idx])
X_train_f = selector.transform(X_train_s)
X_test_f  = selector.transform(X_test_s)
print(f"  Features selected: {X_train_f.shape[1]}")

# ──────────────────────────────────────────────────────────────────────────────
# PART 6: STAGE 1 — BINARY CLASSIFIER (XGBoost)
#         Question: "Is this traffic Benign or an Attack?"
# ──────────────────────────────────────────────────────────────────────────────
# What happens here:
#   1. BALANCE THE DATA:
#      - Problem: ~680k Benign vs ~28k Attacks — the model would just always
#        guess "Benign" and be right 96% of the time (but miss all attacks!)
#      - Solution: Undersample Benign to 400k so it's less lopsided
#
#   2. TRAIN XGBoost:
#      - XGBoost builds 300 "decision trees" — each tree is like a flowchart
#        of yes/no questions ("Is packet size > 500? Is duration > 10s?")
#      - Each new tree focuses on the mistakes of the previous trees
#      - Together, the 300 trees vote on the final answer
#
#   3. PREDICT:
#      - The model outputs a probability (0.0 to 1.0)
#      - If probability >= 0.50 → flag as Attack → send to Stage 2
#      - If probability < 0.50 → label as Benign → done
#
# Analogy: A security guard at the front door. They don't care WHAT kind of
#          threat — they just decide: "safe person" or "suspicious person"
# ──────────────────────────────────────────────────────────────────────────────

print("\n[2/5] Stage 1 — Benign vs Attack...")
rus_bin = RandomUnderSampler(sampling_strategy={0: 400_000}, random_state=42)
X_bin, y_bin = rus_bin.fit_resample(X_train_f, y_train_bin)
print(f"  Training: Benign={int((y_bin==0).sum()):,}  Attack={int((y_bin==1).sum()):,}")

stage1 = XGBClassifier(
    n_estimators=300,        # Build 300 decision trees
    max_depth=6,             # Each tree can have up to 6 levels of questions
    learning_rate=0.1,       # How much each new tree corrects previous mistakes
    subsample=0.8,           # Each tree sees 80% of the data (reduces overfitting)
    colsample_bytree=0.8,    # Each tree sees 80% of features (reduces overfitting)
    scale_pos_weight=1.5,    # Give 1.5x more importance to attacks (minority class)
    tree_method='hist',      # Fast histogram-based algorithm
    device='cpu',            # Use CPU (change to 'cuda' for GPU)
    random_state=42,         # Reproducible results
    n_jobs=-1,               # Use all CPU cores
    eval_metric='logloss',   # Optimization metric
    verbosity=0              # Don't print training progress
)
stage1.fit(X_bin, y_bin)

# Predict on test data
S1_THRESHOLD = 0.50  # If >= 50% sure it's an attack, flag it
s1_test_proba = stage1.predict_proba(X_test_f)[:, 1]  # probability of being Attack
s1_pred       = (s1_test_proba >= S1_THRESHOLD).astype(int)
s1_fp = int(((s1_pred == 1) & (y_test_bin == 0)).sum())  # Benign wrongly flagged
s1_fn = int(((s1_pred == 0) & (y_test_bin == 1)).sum())  # Attacks that slipped through
print(f"  Stage 1 threshold : {S1_THRESHOLD}")
print(f"  Stage 1 accuracy  : {(s1_pred == y_test_bin).mean():.4f}")
print(f"  False positives   : {s1_fp:,}  (Benign sent to Stage 2 by mistake)")
print(f"  False negatives   : {s1_fn:,}  (Attacks missed — predicted as Benign)")

# ──────────────────────────────────────────────────────────────────────────────
# PART 7: STAGE 2 — MULTI-CLASS CLASSIFIER (LightGBM)
#         Question: "What specific TYPE of attack is this?"
# ──────────────────────────────────────────────────────────────────────────────
# What happens here:
#   1. PREPARE TRAINING DATA:
#      - Take all attack samples from the training set
#      - Also include 20k Benign samples — WHY? Because Stage 1 makes mistakes
#        and sends some Benign flows here. Stage 2 can correct those mistakes
#        by recognizing "actually, this looks Benign"
#
#   2. BALANCE THE DATA:
#      - Undersample: cap big attack classes at 15k (e.g., Exploits has too many)
#      - SMOTE: generate synthetic examples for tiny classes up to 5k
#        → SMOTE creates new fake samples by mixing existing ones
#        → Like creating new student profiles by averaging existing ones
#
#   3. TRAIN LightGBM:
#      - Similar to XGBoost but faster and uses a different splitting strategy
#      - 600 trees, deeper (max_depth=10) for more complex patterns
#
# Analogy: A specialist detective. The security guard (Stage 1) says
#          "this person is suspicious." The detective (Stage 2) figures out
#          WHAT they're doing — pickpocketing, trespassing, hacking, etc.
# ──────────────────────────────────────────────────────────────────────────────

print("\n[3/5] Stage 2 — Attack type classifier (with Benign class)...")

# Separate attack and benign training data
atk_mask   = (y_train_bin == 1)
ben_mask   = (y_train_bin == 0)

X_atk      = X_train_f[atk_mask]
y_atk_orig = y_train_multi[atk_mask]

# Sample 20k Benign rows for Stage 2 (so it can correct Stage 1 mistakes)
ben_idx  = np.where(ben_mask)[0]
rng2     = np.random.default_rng(0)
ben_sample_idx = rng2.choice(ben_idx, size=min(20_000, len(ben_idx)), replace=False)
X_ben_s2 = X_train_f[ben_sample_idx]
y_ben_s2 = y_train_multi[ben_sample_idx]   # all = benign_idx

# Combine attacks + Benign sample
X_s2_all = np.vstack([X_atk, X_ben_s2])
y_s2_all = np.concatenate([y_atk_orig, y_ben_s2])

# Re-encode labels so they go 0, 1, 2, ... (required by LightGBM)
le_s2     = LabelEncoder()
y_s2_enc  = le_s2.fit_transform(y_s2_all)
benign_s2_idx = int(le_s2.transform([benign_idx])[0])

s2_counts = np.bincount(y_s2_enc)
print(f"  Stage 2 classes: {le_s2.classes_} (mapped to {list(le.classes_[le_s2.classes_])})")
print(f"  Distribution before balancing: {s2_counts}")

# UNDERSAMPLE: cap large attack classes at 15k
LARGE_CAP    = 15_000
SMOTE_TARGET =  5_000

under_s = {
    cls: LARGE_CAP
    for cls, cnt in enumerate(s2_counts)
    if cnt > LARGE_CAP and cls != benign_s2_idx
}
if under_s:
    rus2 = RandomUnderSampler(sampling_strategy=under_s, random_state=42)
    X_s2_all, y_s2_enc = rus2.fit_resample(X_s2_all, y_s2_enc)

# SMOTE: create synthetic samples for small classes up to 5k
counts_now = np.bincount(y_s2_enc)
smote_s = {
    cls: SMOTE_TARGET
    for cls, cnt in enumerate(counts_now)
    if 0 < cnt < SMOTE_TARGET and cls != benign_s2_idx
}
if smote_s:
    min_cls = min(counts_now[c] for c in smote_s)
    k       = min(5, max(1, min_cls - 1))
    smote   = SMOTE(random_state=42, k_neighbors=k, sampling_strategy=smote_s)
    X_s2_all, y_s2_enc = smote.fit_resample(X_s2_all, y_s2_enc)

print(f"  After balancing: {np.bincount(y_s2_enc)}")

stage2 = LGBMClassifier(
    n_estimators=600,        # Build 600 decision trees
    learning_rate=0.05,      # Smaller steps = more careful learning
    max_depth=10,            # Deeper trees = more complex patterns
    num_leaves=63,           # Max leaves per tree (controls complexity)
    min_child_samples=5,     # Minimum samples in a leaf node
    reg_lambda=1.0,          # Regularization to prevent overfitting
    random_state=42,         # Reproducible results
    n_jobs=-1,               # Use all CPU cores
    verbose=-1               # Silent mode
)
stage2.fit(X_s2_all, y_s2_enc)
print("  Stage 2 trained.")

# ──────────────────────────────────────────────────────────────────────────────
# PART 8: CASCADE PREDICTION (COMBINING BOTH STAGES)
# ──────────────────────────────────────────────────────────────────────────────
# What happens here:
#   1. Start by labeling EVERYTHING as "Benign" (default assumption)
#   2. Stage 1 flags suspicious flows (probability >= 0.50)
#   3. ONLY those flagged flows go to Stage 2
#   4. Stage 2 classifies the exact attack type (or says "actually Benign")
#   5. Update the final predictions
#
# Analogy: Everyone walks through the metal detector (Stage 1).
#          Only those who beep get pulled aside for a bag search (Stage 2).
# ──────────────────────────────────────────────────────────────────────────────

print("\n[4/5] Cascade predictions...")
attack_mask = (s1_test_proba >= S1_THRESHOLD)
final_pred  = np.full(len(X_test_f), benign_idx, dtype=int)

if attack_mask.sum() > 0:
    s2_enc_pred    = stage2.predict(X_test_f[attack_mask])
    s2_orig_labels = le_s2.inverse_transform(s2_enc_pred)  # back to global label ints
    final_pred[attack_mask] = s2_orig_labels

print(f"  Flows sent to Stage 2 : {attack_mask.sum():,}")
print(f"  Flows kept as Benign  : {(~attack_mask).sum():,}")

# ──────────────────────────────────────────────────────────────────────────────
# PART 9: EVALUATION — MEASURING HOW GOOD THE MODEL IS
# ──────────────────────────────────────────────────────────────────────────────
# What happens here:
#   We count how many times the model was right or wrong for EACH class:
#
#   TP (True Positive)  = Correctly detected an attack           ✅
#   TN (True Negative)  = Correctly said "Benign"                ✅
#   FP (False Positive) = Said "Attack" but was actually Benign  ❌ (false alarm)
#   FN (False Negative) = Said "Benign" but was actually Attack  ❌ (missed threat!)
#
#   Then we calculate four metrics:
#   ┌───────────┬──────────────────────────────────────────────────────────────┐
#   │ Metric    │ What it means                                               │
#   ├───────────┼──────────────────────────────────────────────────────────────┤
#   │ Accuracy  │ % of ALL predictions that were correct (overall score)      │
#   │ Precision │ When it says "attack," how often is it right?               │
#   │           │ → Like: if the fire alarm rings, is there really a fire?    │
#   │ Recall    │ Of all real attacks, how many did it catch?                 │
#   │           │ → Like: of all real fires, how many did the alarm detect?   │
#   │ F1-Score  │ Balance between Precision and Recall (harmonic mean)        │
#   └───────────┴──────────────────────────────────────────────────────────────┘
#
#   We evaluate twice:
#   A) Full 10-class: every attack type individually
#   B) Binary: merge all attacks into one "Attack" class vs "Benign"
# ──────────────────────────────────────────────────────────────────────────────

print("\n[5/5] Evaluating...")

# Build confusion matrix manually (no sklearn dependency for this)
def build_confusion_matrix(y_true, y_pred, labels):
    """
    Creates an N×N matrix where:
      - Row i = actual class i
      - Column j = predicted class j
      - Cell [i][j] = number of times class i was predicted as class j
    """
    n = len(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm_mat = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in label_to_idx and pred in label_to_idx:
            cm_mat[label_to_idx[true], label_to_idx[pred]] += 1
    return cm_mat

# Compute TP/TN/FP/FN and metrics from confusion matrix
def compute_metrics(cm_mat, class_names):
    """
    For each class i:
      TP[i] = correctly predicted as class i        (diagonal cell)
      FP[i] = other classes wrongly predicted as i   (column sum minus diagonal)
      FN[i] = class i wrongly predicted as other     (row sum minus diagonal)
      TN[i] = everything else that was correctly NOT predicted as i
    """
    n     = len(class_names)
    total = float(cm_mat.sum())
    TP    = np.zeros(n, dtype=float)
    TN    = np.zeros(n, dtype=float)
    FP    = np.zeros(n, dtype=float)
    FN    = np.zeros(n, dtype=float)

    for i in range(n):
        TP[i] = cm_mat[i, i]
        FP[i] = cm_mat[:, i].sum() - cm_mat[i, i]
        FN[i] = cm_mat[i, :].sum() - cm_mat[i, i]
        TN[i] = total - TP[i] - FP[i] - FN[i]

    # Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    acc_pc  = (TP + TN) / (TP + TN + FP + FN)

    # Precision = TP / (TP + FP) — "of all predicted as X, how many are really X?"
    prec_pc = np.zeros(n, dtype=float)
    for i in range(n):
        prec_pc[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0.0

    # Recall = TP / (TP + FN) — "of all actual X, how many did we catch?"
    rec_pc  = np.zeros(n, dtype=float)
    for i in range(n):
        rec_pc[i]  = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0.0

    # F1-Score = 2 * Precision * Recall / (Precision + Recall)
    f1_pc   = np.zeros(n, dtype=float)
    for i in range(n):
        f1_pc[i] = (2 * prec_pc[i] * rec_pc[i]) / (prec_pc[i] + rec_pc[i]) \
                    if (prec_pc[i] + rec_pc[i]) > 0 else 0.0

    return TP, TN, FP, FN, acc_pc, prec_pc, rec_pc, f1_pc

def print_results(title, cm_mat, class_names, TP, TN, FP, FN,
                  acc_pc, prec_pc, rec_pc, f1_pc):
    """Prints a formatted results table to the console."""
    accuracy  = acc_pc.mean()
    precision = prec_pc.mean()
    recall    = rec_pc.mean()
    f1        = f1_pc.mean()
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"  Accuracy  = (TP+TN)/(TP+TN+FP+FN) = {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision = TP/(TP+FP)             = {precision:.4f}")
    print(f"  Recall    = TP/(TP+FN)             = {recall:.4f}")
    print(f"  F1-Score  = 2*P*R/(P+R)            = {f1:.4f}")
    print(f"\n{'─'*80}")
    print(f"  {'Class':<20} {'TP':>7} {'TN':>8} {'FP':>7} {'FN':>7} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"{'─'*80}")
    for i, name in enumerate(class_names):
        print(f"  {name:<20} {int(TP[i]):>7} {int(TN[i]):>8} {int(FP[i]):>7} "
              f"{int(FN[i]):>7} {prec_pc[i]:>7.4f} {rec_pc[i]:>7.4f} {f1_pc[i]:>7.4f}")
    print(f"{'─'*80}")
    print(f"  {'Average':<20} {'':>7} {'':>8} {'':>7} {'':>7} "
          f"{precision:>7.4f} {recall:>7.4f} {f1:>7.4f}")
    return accuracy, precision, recall, f1

# ═══════════════════════════════════════════════════════════════════════════════
# A) FULL 10-CLASS EVALUATION
#    Every class (Benign, DoS, Exploits, ...) treated individually
# ═══════════════════════════════════════════════════════════════════════════════

labels_full     = np.unique(y_test_multi)
class_names_full = [le.classes_[i] for i in labels_full]
cm_full         = build_confusion_matrix(y_test_multi, final_pred, labels_full)

TP_f, TN_f, FP_f, FN_f, acc_f, prec_f, rec_f, f1_f = compute_metrics(
    cm_full, class_names_full
)
acc_A, prec_A, rec_A, f1_A = print_results(
    "A) Full 10-class  (thesis formula — all classes)",
    cm_full, class_names_full,
    TP_f, TN_f, FP_f, FN_f, acc_f, prec_f, rec_f, f1_f
)

# Save confusion matrix heatmap — full 10-class
plt.figure(figsize=(12, 10))
sns.heatmap(cm_full, annot=True, cmap='Blues', fmt='d',
            xticklabels=class_names_full, yticklabels=class_names_full)
plt.title("Confusion Matrix — Full 10-class")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("confusion_matrix_full.png", dpi=150); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# B) BINARY EVALUATION — Benign vs Attack (all attack types merged)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n\nMerging all attack classes into 'Attack' for binary evaluation...")

y_test_bin2 = np.where(y_test_multi == benign_idx, 0, 1)
pred_bin2   = np.where(final_pred   == benign_idx, 0, 1)

labels_bin      = [0, 1]
class_names_bin = ['Benign', 'Attack']
cm_bin = build_confusion_matrix(y_test_bin2, pred_bin2, labels_bin)

TP_b, TN_b, FP_b, FN_b, acc_b, prec_b, rec_b, f1_b = compute_metrics(
    cm_bin, class_names_bin
)
acc_B, prec_B, rec_B, f1_B = print_results(
    "B) Binary  (Benign vs Attack)",
    cm_bin, class_names_bin,
    TP_b, TN_b, FP_b, FN_b, acc_b, prec_b, rec_b, f1_b
)

# Save confusion matrix heatmap — binary
plt.figure(figsize=(6, 5))
sns.heatmap(cm_bin, annot=True, cmap='Blues', fmt='d',
            xticklabels=class_names_bin, yticklabels=class_names_bin)
plt.title("Confusion Matrix — Benign vs Attack")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("confusion_matrix_binary.png", dpi=150); plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*55}")
print(f"  SUMMARY")
print(f"{'='*55}")
print(f"  {'Metric':<12} {'A) Full':>12} {'B) Merged':>12}")
print(f"{'─'*40}")
print(f"  {'Accuracy':<12} {acc_A:>12.4f} {acc_B:>12.4f}")
print(f"  {'Precision':<12} {prec_A:>12.4f} {prec_B:>12.4f}")
print(f"  {'Recall':<12} {rec_A:>12.4f} {rec_B:>12.4f}")
print(f"  {'F1-Score':<12} {f1_A:>12.4f} {f1_B:>12.4f}")
print(f"{'─'*40}")

print("Pipeline complete!")
