# 🛡️ Two-Stage Cascade Intrusion Detection System (IDS)

> **Master's Thesis Project** — Network intrusion detection using a two-stage machine learning cascade (XGBoost + LightGBM) on the UNSW-NB15 / CICFlowMeter dataset.

---

## 📖 What Does This Project Do?

This project builds a system that **automatically detects cyberattacks** in network traffic. Think of it like a smart security camera for computer networks — it watches all the data flowing through and flags anything suspicious.

It works in **two stages**, like airport security:

| Stage | Model | Question It Answers | Analogy |
|-------|-------|---------------------|---------|
| **Stage 1** | XGBoost | "Is this traffic normal or an attack?" | Metal detector at the door |
| **Stage 2** | LightGBM | "What *type* of attack is it?" | Detailed bag search for flagged passengers |

---

## 🧩 Pipeline Overview

```
Raw CSV Data
    │
    ▼
[1] Clean & Preprocess ──► Remove bad values, encode labels
    │
    ▼
[2] Scale & Select Features ──► Normalize numbers, pick top 50 features
    │
    ▼
[3] Stage 1: Binary Classifier (XGBoost)
    │         "Benign or Attack?"
    │
    ├── Benign ──► Final label: Benign
    │
    └── Attack ──► Goes to Stage 2
                      │
                      ▼
               [4] Stage 2: Multi-class Classifier (LightGBM)
                      │    "What type of attack?"
                      ▼
                   Final label: DoS / Exploits / Fuzzers / etc.
```

---

## 📊 Results

### A) Full 10-Class Classification

| Metric | Value |
|--------|-------|
| Accuracy | 99.64% |
| Precision | 0.5950 |
| Recall | 0.7527 |
| F1-Score | 0.6361 |

### B) Binary Classification (Benign vs Attack)

| Metric | Value |
|--------|-------|
| Accuracy | 98.52% |
| Precision | 0.8157 |
| Recall | 0.9895 |
| F1-Score | 0.8823 |

### Per-Class Breakdown (10-class)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Analysis | 0.1044 | 0.6494 | 0.1799 |
| Backdoor | 0.6486 | 0.5333 | 0.5854 |
| Benign | 0.9998 | 0.9849 | 0.9923 |
| DoS | 0.5399 | 0.5145 | 0.5269 |
| Exploits | 0.7106 | 0.8486 | 0.7735 |
| Fuzzers | 0.4104 | 0.9580 | 0.5746 |
| Generic | 0.7985 | 0.9018 | 0.8470 |
| Reconnaissance | 0.8086 | 0.8975 | 0.8508 |
| Shellcode | 0.4179 | 0.7690 | 0.5415 |
| Worms | 0.5111 | 0.4694 | 0.4894 |

---

## 🔧 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
```

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm
```

---

## 🚀 How to Run

1. Place your dataset CSV file and update the path in `ids_pipeline.py`
2. Run:

```bash
python ids_pipeline.py
```

3. The script will output:
   - Classification metrics printed to the console
   - `confusion_matrix_full.png` — 10-class confusion matrix heatmap
   - `confusion_matrix_binary.png` — binary confusion matrix heatmap

---

## 📁 Project Structure

```
farouk-thesis-ids-code/
├── README.md              # This file — full explanation
├── ids_pipeline.py        # Main pipeline script
└── *.png                  # Generated confusion matrix plots (after running)
```

---

## 📚 Code Explanation (Beginner-Friendly)

### Part 1 — Loading Data
Opens the CSV dataset containing network traffic records. Each row represents one network "flow" (a conversation between two computers). Columns that are just identifiers (IP addresses, timestamps) are removed since they don't help detect patterns.

### Part 2 — Preprocessing
- Replaces broken values (infinity, missing) with reasonable defaults
- Converts text labels ("Benign", "DoS") into numbers the model can understand
- Creates two label versions: **multi-class** (10 types) and **binary** (Benign=0, Attack=1)

### Part 3 — Train/Test Split
Splits data into 80% for training and 20% for testing. The model never sees the test data during learning — this proves it can generalize to new, unseen traffic.

### Part 4 — Scaling & Feature Selection
- **Scaling:** Makes all numbers comparable (so a column with millions doesn't dominate one with decimals)
- **Feature Selection:** Picks the 50 most informative features out of all available columns using mutual information

### Part 5 — Stage 1 (XGBoost Binary Classifier)
- Balances the data (680k Benign vs 28k Attacks → reduces Benign to 400k)
- Trains XGBoost with 300 decision trees
- Flags anything with ≥50% attack probability

### Part 6 — Stage 2 (LightGBM Multi-class Classifier)
- Only receives flows flagged as "Attack" by Stage 1
- Includes 20k Benign samples so it can correct Stage 1 mistakes
- Balances rare attack types using SMOTE (creates synthetic examples)
- Trains LightGBM with 600 trees to classify the specific attack type

### Part 7 — Evaluation
Computes TP/TN/FP/FN for every class and calculates Accuracy, Precision, Recall, and F1-Score using the thesis formulas. Generates confusion matrix heatmaps.

---

## 📝 License

This project is part of a Master's thesis. Feel free to reference with proper attribution.
