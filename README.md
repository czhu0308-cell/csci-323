# Wine Quality Modelling — CSCI323 Group Project

**University of Wollongong | CSCI323 Modern Artificial Intelligence 

---

## Project Overview

This project investigates the use of machine learning models to predict wine quality based on physicochemical properties. Using the UCI Wine Quality Dataset, three models were implemented, tuned, and compared: Ordinal Logistic Regression, Random Forest Classifier, and Multi-Layer Perceptron (MLP) Neural Network.

---

## Group Members

| Name | UOW ID |
|---|---|
| Chen Zhu | 8750786 |
| Kiefer Russell Sulijanto | 9088787 |
| Zhang YiNing | 8751195 |
| Gao HaoRan | 8750038 |
| Yeo Zhi Cheng | 9182810 |

**Tutor:** Ms Cher Lim

---

## Dataset

- **Source:** [UCI Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Files:** `winequality-red.csv`, `winequality-white.csv`
- **Combined size:** 6,497 samples (5,320 after duplicate removal)
- **Features:** 11 physicochemical input features + wine type
- **Target:** Wine quality score (ordinal, range 3–9)

---

## Models Implemented

| Model | Library | Tuned Parameter |
|---|---|---|
| Ordinal Logistic Regression | `mord` (LogisticIT) | alpha (regularization) — best: 5.0 |
| Random Forest Classifier | `sklearn` | n_estimators — best: 50 |
| MLP Neural Network | `sklearn` (MLPClassifier) | alpha (L2 regularization) — best: 1.0 |

---

## Results (Test Set)

| Model | Accuracy | F1 (weighted) | MAE | Within 1 Grade |
|---|---|---|---|---|
| Ordinal Regression | 53.85% | 0.5036 | 0.5169 | 95.21% |
| **Random Forest** | **55.17%** | **0.5167** | **0.4981** | **95.58%** |
| MLP Neural Network | 53.01% | 0.4853 | 0.5226 | 95.30% |

**Best overall model: Random Forest Classifier** — highest accuracy, F1, and within-1-grade rate, with lowest MAE.

### Key findings
- All three models struggled with extreme quality scores (3, 4, 8, 9) due to severe class imbalance
- RF feature importance: alcohol (0.2294), volatile acidity (0.1166), density (0.1099) were the most influential features
- RF training accuracy reduced to 67.89% (from 100%) after applying max_depth=8, min_samples_leaf=5, min_samples_split=20

---

## Notebook Structure

```
wine_quality_modelling.ipynb
├── Data Preprocessing
│   ├── Load & merge red/white wine datasets
│   ├── Missing value & duplicate checks
│   ├── Encode wine_type (red=0, white=1)
│   ├── Target variable distribution analysis
│   ├── Stratified train/validation/test split (50/30/20)
│   └── Feature standardization (StandardScaler)
├── Method 1: Ordinal Logistic Regression
│   ├── Train LogisticIT model (mord library)
│   ├── Hyperparameter tuning (alpha) on validation set
│   └── Evaluation: Accuracy, F1, MAE, Confusion Matrix, Error Distribution
├── Method 2: Random Forest Classifier
│   ├── Train RandomForestClassifier (max_depth=8, min_samples_leaf=5, min_samples_split=20)
│   ├── Hyperparameter tuning (n_estimators) on validation set
│   └── Evaluation: Accuracy, F1, MAE, Confusion Matrix, Error Distribution
├── Method 3: MLP Neural Network
│   ├── Train MLPClassifier (hidden layers: 100, 50; ReLU activation; early stopping)
│   ├── Hyperparameter tuning (alpha) on validation set
│   └── Evaluation: Accuracy, F1, MAE, Confusion Matrix, Error Distribution
├── Feature Importance (Random Forest)
└── Overall Model Comparison
```

---

## How to Run

1. Open `wine_quality_modelling.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload the two dataset files to the Colab session:
   - `winequality-red.csv`
   - `winequality-white.csv`
3. Click **Runtime → Run all**

> The notebook will automatically install the `mord` library if not already available.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
mord
```

---

## Repository Structure

```
csci-323/
├── wine_quality_modelling.ipynb   # Main notebook with all 3 models
├── winequality-red.csv            # Red wine dataset
├── winequality-white.csv          # White wine dataset
└── README.md
```
