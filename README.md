# Predicting Diabetes Risk from Health-Survey Indicators

A controlled comparison of classical machine learning and deep learning for
predicting diabetes / pre-diabetes from cheaply collectable survey indicators,
using the CDC BRFSS 2015 health-indicators dataset.

**Author:** Alliane Umutoniwase — African Leadership University, Machine Learning Summative

## Problem

Diabetes is rising fastest in low- and middle-income settings, where most cases
go undiagnosed and lab-based screening is scarce. This project asks how well
diabetes status can be predicted from survey indicators, and whether classical
ML or deep learning is better suited to the task — then looks at *who* the model
gets wrong, not just how often.

## Dataset

CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 health indicators,
UCI ML Repository dataset **891** (253,680 respondents, 21 features, binary
target). The notebook loads it programmatically:

```python
from ucimlrepo import fetch_ucirepo
data = fetch_ucirepo(id=891)
```

No manual download is needed. A Colab file-upload fallback is included in case
the fetch is unavailable.

## Repo structure
├── Summative.ipynb       # full pipeline: EDA → preprocessing → 9 experiments → error analysis

├── Summative_Report.pdf  # written report (3,500–5,000 words, IEEE refs)

└── README.md

## How to run

1. Open `Diabetes_Summative.ipynb` in Jupyter or Google Colab.
2. `pip install -r requirements.txt`
3. Run all cells top to bottom. The dataset downloads automatically; a fixed
   seed (42) makes the run reproducible.

## Methods

A single leak-free preprocessing pipeline (exact duplicates dropped before the
split, BMI capped, one ColumnTransformer fitted on train only) feeds nine
experiments through one evaluation harness:

- **Classical:** logistic regression (±balanced), random forest (default and
  tuned+balanced), histogram gradient boosting, RBF SVM on a stratified subsample.
- **Deep:** Sequential baseline, regularised Sequential (batch norm + dropout +
  class weights), and a two-branch Functional model — all on a `tf.data` pipeline
  with early stopping on validation ROC-AUC.

## Results

- All nine models cluster in a narrow ROC-AUC band (~0.78–0.82). Best deep
  network 0.818 and gradient boosting 0.817 are statistically tied.
- Class imbalance is a **threshold problem, not a capacity problem**: tuning the
  decision threshold lifted the network's recall from 0.17 to 0.61 with no change
  in ROC-AUC.
- **Equity finding:** recall on true cases falls steadily as income and education
  rise — a socio-economic disparity hidden by aggregate accuracy. The fix is
  group-aware thresholds, not a bigger model.

## Demo video

[[INSERT VIDEO LINK]](https://drive.google.com/file/d/1trtPlEiZ5K1PZY_gAGPhfAHuo4Y6L5PG/view?usp=drive_link)

## Report

Full write-up in `Diabetes_Summative_Report.pdf`.
