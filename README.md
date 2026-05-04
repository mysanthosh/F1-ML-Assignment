# F1 Strategy Performance Analytics
## Predicting Pit Stops Using Lap-by-Lap Race Data

 Course Assignment 1 | Supervised Machine Learning   
 Supervisor:  Prof. Raja Hashim Ali


## Project Overview

This project applies supervised machine learning to Formula 1 lap-level race data to predict pit-stop events — specifically whether a pit stop will occur on a given lap, which model best achieves this prediction, and how race strategy features influence outcomes. The study addresses seven research questions covering baseline benchmarking, model selection, preprocessing effects, feature importance, metric-based rankings, cross-validation robustness, and practical strategy utility.

## Dataset

| Attribute | Details |
|-----------|---------|
|  Name  | F1 Strategy Dataset — Pit Stop Prediction |
|  Source  | [Kaggle — F1 Strategy Dataset: Pit Stop Prediction](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction) |
|  Target (primary)  | `pit_stop` (binary classification: 1 = pit stop on this lap, 0 = no pit stop) |
|  Coverage  | Multi-season Formula 1 lap-by-lap records |
|  Key Features  | Lap number, tyre compound, stint length, lap time, track position, gap to leader, race phase |


## Repository Structure

f1-pitstop-ml-assignment/
│
├── RQ1_baseline_models.ipynb           # Logistic Regression, Decision Tree, Naive Bayes baselines
├── RQ2_best_model.ipynb                # Model selection: RF, XGBoost, SVM, tuned comparison
├── RQ3_preprocessing_effects.ipynb     # Scaling, encoding, SMOTE, missing value impact
├── RQ4_feature_importance.ipynb        # SHAP values, RF importances, permutation importance
├── RQ5_metric_rankings.ipynb           # Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC rankings
├── RQ6_xgboost_crossval.ipynb          # XGBoost k-fold cross-validation & stability analysis
├── RQ7_practical_utility.ipynb         # Threshold tuning, confusion analysis, strategy simulation
│
├── figures/                            # All generated figures (PDF)
├── tables/                             # All generated tables (CSV)
│
├── requirements.txt                    # Required Python libraries
└── README.md                           # This file


## Research Questions

| # | Research Question | Notebook |
|---|-------------------|----------|
| RQ1 | How do baseline models perform on pit-stop prediction? | `RQ1_baseline_models.ipynb` |
| RQ2 | Which model achieves the best predictive performance? | `RQ2_best_model.ipynb` |
| RQ3 | How does preprocessing affect model performance? | `RQ3_preprocessing_effects.ipynb` |
| RQ4 | Which features are most important for prediction? | `RQ4_feature_importance.ipynb` |
| RQ5 | How do model rankings change across different metrics? | `RQ5_metric_rankings.ipynb` |
| RQ6 | How robust is XGBoost under cross-validation? | `RQ6_xgboost_crossval.ipynb` |
| RQ7 | Is the final model practically useful for race strategy decisions? | `RQ7_practical_utility.ipynb` |


## Models Used

- Logistic Regression
- Decision Tree
- Naive Bayes
- Random Forest
- XGBoost
- Support Vector Machine (SVM)


## How to Run on Kaggle

1. Go to [Kaggle](https://www.kaggle.com) and log in.
2. Navigate to the dataset: [F1 Strategy Dataset — Pit Stop Prediction](https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction)
3. Click  "New Notebook"  on the dataset page.
4. Upload the `.ipynb` files from this repository one at a time.
5. The dataset path will be: `/kaggle/input/f1-strategy-dataset-pit-stop-prediction/`
6. Run all cells in order using  Run All .
7. Figures are saved to `figures/` and tables to `tables/` in the Kaggle working directory.

> Note: Each notebook is self-contained. Run them independently in any order.


## Generated Outputs

### Figures (saved as PDF) — 14 files

| File | Description | RQ |
|------|-------------|----|
| `Figure_1_1_class_distribution.png`  | Class imbalance bar chart (pit stop vs no pit stop) | RQ1 |
| `Figure_1_2_baseline_ROC_curves.png` | ROC-AUC curves for all 3 baseline classifiers | RQ1 |
| `Figure_2_1_model_comparison.png`    | Bar chart of all models by F1 and ROC-AUC | RQ2 |
| `Figure_2_2_best_model_ROC.png`      | ROC curve of the best-performing model | RQ2 |
| `Figure_3_1_preprocessing_impact.png`| Performance bars before vs after scaling, encoding, SMOTE | RQ3 |
| `Figure_3_2_smote_pr_curve.png`      | PR-AUC curves with and without SMOTE | RQ3 |
| `Figure_4_1_rf_feature_importance.png` | Random Forest feature importances (top 15 features) | RQ4 |
| `Figure_4_2_shap_summary.png`          | SHAP summary plot for XGBoost feature contributions | RQ4 |
| `Figure_5_1_metric_heatmap.png`        | Heatmap of model rankings across all evaluation metrics | RQ5 |
| `Figure_5_2_metric_rank_shifts.png`    | Rank shift chart showing metric-driven reorderings | RQ5 |
| `Figure_6_1_crossval_boxplot.png`    | XGBoost k-fold F1 and ROC-AUC score distribution boxplots | RQ6 |
| `Figure_6_2_crossval_learning_curve.png` | Learning curves showing training vs validation F1 | RQ6 |
| `Figure_7_1_threshold_analysis.png`      | Precision-Recall trade-off across decision thresholds | RQ7 |
| `Figure_7_2_confusion_matrix_final.png`  | Confusion matrix of final model at optimal threshold | RQ7 |

### Tables (saved as CSV) — 7 files

| File | Description | RQ |
|------|-------------|----|
| `Table_1_1_baseline_performance.csv` | Accuracy, Precision, Recall, F1, ROC-AUC for baseline models | RQ1 |
| `Table_2_1_full_model_comparison.csv` | All models ranked by F1, ROC-AUC, and PR-AUC | RQ2 |
| `Table_3_1_preprocessing_ablation.csv` | Metric changes per preprocessing step (ablation study) | RQ3 |
| `Table_4_1_feature_importance_ranked.csv` | Top features ranked by RF importance and SHAP value | RQ4 |
| `Table_5_1_metric_rank_table.csv` | Model ranking positions under Accuracy, F1, ROC-AUC, PR-AUC | RQ5 |
| `Table_6_1_crossval_summary.csv` | XGBoost mean, std, min, max F1 and ROC-AUC across folds | RQ6 |
| `Table_7_1_strategy_utility_report.csv` | False positive/negative costs and threshold selection summary | RQ7 |


## Key Findings

- Baseline models (Logistic Regression, Decision Tree, Naive Bayes) establish a weak performance floor due to severe class imbalance — pit stops occur on a small fraction of laps.
- XGBoost and Random Forest consistently outperform baselines across all metrics, with XGBoost achieving the highest ROC-AUC and PR-AUC.
- Preprocessing matters significantly: SMOTE and feature scaling together improve minority-class recall substantially, while raw accuracy alone is a misleading metric.
- Stint length, lap time delta, and tyre compound are the strongest predictors of an imminent pit stop.
- Model rankings shift depending on the metric used — models that rank highly on accuracy can rank poorly on PR-AUC, underscoring the importance of metric selection for imbalanced problems.
- XGBoost is robust under cross-validation , showing low variance across folds, indicating it generalises well to unseen race data.
- Threshold tuning at inference time can align model outputs with real race-strategy risk tolerance, making the final model practically deployable for pit-window decision support.


## Requirements

See `requirements.txt`. Key libraries: `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `pandas`, `numpy`, `matplotlib`, `seaborn`.
