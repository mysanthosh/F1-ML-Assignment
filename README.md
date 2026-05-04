# F1 Strategy Dataset – Pit-Stop Prediction
## Supervised Learning Assignment – Submission 1

Student: Mysanthosh Karnati  

## Dataset
Dataset Title	  :	F1 Strategy Dataset (Lap-Level Race Data)
Domain            : Sports Analytics / Motorsports Data Science  
Task              : Binary Classification  
Target Variable   : `pit_stop` (0 = No Pit Stop, 1 = Pit Stop)
Source Link	      : https://www.kaggle.com/datasets/aadigupta1601/f1-strategy-dataset-pit-stop-prediction
Number of Rows    : 101,371 rows
Number of Columns : 16 columns
Short Note	      :	The selected dataset is a structured, publicly available Formula 1 race dataset   containing lap-by-lap performance and strategy data with over 100,000 rows and more than 15 features. It includes recent data up to 2025, making it suitable for supervised machine learning tasks.


## Project Structure
F1-ML-Assignment/
│
├── data/
│   └── f1_strategy_dataset_v4.csv
│       → Contains the dataset used for training and evaluation
│
├── notebooks/
│   └── F1_PitStop_Prediction.ipynb
│       → Used for experimentation, visualization, and step-by-step analysis (optional)
│
├── src/
│   ├── RQ1_baseline_models.py
│   │       → Builds and evaluates baseline models
│   │
│   ├── RQ2_model_comparison.py
│   │       → Compares multiple models and identifies the best-performing model
│   │
│   ├── RQ3_preprocessing_analysis.py
│   │       → Analyzes the impact of different preprocessing techniques
│   │
│   ├── RQ4_feature_importance.py
│   │       → Identifies and explains important features influencing predictions
│   │
│   ├── RQ5_metric_sensitivity.py
│   │       → Evaluates how model rankings change across different metrics
│   │
│   ├── RQ6_robustness_cv.py
│   │       → Tests model robustness using cross-validation
│   │
│   └── RQ7_practical_usefulness.py
│           → Assesses real-world usability and reliability of the model
│
├── figures/
│   └── (all output graphs)
│       → Stores all generated visualizations and results
│
├── docs/
│   └── README_Project.pdf
│       → Contains detailed project documentation/report
│
├── requirements.txt
│       → Lists all required Python libraries
│
├── README.md
│       → Provides project overview, setup instructions, and explanations
│
└── .gitignore
        → Specifies files and folders to be ignored by Git

## Quick Start
pip install -r requirements.txt
jupyter notebook F1_PitStop_Prediction.ipynb

To use the real Kaggle dataset, download it from the link above and replace the data generation block with:
df = pd.read_csv('f1_strategy.csv')


## Models Implemented
| Model | Accuracy | F1-score | AUC |
| Logistic Regression | 0.8291 | 0.2249 | 0.7437 |
| Decision Tree | 0.8282 | 0.2228 | 0.7550 |
| k-NN | 0.8249 | 0.2339 | 0.7028 |
| Random Forest | 0.8266 | 0.2338 | 0.7422 |
| **XGBoost** | **0.8303** | **0.2409** | **0.7632** |

**Winner: XGBoost** with highest AUC (0.7632) and Precision (0.578).

## Research Questions

1. How do baseline models perform on pit-stop prediction?  → File: src/RQ1_baseline_models.ipynb

2. Which model achieves best predictive performance?       → File: src/RQ2_model_comparison.ipynb

3. How does preprocessing affect model performance?        → File: src/RQ3_preprocessing_analysis.ipynb
 
4. Which features are most important for prediction?       → File: src/RQ4_feature_importance.ipynb

5. How do model rankings change across different metrics?  → File: src/RQ5_metric_sensitivity.ipynb

6. How robust is XGBoost under cross-validation?           → File: src/RQ6_robustness_cv.ipynb

7. Is the final model practically useful for race strategy decisions?
→ File: src/RQ7_practical_usefulness.ipynb

## Requirements
See `requirements.txt`. Key packages: scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn, jupyter.
