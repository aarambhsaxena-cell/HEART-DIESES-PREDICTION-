# Heart Disease Prediction — Machine Learning & Deep Learning

**Student:** Aarambh Saxena  
**Roll No.:** 252010065  
**Course:** BCA (AI Analyst), in collaboration with IBM  
**Guide:** Hritik Nayak  
**University:** United University, Prayagraj  
**Session:** 2026–27

## Project overview

This educational project explores a public Kaggle dataset and builds classification models to estimate whether a record is labelled as having heart disease. It follows a complete beginner-friendly ML workflow: dataset loading, data-quality checks, exploratory analysis, preprocessing, training, evaluation, comparison and saving models. It is **not** a medical or diagnostic tool.

## Dataset

- **Source:** [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Dataset file:** `heart.csv`
- **Rows:** 918
- **Target:** `HeartDisease` (0 = no disease label, 1 = disease label)
- **Features:** Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope

The notebook downloads the dataset using `kagglehub`. If downloading fails, upload `heart.csv` to the notebook runtime and set the CSV path accordingly.

## Workflow

1. Load dataset from Kaggle.
2. Inspect shape, data types, missing values, duplicate rows and summary statistics.
3. Treat zero values in `RestingBP` and `Cholesterol` as missing placeholders, then impute within preprocessing pipelines.
4. Perform exploratory data analysis (target distribution, age, MaxHR, category disease rates and numeric correlations).
5. Split data into stratified training and test sets (80/20).
6. Impute, scale numerical features and one-hot encode categorical features.
7. Compare Logistic Regression, Random Forest and Gradient Boosting using stratified 5-fold cross-validation.
8. Train a Keras MLP neural network with validation and early stopping.
9. Evaluate on the held-out test set using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices and ROC curves.
10. Visualize tree-based feature importance when available and save models.

## Why these models?

- **Logistic Regression:** a simple baseline for binary classification.
- **Random Forest:** combines decision trees and can capture non-linear patterns.
- **Gradient Boosting:** builds successive trees to improve prediction performance.
- **Keras MLP:** a neural network comparison to explore a deep-learning approach to tabular data.

## Run the project

### Google Colab

1. Open the notebook in Google Colab.
2. Run the cells from top to bottom.
3. The notebook downloads the dataset (or use the manual-upload fallback).
4. Review the printed metrics and plots.

### Local setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Heart_Disease_Prediction.ipynb
```

## Files

- `Heart_Disease_Prediction.ipynb` — complete analysis, training and evaluation workflow.
- `README.md` — project documentation.
- `requirements.txt` — required Python packages.
- `heart_model.joblib` and `heart_mlp.keras` — generated after running the notebook.

## Results

The results table should be filled from the notebook after it has been run. No metric values are pre-filled because they have not been verified from an execution.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Best ML model (selected using CV F1) | — | — | — | — | — |
| Keras MLP | — | — | — | — | — |

## Limitations

- The dataset is relatively small and may not represent other populations or hospitals.
- Metrics are dataset-specific and do not establish clinical usefulness.
- The model is for academic learning only, not diagnosis, treatment or medical decisions.

## GitHub checklist

Before publishing, run all notebook cells and save the notebook with outputs if appropriate. Add the dataset only if its licence and Kaggle terms permit redistribution; otherwise link to the source. Update the results table using the actual run, and include generated model files only if desired.
