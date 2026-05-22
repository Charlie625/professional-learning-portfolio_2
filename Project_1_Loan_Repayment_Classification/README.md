# Loan Repayment Classification Project

## Project Overview

This project is a Python-based machine learning classification project. The aim is to predict whether a loan will be paid back using customer, financial, loan, and credit-related variables.

This project was completed as part of my professional learning portfolio. It demonstrates practical skills in Python programming, data preprocessing, machine learning model training, model evaluation, and evidence generation.

## Dataset

The dataset contains loan applicant information, including demographic, income, credit, loan, and repayment-related variables.

The target variable is:

- `loan_paid_back`

A value of `1` means the loan was paid back.
A value of `0` means the loan was not paid back.

## Tools Used

- Python
- pandas
- matplotlib
- scikit-learn
- PyCharm
- GitHub

## Machine Learning Workflow

The project follows this workflow:

1. Load the dataset
2. Check dataset shape, missing values, and target distribution
3. Remove identifier columns that should not be used as prediction features
4. Split the dataset into training and testing sets
5. Preprocess numeric and categorical variables
6. Train multiple classification models
7. Evaluate model performance
8. Compare models using several evaluation metrics
9. Select the best model based on Macro F1 score
10. Generate CSV summaries, charts, and text-based evidence files

## Models Used

The following models were used:

- Dummy Baseline
- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbours

The dummy baseline is included to provide a simple comparison point. The other models are compared against this baseline to show whether machine learning provides stronger performance.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1 Score
- Class-level precision, recall, and F1 score
- Confusion matrix

Because the target classes are imbalanced, accuracy alone may be misleading. Therefore, Macro F1 and Balanced Accuracy are used to compare models more fairly.

## Output Files

When the script is run, the following output files are created in the `outputs` folder:

- `dataset_overview.csv`
- `model_results.csv`
- `classification_reports.txt`
- `project_summary.txt`
- `class_distribution.png`
- `model_comparison_macro_f1.png`
- `confusion_matrix_best_model.png`

These files provide evidence of the analysis workflow and final results.

## How to Run

Place `trainData.csv` in the same folder as the Python script, then run:

```bash
python loan_repayment_classification.py
```

## Professional Learning Reflection

This project helped me understand how a complete machine learning workflow is structured. I practised preparing data, building pipelines, comparing models, and interpreting model evaluation results. I also learned that model accuracy is not always enough, especially when the target classes are imbalanced.
