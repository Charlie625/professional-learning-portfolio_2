"""
Loan Repayment Classification Project

This script trains and compares several classification models to predict
whether a loan will be paid back. It creates summary tables, evaluation
reports, and charts as evidence for a professional learning portfolio.
"""

from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = PROJECT_DIR / "trainData.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET_COLUMN = "loan_paid_back"
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(exist_ok=True)


def build_one_hot_encoder():
    """Create a OneHotEncoder that works across different scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def save_dataset_overview(df: pd.DataFrame) -> None:
    """Save a simple overview of all columns in the dataset."""
    overview = pd.DataFrame(
        {
            "column": df.columns,
            "data_type": df.dtypes.astype(str).values,
            "missing_values": df.isna().sum().values,
            "unique_values": df.nunique().values,
        }
    )
    overview.to_csv(OUTPUT_DIR / "dataset_overview.csv", index=False)


def main() -> None:
    print("=" * 70)
    print("Loan Repayment Classification Project")
    print("=" * 70)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Cannot find trainData.csv. Please place trainData.csv in the same folder as this script."
        )

    df = pd.read_csv(DATA_FILE)

    print("\nDataset shape:")
    print(df.shape)

    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isna().sum())

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in the dataset.")

    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    save_dataset_overview(df)

    columns_to_drop = [TARGET_COLUMN]

    for col in ["id", "source ID", "source_id"]:
        if col in df.columns:
            columns_to_drop.append(col)

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN]

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    print("\nNumeric features:")
    print(numeric_features)

    print("\nCategorical features:")
    print(categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    models = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=7,
        ),
    }

    results = []
    trained_models = {}
    classification_reports = {}

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Training model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        result = {
            "Model": model_name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Balanced Accuracy": round(
                balanced_accuracy_score(y_test, y_pred), 4
            ),
            "Precision Macro": round(
                precision_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                4,
            ),
            "Recall Macro": round(
                recall_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                4,
            ),
            "F1 Macro": round(
                f1_score(
                    y_test,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                4,
            ),
            "Precision Not Paid Back": round(
                precision_score(
                    y_test,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                4,
            ),
            "Recall Not Paid Back": round(
                recall_score(
                    y_test,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                4,
            ),
            "F1 Not Paid Back": round(
                f1_score(
                    y_test,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                4,
            ),
            "Precision Paid Back": round(
                precision_score(
                    y_test,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                4,
            ),
            "Recall Paid Back": round(
                recall_score(
                    y_test,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                4,
            ),
            "F1 Paid Back": round(
                f1_score(
                    y_test,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                4,
            ),
        }

        results.append(result)
        trained_models[model_name] = pipeline

        report_text = classification_report(
            y_test,
            y_pred,
            target_names=["Not Paid Back", "Paid Back"],
            zero_division=0,
        )

        classification_reports[model_name] = report_text
        print(report_text)

    results_df = pd.DataFrame(results).sort_values(
        by="F1 Macro",
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("Model comparison sorted by Macro F1:")
    print(results_df)

    results_df.to_csv(
        OUTPUT_DIR / "model_results.csv",
        index=False,
    )

    best_model_name = str(results_df.iloc[0]["Model"])
    best_model = trained_models[best_model_name]

    print(f"\nBest model selected: {best_model_name}")

    best_predictions = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Paid Back", "Paid Back"],
    )

    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "confusion_matrix_best_model.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["F1 Macro"])
    plt.title("Model Comparison by Macro F1 Score")
    plt.xlabel("Model")
    plt.ylabel("Macro F1 Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "model_comparison_macro_f1.png",
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(6, 4))
    class_counts = y.value_counts().sort_index()
    plt.bar(
        ["Not Paid Back", "Paid Back"],
        [
            class_counts.get(0, 0),
            class_counts.get(1, 0),
        ],
    )
    plt.title("Target Class Distribution")
    plt.xlabel("Loan Paid Back")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "class_distribution.png",
        dpi=300,
    )
    plt.close()

    with open(
        OUTPUT_DIR / "classification_reports.txt",
        "w",
        encoding="utf-8",
    ) as f:
        for model_name, report_text in classification_reports.items():
            f.write("=" * 70 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write("=" * 70 + "\n")
            f.write(report_text)
            f.write("\n\n")

    with open(
        OUTPUT_DIR / "project_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("Loan Repayment Classification Project Summary\n")
        f.write("===========================================\n\n")
        f.write(f"Dataset shape: {df.shape}\n")
        f.write(f"Target column: {TARGET_COLUMN}\n")
        f.write(f"Dropped columns: {columns_to_drop}\n")
        f.write(f"Numeric features: {numeric_features}\n")
        f.write(f"Categorical features: {categorical_features}\n")
        f.write(f"Best model selected: {best_model_name}\n\n")

        f.write("Reason for evaluation method:\n")
        f.write(
            "The target classes are imbalanced, so accuracy alone may be misleading. "
            "For this reason, Macro F1 and Balanced Accuracy were used to compare models more fairly.\n\n"
        )

        f.write("Model comparison:\n")
        f.write(results_df.to_string(index=False))

        f.write("\n\nReflection note:\n")
        f.write(
            "This project helped me practise a complete machine learning workflow, including data loading, "
            "preprocessing, train-test splitting, model training, evaluation, model comparison, and evidence generation."
        )

    print("\nAll output files have been saved in the outputs folder.")


if __name__ == "__main__":
    main()
