import sys
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import GridSearchCV

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.config import (
    RAW_DATA,
    PROCESSED_DATA,
    MODEL_PATH,
    MODEL_COMPARISON
)

from src.preprocess import (
    load_data,
    prepare_data,
    create_preprocessor,
    split_data
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    }


def main():

    # Load data
    df = load_data(RAW_DATA)

    # Save cleaned data
    df.to_csv(PROCESSED_DATA, index=False)

    # Prepare data
    X, y = prepare_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Preprocessor
    preprocessor = create_preprocessor(X)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    results = []

    # Train models
    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        metrics["Model"] = name
        results.append(metrics)

    # Create comparison dataframe
    comparison = pd.DataFrame(results)

    comparison = comparison[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    ]

    print("\nModel Comparison:")
    print(comparison)

    # Save comparison
    comparison.to_csv(
        MODEL_COMPARISON,
        index=False
    )

    # Hyperparameter tuning for Random Forest
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42
        ))
    ])

    parameter_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [10, 15],
        "classifier__min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        rf_pipeline,
        parameter_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross Validation F1:")
    print(grid_search.best_score_)

    # Save tuned Random Forest
    joblib.dump(
        grid_search.best_estimator_,
        MODEL_PATH
    )

    print("\nFinal model saved successfully.")


if __name__ == "__main__":
    main()