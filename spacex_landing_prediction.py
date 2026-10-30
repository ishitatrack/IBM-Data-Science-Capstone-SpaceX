"""Tune classifiers to predict SpaceX Falcon 9 first-stage landing success."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
CATEGORICAL_FEATURES = {"launchsite", "orbit", "orbits", "serial"}
LEAKAGE_FEATURES = {
    "outcome",
    "landingoutcome",
    "landingsuccess",
    "corelandingoutcomes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate tuned SpaceX landing classifiers."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset_part_2.csv",
        help="Processed CSV path (default: dataset_part_2.csv beside this script).",
    )
    parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default=Path(__file__).resolve().parent / "spacex_confusion_matrices.png",
        help="File to save the four test-set confusion matrices.",
    )
    return parser.parse_args()


def prepare_features(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}. "
            "Pass its CSV path as the first argument."
        )

    data = pd.read_csv(dataset_path)
    if "Class" not in data.columns:
        raise ValueError("The processed dataset must contain a 'Class' target column.")

    target = pd.to_numeric(data["Class"], errors="coerce")
    valid_target = target.isin([0, 1])
    dropped_count = int((~valid_target).sum())
    if dropped_count:
        print(f"Dropping {dropped_count} row(s) with missing or non-binary Class.")
    data = data.loc[valid_target].copy()
    target = target.loc[valid_target].astype("int8")

    excluded = [
        column
        for column in data.columns
        if column.casefold().replace("_", "") in LEAKAGE_FEATURES
    ]
    feature_frame = data.drop(columns=["Class", *excluded], errors="ignore")

    categorical_columns = [
        column
        for column in feature_frame.columns
        if column.casefold() in CATEGORICAL_FEATURES
    ]
    numeric_columns = [
        column
        for column in feature_frame.columns
        if column not in categorical_columns
        and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    ignored_columns = [
        column
        for column in feature_frame.columns
        if column not in categorical_columns and column not in numeric_columns
    ]
    if ignored_columns:
        print("Ignoring non-numeric, non-categorical columns: " + ", ".join(ignored_columns))

    features = feature_frame[numeric_columns + categorical_columns].copy()
    for column in categorical_columns:
        features[column] = features[column].astype("string")

    features = pd.get_dummies(
        features,
        columns=categorical_columns,
        dummy_na=True,
        dtype=int,
    )
    if features.empty:
        raise ValueError("No usable model features were found in the processed dataset.")
    if target.nunique() != 2:
        raise ValueError("Class must contain both binary labels 0 and 1.")

    return features, target


def build_searches(cv: StratifiedKFold, train_size: int) -> dict[str, GridSearchCV]:
    knn_neighbors = [
        neighbors for neighbors in (1, 3, 5, 7, 9) if neighbors <= train_size
    ]
    if not knn_neighbors:
        raise ValueError("The training split is too small to fit KNN.")

    candidates = {
        "Logistic Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            solver="liblinear",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l1", "l2"],
            },
        ),
        "Support Vector Classifier": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC()),
                ]
            ),
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
            },
        ),
        "Decision Tree Classifier": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        DecisionTreeClassifier(random_state=RANDOM_STATE),
                    ),
                ]
            ),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 3, 5, 10],
                "model__min_samples_split": [2, 5, 10],
            },
        ),
        "K-Nearest Neighbors": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {
                "model__n_neighbors": knn_neighbors,
                "model__weights": ["uniform", "distance"],
            },
        ),
    }

    return {
        name: GridSearchCV(
            estimator=estimator,
            param_grid=parameters,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        )
        for name, (estimator, parameters) in candidates.items()
    }


def main() -> None:
    args = parse_args()
    features, target = prepare_features(args.dataset)

    class_counts = target.value_counts()
    if len(features) < 5 or class_counts.min() < 2:
        raise ValueError(
            "Need at least five rows and at least two examples of each Class "
            "to create a stratified 80/20 split."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    train_class_counts = y_train.value_counts()
    cv_splits = min(5, int(train_class_counts.min()))
    if cv_splits < 2:
        raise ValueError(
            "Not enough examples in each training class for cross-validation."
        )
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    searches = build_searches(cv, len(x_train))
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.ravel()

    for axis, (name, search) in zip(axes_flat, searches.items()):
        search.fit(x_train, y_train)
        accuracy = search.score(x_test, y_test)
        print(f"{name} test accuracy: {accuracy:.4f}")
        print(f"  Best parameters: {search.best_params_}")

        ConfusionMatrixDisplay.from_estimator(
            search.best_estimator_,
            x_test,
            y_test,
            display_labels=["Failure (0)", "Success (1)"],
            cmap="Blues",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(f"{name}\nTest accuracy: {accuracy:.3f}")

    figure.suptitle("SpaceX First-Stage Landing: Test Confusion Matrices")
    figure.tight_layout()
    args.confusion_matrix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.confusion_matrix, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Confusion matrices saved to: {args.confusion_matrix}")


if __name__ == "__main__":
    main()
