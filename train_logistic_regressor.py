import argparse
import os
from glob import glob
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from util import get_list_distances_from_preds


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Train a logistic regression model to predict whether a query is hard "
            "(i.e. the initial VPR top-1 is wrong but reranking fixes it), "
            "based on image-matching inlier statistics."
        )
    )

    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="Directory with predictions of a VPR model (Phase 1 txt files).",
    )
    parser.add_argument(
        "--inliers-dir",
        type=str,
        required=True,
        help="Directory with image matching results (Phase 2 torch files).",
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=100,
        help="Number of predictions per query to consider.",
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=float,
        default=25.0,
        help="Distance (meters) under which a prediction is considered correct.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data used for validation (train/val split).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/val split.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="benefit_from_rerank",
        choices=["benefit_from_rerank", "baseline_incorrect"],
        help=(
            "How to define a hard query:\n"
            "- benefit_from_rerank: hard=1 if R@1 is wrong before reranking "
            "but correct after reranking.\n"
            "- baseline_incorrect: hard=1 if R@1 is wrong before reranking."
        ),
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        default="logistic_regressor.joblib",
        help="Path where the trained logistic regression model will be saved.",
    )

    return parser.parse_args()


def build_dataset(
    preds_dir: str,
    inliers_dir: str,
    num_preds: int,
    positive_dist_threshold: float,
    label_mode: str,
):
    """
    Build a dataset of per-query features and labels for logistic regression.

    Features (X):
        - x0: inlier count between query and FIRST retrieved image (before reranking).
        - x1: mean inlier count over all top-k predictions.
        - x2: max inlier count over all top-k predictions.
        - x3: std-dev of inlier counts over top-k predictions.

    Labels (y):
        - If label_mode == 'benefit_from_rerank':
            y = 1 if R@1 is wrong before reranking but correct after reranking.
        - If label_mode == 'baseline_incorrect':
            y = 1 if R@1 is wrong before reranking, regardless of reranking.
    """
    preds_dir = Path(preds_dir)
    inliers_dir = Path(inliers_dir)

    txt_files = glob(os.path.join(str(preds_dir), "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    X_list = []
    y_list = []

    for txt_file_query in txt_files:
        # Phase 1: geographic distances in original (descriptor-based) order.
        geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[
            :num_preds
        ]
        if geo_dists.numel() == 0:
            continue

        # Corresponding Phase 2 results.
        torch_file_query = inliers_dir.joinpath(
            Path(txt_file_query).name.replace("txt", "torch")
        )
        if not torch_file_query.exists():
            continue

        query_results = torch.load(torch_file_query, weights_only=False)
        if len(query_results) == 0:
            continue

        num_considered = min(num_preds, len(query_results))

        inliers = torch.zeros(num_considered, dtype=torch.float32)
        for i in range(num_considered):
            inliers[i] = query_results[i]["num_inliers"]

        # Baseline (before reranking): top-1 is first prediction.
        baseline_top1_dist = float(geo_dists[0])
        baseline_correct = baseline_top1_dist <= positive_dist_threshold

        # Reranked by inliers (same as reranking.py).
        inliers_sorted, indices = torch.sort(inliers, descending=True)
        geo_dists_reranked = geo_dists[:num_considered][indices]
        rerank_top1_dist = float(geo_dists_reranked[0])
        rerank_correct = rerank_top1_dist <= positive_dist_threshold

        if label_mode == "benefit_from_rerank":
            is_hard = (not baseline_correct) and rerank_correct
        elif label_mode == "baseline_incorrect":
            is_hard = not baseline_correct
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        # Feature vector for this query.
        top1_inliers = float(inliers[0])  # inliers for first retrieved image
        mean_inliers = float(inliers.mean())
        max_inliers = float(inliers.max())
        std_inliers = float(inliers.std()) if num_considered > 1 else 0.0

        X_list.append(
            [
                top1_inliers,
                mean_inliers,
                max_inliers,
                std_inliers,
            ]
        )
        y_list.append(int(is_hard))

    if not X_list:
        raise RuntimeError(
            "No data points were collected. "
            "Check that preds-dir and inliers-dir are compatible and non-empty."
        )

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def train_logistic_regressor(X, y, test_size: float, random_state: int):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    print("=== Logistic Regression (Hard Query Prediction) ===")
    print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.3f}")
    print(f"Val accuracy:   {accuracy_score(y_val, y_val_pred):.3f}")
    print("\nValidation confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, y_val_pred))
    print("\nValidation classification report:")
    print(classification_report(y_val, y_val_pred, digits=3))

    return clf


def main():
    args = parse_arguments()

    X, y = build_dataset(
        preds_dir=args.preds_dir,
        inliers_dir=args.inliers_dir,
        num_preds=args.num_preds,
        positive_dist_threshold=args.positive_dist_threshold,
        label_mode=args.label_mode,
    )

    print(f"Collected {len(y)} queries for training.")
    print(f"Class balance (0=easy, 1=hard): {np.bincount(y)}")

    clf = train_logistic_regressor(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    joblib.dump(clf, args.output_model_path)
    print(f"\nSaved trained logistic regression model to: {args.output_model_path}")


if __name__ == "__main__":
    main()



