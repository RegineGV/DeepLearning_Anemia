import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> dict:
    """
    Computes comprehensive medical diagnostic evaluation metrics:
    - Accuracy
    - Sensitivity (Recall / True Positive Rate)
    - Specificity (True Negative Rate)
    - Precision (Positive Predictive Value)
    - F1-Score
    - ROC-AUC (Area Under ROC Curve)
    - Confusion Matrix (TN, FP, FN, TP)
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    sens = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC-AUC
    auc = np.nan
    if y_prob is not None:
        try:
            # If probabilities are 2D [N, 2], take probability for class 1 (Anemic)
            if len(y_prob.shape) == 2:
                prob_anemic = y_prob[:, 1]
            else:
                prob_anemic = y_prob
            auc = roc_auc_score(y_true, prob_anemic)
        except Exception:
            auc = np.nan

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "sensitivity": float(sens),
        "recall": float(sens),
        "specificity": float(spec),
        "f1_score": float(f1),
        "roc_auc": float(auc) if not np.isnan(auc) else None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

def print_metrics_table(metrics: dict, title: str = "Evaluation Metrics") -> None:
    print("\n" + "=" * 45)
    print(f"{title:^45}")
    print("=" * 45)
    print(f"Accuracy         : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Sensitivity/Rec  : {metrics['sensitivity']:.4f} ({metrics['sensitivity']*100:.2f}%)")
    print(f"Specificity      : {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"Precision        : {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"F1-Score         : {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
    print("-" * 45)
    print(f"Confusion Matrix : [TN: {metrics['true_negatives']}, FP: {metrics['false_positives']}]")
    print(f"                   [FN: {metrics['false_negatives']}, TP: {metrics['true_positives']}]")
    print("=" * 45)
