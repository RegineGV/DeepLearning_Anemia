import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_training_history(history_csv: str, output_path: str = None) -> None:
    """
    Plots Train vs Val Loss and Accuracy curves across training epochs.
    """
    df = pd.read_csv(history_csv)
    epochs = df["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    axes[0].plot(epochs, df["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, df["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title("Training and Validation Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Accuracy Curve
    axes[1].plot(epochs, df["train_acc"] * 100, "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    axes[1].plot(epochs, df["val_acc"] * 100, "s--", label="Val Accuracy", color="#d62728", linewidth=2)
    axes[1].set_title("Training and Validation Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy (%)", fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Training curves saved to {output_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names=("Normal", "Anemic"), output_path: str = None) -> None:
    """
    Plots a publication-quality confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j, y=i, s=f"{cm[i, j]}",
                va="center", ha="center", size="xx-large", weight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual Ground Truth", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Confusion matrix saved to {output_path}")
    plt.close()

def plot_roc_curve(y_true, y_prob, output_path: str = None) -> float:
    """
    Plots Receiver Operating Characteristic (ROC) curve and returns AUC score.
    """
    if len(y_prob.shape) == 2:
        y_prob = y_prob[:, 1]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#d62728", lw=2.5, label=f"EfficientNet-B0 + CBAM (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"ROC curve saved to {output_path}")
    plt.close()
    return roc_auc
