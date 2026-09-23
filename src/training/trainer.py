import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED
)
from src.utils.seed import set_seed
from src.evaluation.metrics import compute_classification_metrics, print_metrics_table
from src.training.losses import get_loss_function

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class ModelTrainer:
    """
    Standardized trainer for independent modality models (EfficientNet-B0 + CBAM).
    """
    def __init__(
        self,
        model: nn.Module,
        modality: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = None,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        epochs: int = NUM_EPOCHS,
        patience: int = EARLY_STOPPING_PATIENCE,
        loss_type: str = "cross_entropy"
    ):
        self.model = model
        self.modality = modality
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or get_device()
        self.epochs = epochs
        self.patience = patience

        self.model.to(self.device)

        # Compute class weights from training set to handle minor class imbalance
        all_train_labels = [label.item() for _, label, _ in train_loader.dataset]
        class_counts = np.bincount(all_train_labels)
        total_samples = len(all_train_labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        self.criterion = get_loss_function(loss_type, class_weights=weights_tensor)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        # Output paths
        self.checkpoint_dir = CHECKPOINTS_DIR / modality
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pth"

        self.log_dir = LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = []

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader.dataset)
        acc = (np.array(all_preds) == np.array(all_targets)).mean()
        return {"loss": avg_loss, "accuracy": float(acc)}

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        for images, labels, _ in data_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader.dataset)
        metrics = compute_classification_metrics(
            y_true=np.array(all_targets),
            y_pred=np.array(all_preds),
            y_prob=np.array(all_probs)
        )
        metrics["loss"] = float(avg_loss)
        return metrics

    def fit(self) -> dict:
        print(f"\nStarting training for modality: [{self.modality.upper()}] on device [{self.device}]")
        print(f"Total Epochs: {self.epochs} | Early Stopping Patience: {self.patience}")

        best_val_f1 = -1.0
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()
            elapsed = time.time() - start_time

            val_f1 = val_metrics["f1_score"]
            val_acc = val_metrics["accuracy"]
            val_loss = val_metrics["loss"]

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_metrics["roc_auc"],
                "val_sens": val_metrics["sensitivity"],
                "val_spec": val_metrics["specificity"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "time_sec": elapsed
            }
            self.history.append(epoch_record)

            # Checkpoint condition: Best validation F1 score
            if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < self.history[best_epoch-1]["val_loss"] if best_epoch > 0 else False):
                best_val_f1 = val_f1
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "modality": self.modality
                }, self.best_checkpoint_path)
                improved_mark = "(* Best Saved)"
            else:
                epochs_no_improve += 1
                improved_mark = ""

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']*100:.1f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%, F1: {val_f1:.4f} {improved_mark}"
            )

            if epochs_no_improve >= self.patience:
                print(f"\n[Early Stopping] No improvement for {self.patience} epochs. Stopping.")
                break

        # Save history log
        history_df = pd.DataFrame(self.history)
        history_csv = self.log_dir / f"{self.modality}_training_history.csv"
        history_df.to_csv(history_csv, index=False)
        print(f"\nTraining completed! Best Model from Epoch {best_epoch} saved to {self.best_checkpoint_path}")
        print(f"Training history saved to {history_csv}")

        # Load best model weights for subsequent evaluation
        best_checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(best_checkpoint["model_state_dict"])

        return {
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "checkpoint_path": str(self.best_checkpoint_path)
        }
