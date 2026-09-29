import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED
)
from src.utils.seed import set_seed
from src.models.efficientnet_cbam import EfficientNetB0_CBAM
from src.data.dataset import get_dataset
from src.training.trainer import ModelTrainer, get_device
from src.evaluation.metrics import print_metrics_table

def train_modality(modality: str, epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE):
    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"Executing on hardware device: {device}")

    train_ds = get_dataset(modality, split="train")
    val_ds = get_dataset(modality, split="val")
    test_ds = get_dataset(modality, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    print(f"\nInitialized {modality.upper()} DataLoader:")
    print(f"Train batches: {len(train_loader)} ({len(train_ds)} samples)")
    print(f"Val batches  : {len(val_loader)} ({len(val_ds)} samples)")
    print(f"Test batches : {len(test_loader)} ({len(test_ds)} samples)")

    # Instantiate EfficientNet-B0 + CBAM
    model = EfficientNetB0_CBAM(num_classes=2, pretrained=True)
    trainer = ModelTrainer(
        model=model,
        modality=modality,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        epochs=epochs,
        patience=EARLY_STOPPING_PATIENCE
    )

    fit_result = trainer.fit()

    print("\n" + "=" * 50)
    print("Evaluating Best Model on Unseen Internal TEST SET...")
    print("=" * 50)
    test_metrics = trainer.evaluate(test_loader)
    print_metrics_table(test_metrics, title=f"TEST SET PERFORMANCE - {modality.upper()}")

    return test_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Independent Modality Model (EfficientNet-B0 + CBAM)")
    parser.add_argument("--modality", type=str, default="conjunctiva", choices=["conjunctiva", "cp_anemic", "palm", "fingernail"])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)

    args = parser.parse_args()
    train_modality(modality=args.modality, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

if __name__ == "__main__":
    main()
