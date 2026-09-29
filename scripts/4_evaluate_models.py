import os
import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from src.config import (
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    BATCH_SIZE,
    NUM_WORKERS
)
from src.models.efficientnet_cbam import EfficientNetB0_CBAM
from src.data.dataset import get_dataset
from src.evaluation.metrics import compute_classification_metrics, print_metrics_table
from src.evaluation.gradcam import GradCAM
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_training_history
from src.training.trainer import get_device

def evaluate_modality(modality: str = "conjunctiva"):
    device = get_device()
    checkpoint_path = CHECKPOINTS_DIR / modality / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train model first using scripts/3_train_models.py.")

    print(f"\nLoading best model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = EfficientNetB0_CBAM(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 1. Internal Test Set Evaluation
    test_ds = get_dataset(modality, split="test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_preds, all_targets, all_probs, all_pids = [], [], [], []

    with torch.no_grad():
        for images, labels, pids in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_pids.extend(pids)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
    print_metrics_table(metrics, title=f"TEST SET PERFORMANCE - {modality.upper()}")

    # Save metrics table (CSV and LaTeX for paper)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = TABLES_DIR / f"{modality}_test_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics to {metrics_csv}")

    # 2. Visualizations
    curves_dir = FIGURES_DIR / "curves"
    cm_dir = FIGURES_DIR / "confusion_matrices"
    gradcam_dir = FIGURES_DIR / "gradcam" / modality
    curves_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    # Plot Confusion Matrix
    cm_path = cm_dir / f"{modality}_confusion_matrix.png"
    plot_confusion_matrix(all_targets, all_preds, output_path=str(cm_path))

    # Plot ROC Curve
    roc_path = curves_dir / f"{modality}_roc_curve.png"
    plot_roc_curve(all_targets, all_probs, output_path=str(roc_path))

    # Plot Training curves if history file exists
    history_csv = PROJECT_ROOT / "results" / "logs" / f"{modality}_training_history.csv"
    if history_csv.exists():
        loss_curve_path = curves_dir / f"{modality}_training_curves.png"
        plot_training_history(str(history_csv), output_path=str(loss_curve_path))

    # 3. Explainability: Grad-CAM Visualization on Sample Test Images
    print(f"\nGenerating Grad-CAM visual heatmaps in {gradcam_dir}...")
    gradcam = GradCAM(model=model)
    from src.data.transforms import get_eval_transforms
    eval_transform = get_eval_transforms()

    num_samples = min(6, len(test_ds))
    for i in range(num_samples):
        row = test_ds.data.iloc[i]
        label_val = int(row["who_anemic"])
        actual_label = "Anemic" if label_val == 1 else "Normal"
        
        # Determine image path
        if "palpebral_mask_path" in row and pd.notna(row["palpebral_mask_path"]):
            img_path = row["palpebral_mask_path"]
            orig_pil = Image.open(img_path)
            bbox = orig_pil.convert("L").getbbox()
            if bbox:
                orig_pil = orig_pil.crop(bbox)
        else:
            img_path = row.get("image_path") or row.get("original_image_path")
            orig_pil = Image.open(img_path)

        orig_pil = orig_pil.convert("RGB")
        input_tensor = eval_transform(orig_pil).unsqueeze(0).to(device)

        # Generate heatmap for predicted class
        with torch.enable_grad():
            heatmap = gradcam.generate_heatmap(input_tensor)

        sample_name = row.get("patient_id") or row.get("subject_id") or f"sample_{i}"
        overlay_img = gradcam.overlay_heatmap(orig_pil, heatmap, alpha=0.5)
        save_path = gradcam_dir / f"{sample_name}_pred_{actual_label}.png"
        overlay_img.save(save_path)

    print(f"Saved Grad-CAM overlays to {gradcam_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Model and Generate Paper Artifacts")
    parser.add_argument("--modality", type=str, default="conjunctiva", choices=["conjunctiva", "cp_anemic", "palm", "fingernail"])
    args = parser.parse_args()
    evaluate_modality(modality=args.modality)

if __name__ == "__main__":
    main()
