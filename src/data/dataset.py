import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

# Ensure Pillow handles images with broken headers/ICC profiles gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.config import SPLITS_DIR
from src.data.transforms import get_train_transforms, get_eval_transforms

class ConjunctivaDataset(Dataset):
    """
    Dataset class for Eyes-defy-anemia conjunctiva images.
    Supports both cropped palpebral ROI tissue and full eye images.
    """
    def __init__(
        self,
        split: str = "train",
        splits_csv: str = None,
        use_palpebral_crop: bool = True,
        transform = None,
        cross_domain: bool = False
    ):
        """
        Args:
            split: 'train', 'val', 'test', 'source_train', or 'target_test'
            splits_csv: Path to conjunctiva_splits.csv
            use_palpebral_crop: If True, crops the non-black bounding box of the segmented palpebral mask
            transform: PyTorch transforms
            cross_domain: If True, filters using 'cross_domain_split' column instead of 'split'
        """
        self.split = split
        self.use_palpebral_crop = use_palpebral_crop

        csv_path = Path(splits_csv) if splits_csv else (SPLITS_DIR / "conjunctiva_splits.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file missing: {csv_path}. Run scripts/1_prepare_splits.py first.")

        df = pd.read_csv(csv_path)

        if cross_domain:
            self.data = df[df["cross_domain_split"] == split].reset_index(drop=True)
        else:
            self.data = df[df["split"] == split].reset_index(drop=True)

        # Filter only labeled samples
        self.data = self.data[self.data["who_anemic"].notna()].reset_index(drop=True)
        self.data["who_anemic"] = self.data["who_anemic"].astype(int)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_train_transforms() if split == "train" else get_eval_transforms()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        patient_id = row["patient_id"]
        label = int(row["who_anemic"])

        if self.use_palpebral_crop and pd.notna(row["palpebral_mask_path"]):
            img_path = row["palpebral_mask_path"]
            img = Image.open(img_path)
            
            # Crop to actual tissue bounding box (discard empty background)
            # Check alpha channel if available, else convert to L
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                alpha = img.split()[-1]
                bbox = alpha.getbbox()
            else:
                gray = img.convert("L")
                bbox = gray.getbbox()
                
            if bbox:
                img = img.crop(bbox)
            img = img.convert("RGB")
        else:
            img_path = row["original_image_path"]
            img = Image.open(img_path).convert("RGB")

        if self.transform:
            image_tensor = self.transform(img)
        else:
            image_tensor = img

        return image_tensor, torch.tensor(label, dtype=torch.long), patient_id


class CPAnemicDataset(Dataset):
    """
    Dataset class for CP-AnemiC pediatric conjunctiva images (Ghana cohort).
    """
    def __init__(
        self,
        split: str = "train",
        splits_csv: str = None,
        transform = None,
    ):
        self.split = split
        csv_path = Path(splits_csv) if splits_csv else (SPLITS_DIR / "cp_anemic_splits.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file missing: {csv_path}. Run scripts/1_prepare_splits.py first.")

        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.data["who_anemic"] = self.data["who_anemic"].astype(int)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_train_transforms() if split == "train" else get_eval_transforms()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB")
        label = int(row["who_anemic"])
        patient_id = row["patient_id"]

        if self.transform:
            image_tensor = self.transform(img)
        else:
            image_tensor = img

        return image_tensor, torch.tensor(label, dtype=torch.long), patient_id


class PalmDataset(Dataset):
    """
    Dataset class for Palpable Palm images (Ghana cohort).
    """
    def __init__(
        self,
        split: str = "train",
        splits_csv: str = None,
        transform = None,
    ):
        self.split = split
        csv_path = Path(splits_csv) if splits_csv else (SPLITS_DIR / "palm_splits.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file missing: {csv_path}. Run scripts/1_prepare_splits.py first.")

        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.data["who_anemic"] = self.data["who_anemic"].astype(int)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_train_transforms() if split == "train" else get_eval_transforms()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB")
        label = int(row["who_anemic"])
        sample_id = f"{row['subject_id']}_{row['filename']}"

        if self.transform:
            image_tensor = self.transform(img)
        else:
            image_tensor = img

        return image_tensor, torch.tensor(label, dtype=torch.long), sample_id


class FingernailDataset(Dataset):
    """
    Dataset class for Fingernail Colour images (Ghana cohort).
    """
    def __init__(
        self,
        split: str = "train",
        splits_csv: str = None,
        transform = None,
    ):
        self.split = split
        csv_path = Path(splits_csv) if splits_csv else (SPLITS_DIR / "fingernail_splits.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file missing: {csv_path}. Run scripts/1_prepare_splits.py first.")

        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.data["who_anemic"] = self.data["who_anemic"].astype(int)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_train_transforms() if split == "train" else get_eval_transforms()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB")
        label = int(row["who_anemic"])
        sample_id = f"{row['subject_id']}_{row['filename']}"

        if self.transform:
            image_tensor = self.transform(img)
        else:
            image_tensor = img

        return image_tensor, torch.tensor(label, dtype=torch.long), sample_id


class ConjunctivaMergedDataset(Dataset):
    """
    Dataset class for the unified/merged Conjunctiva dataset (Eyes-defy-anemia + CP-AnemiC: 928 images).
    """
    def __init__(
        self,
        split: str = "train",
        splits_csv: str = None,
        transform = None,
    ):
        self.split = split
        csv_path = Path(splits_csv) if splits_csv else (SPLITS_DIR / "conjunctiva_merged_splits.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file missing: {csv_path}. Run scripts/1_prepare_splits.py first.")

        df = pd.read_csv(csv_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.data["who_anemic"] = self.data["who_anemic"].astype(int)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_train_transforms() if split == "train" else get_eval_transforms()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = row["image_path"]
        img = Image.open(img_path)

        # If it's a segmented mask from eyes_defy, crop bounding box to remove empty background
        if row.get("is_mask", False):
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                alpha = img.split()[-1]
                bbox = alpha.getbbox()
            else:
                gray = img.convert("L")
                bbox = gray.getbbox()
            if bbox:
                img = img.crop(bbox)

        img = img.convert("RGB")
        label = int(row["who_anemic"])
        patient_id = row["patient_id"]

        if self.transform:
            image_tensor = self.transform(img)
        else:
            image_tensor = img

        return image_tensor, torch.tensor(label, dtype=torch.long), patient_id


def get_dataset(modality: str, split: str = "train", **kwargs) -> Dataset:
    """
    Unified dataset factory for all modalities.
    'conjunctiva' returns the merged dataset (Eyes-defy + CP-AnemiC) for single-model comparative training.
    """
    mod_lower = modality.lower()
    if mod_lower in ("conjunctiva", "conjunctiva_merged"):
        return ConjunctivaMergedDataset(split=split, **kwargs)
    elif mod_lower in ("eyes_defy_anemia", "eyes_defy"):
        return ConjunctivaDataset(split=split, **kwargs)
    elif mod_lower in ("cp_anemic", "conjunctiva_cp"):
        return CPAnemicDataset(split=split, **kwargs)
    elif mod_lower == "palm":
        return PalmDataset(split=split, **kwargs)
    elif mod_lower in ("fingernail", "fingernails", "nail"):
        return FingernailDataset(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown modality '{modality}'. Expected one of: conjunctiva, cp_anemic, palm, fingernail.")


if __name__ == "__main__":
    for mod in ["conjunctiva", "cp_anemic", "palm", "fingernail"]:
        ds = get_dataset(mod, split="train")
        img, lbl, sid = ds[0]
        print(f"Modality: {mod:<12} | Train Samples: {len(ds):<4} | Sample 0: {sid[:25]} | Label: {lbl.item()} | Tensor: {img.shape}")

