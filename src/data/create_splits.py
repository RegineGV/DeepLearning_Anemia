import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold

from src.config import (
    RAW_DATA_DIR,
    SPLITS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)
from src.utils.seed import set_seed

def create_conjunctiva_splits():
    """
    Creates stratified 70% Train / 15% Val / 15% Test split for the conjunctiva modality,
    stratified jointly by country (ethnicity) and WHO anemia status.
    Also creates a cross-domain validation split (Italy vs India).
    """
    set_seed(RANDOM_SEED)
    
    metadata_path = SPLITS_DIR / "eyes_defy_anemia_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}. Run prepare_metadata.py first.")

    df = pd.read_csv(metadata_path)
    
    # Filter out samples with missing labels
    labeled_mask = df["who_anemic"].notna()
    labeled_df = df[labeled_mask].copy()
    unlabeled_df = df[~labeled_mask].copy()

    labeled_df["who_anemic"] = labeled_df["who_anemic"].astype(int)

    # Composite stratum: combines ethnicity and disease status for perfectly balanced splits
    labeled_df["stratum"] = labeled_df["country"] + "_" + labeled_df["who_anemic"].astype(str)

    # Split 1: 70% Train vs 30% Temp (Val + Test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED)
    train_idx, temp_idx = next(sss1.split(labeled_df, labeled_df["stratum"]))

    train_data = labeled_df.iloc[train_idx].copy()
    temp_data = labeled_df.iloc[temp_idx].copy()

    # Split 2: 50% / 50% on Temp (each 15% of total) -> Val (15%), Test (15%)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_SEED)
    val_sub_idx, test_sub_idx = next(sss2.split(temp_data, temp_data["stratum"]))

    val_data = temp_data.iloc[val_sub_idx].copy()
    test_data = temp_data.iloc[test_sub_idx].copy()

    train_data["split"] = "train"
    val_data["split"] = "val"
    test_data["split"] = "test"

    combined_df = pd.concat([train_data, val_data, test_data]).drop(columns=["stratum"])

    if len(unlabeled_df) > 0:
        unlabeled_df["split"] = "unlabeled"
        combined_df = pd.concat([combined_df, unlabeled_df])

    # Add cross-domain split (Train on Italy, Test on India)
    combined_df["cross_domain_split"] = combined_df.apply(
        lambda r: "source_train" if r["country"] == "Italy" else ("target_test" if pd.notna(r["who_anemic"]) else "unlabeled"),
        axis=1
    )

    # Sort by country and subject_number
    combined_df = combined_df.sort_values(by=["country", "subject_number"]).reset_index(drop=True)

    out_csv = SPLITS_DIR / "conjunctiva_splits.csv"
    combined_df.to_csv(out_csv, index=False)
    print(f"Stratified split saved to {out_csv} ({len(combined_df)} samples)")

    # Print summary statistics
    print("\n--- Distribution across Splits (70% / 15% / 15%) ---")
    summary = combined_df[combined_df["split"] != "unlabeled"].groupby(["split", "who_anemic"]).size().unstack(fill_value=0)
    summary.columns = ["Normal (0)", "Anemic (1)"]
    summary["Total"] = summary.sum(axis=1)
    summary["% Anemic"] = (summary["Anemic (1)"] / summary["Total"] * 100).round(1)
    print(summary)

    print("\n--- Country Distribution across Splits ---")
    country_summary = combined_df[combined_df["split"] != "unlabeled"].groupby(["split", "country"]).size().unstack(fill_value=0)
    country_summary["Total"] = country_summary.sum(axis=1)
    print(country_summary)

    return combined_df
    

def create_cp_anemic_splits():
    """
    Creates stratified 70% Train / 15% Val / 15% Test split for CP-AnemiC dataset (Ghana pediatric cohort).
    Stratified by WHO anemia status (Hb < 11.0 g/dL).
    """
    set_seed(RANDOM_SEED)
    excel_path = RAW_DATA_DIR / "cp_anemic" / "Anemia_Data_Collection_Sheet.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"CP-Anemic sheet missing: {excel_path}")

    df_sheet = pd.read_excel(excel_path)
    records = []
    for _, row in df_sheet.iterrows():
        img_id = str(row["IMAGE_ID"]).strip()
        fname = f"{img_id}.png"
        anemic = 0 if str(row["Severity"]).strip().lower() == "non-anemic" else 1
        sub_folder = "Anemic" if anemic == 1 else "Non-anemic"
        img_path = (RAW_DATA_DIR / "cp_anemic" / sub_folder / fname).resolve()
        
        if not img_path.exists():
            print(f"Warning: {img_path} not found.")
            continue

        records.append({
            "patient_id": img_id,
            "image_path": str(img_path),
            "hb": float(row["HB_LEVEL"]) if pd.notna(row.get("HB_LEVEL")) else np.nan,
            "severity": str(row["Severity"]).strip(),
            "age_months": row.get("Age(Months)"),
            "gender": str(row["GENDER"]).strip() if pd.notna(row.get("GENDER")) else "Unknown",
            "hospital": str(row["HOSPITAL"]).strip() if pd.notna(row.get("HOSPITAL")) else "Unknown",
            "who_anemic": anemic
        })

    df_cp = pd.DataFrame(records)

    # Stratified 70% Train, 15% Val, 15% Test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED)
    train_idx, temp_idx = next(sss1.split(df_cp, df_cp["who_anemic"]))
    train_df = df_cp.iloc[train_idx].copy()
    temp_df = df_cp.iloc[temp_idx].copy()

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    val_sub, test_sub = next(sss2.split(temp_df, temp_df["who_anemic"]))
    val_df = temp_df.iloc[val_sub].copy()
    test_df = temp_df.iloc[test_sub].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    out_csv = SPLITS_DIR / "cp_anemic_splits.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\n[CP-AnemiC] Stratified split saved to {out_csv} ({len(combined)} samples)")
    print(combined.groupby(["split", "who_anemic"]).size().unstack(fill_value=0))
    return combined


def _parse_filename_label_and_subject(filename: str, modality: str):
    clean_name = filename.strip()
    fn_lower = clean_name.lower()
    if "non" in fn_lower:
        is_anemic = 0
    elif "anemic" in fn_lower or "anmeic" in fn_lower:
        is_anemic = 1
    else:
        return None, None

    # Parse subject ID to prevent data leakage (patient-level grouping)
    m = re.search(r"(\d+)", clean_name.split("(")[0])
    if m:
        subj_id = f"{modality}_{is_anemic}_{int(m.group(1)):04d}"
    else:
        m2 = re.search(r"(\d+)", clean_name)
        subj_id = f"{modality}_{is_anemic}_{int(m2.group(1)):04d}" if m2 else f"{modality}_{clean_name}"
    return is_anemic, subj_id


def create_grouped_splits_for_modality(modality: str):
    """
    Creates Group-Stratified 70% Train / 15% Val / 15% Test split for Palm or Fingernail dataset.
    Uses StratifiedGroupKFold to guarantee zero patient/subject overlap between train, val, and test.
    """
    set_seed(RANDOM_SEED)
    folder = RAW_DATA_DIR / modality
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])
    if not files:
        raise ValueError(f"No PNG images found in {folder}")

    records = []
    for f in files:
        label, subj = _parse_filename_label_and_subject(f, modality)
        if label is None or subj is None:
            continue
        img_path = (folder / f).resolve()
        records.append({
            "subject_id": subj,
            "filename": f,
            "image_path": str(img_path),
            "who_anemic": label
        })

    df = pd.DataFrame(records)

    # 20 folds: 14 train (70%), 3 val (15%), 3 test (15%)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=RANDOM_SEED)
    folds = list(sgkf.split(df, df["who_anemic"], df["subject_id"]))

    val_folds = [0, 1, 2]
    test_folds = [3, 4, 5]
    train_folds = list(range(6, 20))

    val_idx = np.concatenate([folds[i][1] for i in val_folds])
    test_idx = np.concatenate([folds[i][1] for i in test_folds])
    train_idx = np.concatenate([folds[i][1] for i in train_folds])

    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    # Leakage assertion
    train_g = set(df[df["split"] == "train"]["subject_id"])
    val_g = set(df[df["split"] == "val"]["subject_id"])
    test_g = set(df[df["split"] == "test"]["subject_id"])
    assert len((train_g & val_g) | (train_g & test_g) | (val_g & test_g)) == 0, f"Subject leakage detected in {modality}!"

    out_csv = SPLITS_DIR / f"{modality}_splits.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[{modality.upper()}] Group-stratified split saved to {out_csv} ({len(df)} images, {df['subject_id'].nunique()} unique subjects)")
    print(df.groupby(["split", "who_anemic"]).size().unstack(fill_value=0))
    return df


def create_palm_splits():
    return create_grouped_splits_for_modality("palm")


def create_fingernail_splits():
    return create_grouped_splits_for_modality("fingernail")


def create_merged_conjunctiva_splits():
    """
    Creates stratified 70% Train / 15% Val / 15% Test split for the MERGED conjunctiva dataset
    (Eyes-defy-anemia 218 images + CP-AnemiC 710 images = 928 images).
    Stratified jointly by source dataset and WHO anemia status.
    """
    set_seed(RANDOM_SEED)

    # 1. Load Eyes-defy-anemia
    eyes_csv = SPLITS_DIR / "conjunctiva_splits.csv"
    if not eyes_csv.exists():
        create_conjunctiva_splits()
    eyes_df = pd.read_csv(eyes_csv)
    eyes_records = []
    for _, r in eyes_df.iterrows():
        if pd.isna(r["who_anemic"]):
            continue
        img_path = r["palpebral_mask_path"] if pd.notna(r.get("palpebral_mask_path")) else r["original_image_path"]
        eyes_records.append({
            "patient_id": f"eyes_{r['patient_id']}",
            "source_dataset": "eyes_defy",
            "image_path": str(Path(img_path).resolve()),
            "is_mask": pd.notna(r.get("palpebral_mask_path")),
            "who_anemic": int(r["who_anemic"]),
            "hb": r.get("hgb"),
            "gender": r.get("gender"),
            "severity": r.get("severity")
        })

    # 2. Load CP-AnemiC
    cp_csv = SPLITS_DIR / "cp_anemic_splits.csv"
    if not cp_csv.exists():
        create_cp_anemic_splits()
    cp_df = pd.read_csv(cp_csv)
    cp_records = []
    for _, r in cp_df.iterrows():
        cp_records.append({
            "patient_id": f"cp_{r['patient_id']}",
            "source_dataset": "cp_anemic",
            "image_path": str(Path(r["image_path"]).resolve()),
            "is_mask": False,
            "who_anemic": int(r["who_anemic"]),
            "hb": r.get("hb"),
            "gender": r.get("gender"),
            "severity": r.get("severity")
        })

    merged = pd.DataFrame(eyes_records + cp_records)

    # Composite stratum: source_dataset + who_anemic
    merged["stratum"] = merged["source_dataset"] + "_" + merged["who_anemic"].astype(str)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED)
    train_idx, temp_idx = next(sss1.split(merged, merged["stratum"]))

    train_data = merged.iloc[train_idx].copy()
    temp_data = merged.iloc[temp_idx].copy()

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    val_sub, test_sub = next(sss2.split(temp_data, temp_data["stratum"]))

    val_data = temp_data.iloc[val_sub].copy()
    test_data = temp_data.iloc[test_sub].copy()

    train_data["split"] = "train"
    val_data["split"] = "val"
    test_data["split"] = "test"

    final_df = pd.concat([train_data, val_data, test_data]).drop(columns=["stratum"]).reset_index(drop=True)
    out_csv = SPLITS_DIR / "conjunctiva_merged_splits.csv"
    final_df.to_csv(out_csv, index=False)
    print(f"\n[MERGED CONJUNCTIVA] Stratified split saved to {out_csv} ({len(final_df)} samples total)")
    print(final_df.groupby(["split", "source_dataset", "who_anemic"]).size().unstack(fill_value=0))
    return final_df


def create_all_splits():
    print("=" * 60)
    print("Creating Splits for All Modalities...")
    print("=" * 60)
    create_conjunctiva_splits()
    create_cp_anemic_splits()
    create_merged_conjunctiva_splits()
    create_palm_splits()
    create_fingernail_splits()
    print("\n[SUCCESS] All modality splits successfully generated!")


if __name__ == "__main__":
    create_all_splits()
