import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_DIR,
    SPLITS_DIR,
    WHO_MALE_THRESHOLD,
    WHO_FEMALE_THRESHOLD,
)

def clean_hgb_value(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ["_", "", "nan", "None"]:
        return np.nan
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan

def determine_anemia_who(row):
    hgb = row["hgb"]
    if pd.isna(hgb):
        return np.nan
    gender = str(row["gender"]).strip().upper()
    if gender == "M":
        return 1 if hgb < WHO_MALE_THRESHOLD else 0
    else:  # Female or default
        return 1 if hgb < WHO_FEMALE_THRESHOLD else 0

def determine_severity(row):
    hgb = row["hgb"]
    if pd.isna(hgb):
        return "Unknown"
    gender = str(row["gender"]).strip().upper()
    anemic = determine_anemia_who(row)
    if anemic == 0:
        return "Normal"
    
    # WHO Severity Guidelines
    if hgb < 8.0:
        return "Severe"
    elif hgb < 11.0:
        return "Moderate"
    else:
        return "Mild"

def find_file_by_patterns(folder, include_patterns, exclude_patterns=None, extension=None):
    if not os.path.exists(folder):
        return None
    files = os.listdir(folder)
    for f in sorted(files):
        fl = f.lower()
        if extension and not fl.endswith(extension):
            continue
        if any(exc in fl for exc in (exclude_patterns or [])):
            continue
        if all(inc in fl for inc in include_patterns):
            return os.path.join(folder, f)
    return None

def process_eyes_defy_anemia():
    """
    Parses Eyes-defy-anemia dataset across Italy and India cohorts.
    Returns a unified DataFrame with validated paths, demographic, and clinical data.
    """
    base_dir = RAW_DATA_DIR / "eyes_defy_anemia"
    records = []

    for country in ["Italy", "India"]:
        country_dir = base_dir / country
        excel_path = country_dir / f"{country}.xlsx"
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel metadata missing: {excel_path}")

        df = pd.read_excel(excel_path)
        
        # Standardize columns
        df.columns = [c.strip() for c in df.columns]
        
        for _, row in df.iterrows():
            subject_num = int(row["Number"])
            folder = country_dir / str(subject_num)
            
            if not folder.exists():
                print(f"Warning: Folder for {country} subject {subject_num} not found.")
                continue

            files = os.listdir(folder)
            
            # 1. Original JPG
            jpgs = [f for f in files if f.lower().endswith(".jpg")]
            orig_path = str(folder / jpgs[0]) if jpgs else None

            # 2. Palpebral conjunctiva mask (handle typos like 'palplebral' or 'papebral', ignore duplicates '(1)')
            palp_candidates = [
                f for f in files 
                if ("palp" in f.lower() or "papebral" in f.lower()) 
                and "forniceal" not in f.lower() 
                and f.lower().endswith(".png")
                and "(1)" not in f
            ]
            palp_path = str(folder / palp_candidates[0]) if palp_candidates else None

            # 3. Forniceal conjunctiva mask
            forn_candidates = [
                f for f in files 
                if "forniceal" in f.lower() 
                and "palp" not in f.lower() 
                and "papebral" not in f.lower()
                and f.lower().endswith(".png")
                and "(1)" not in f
            ]
            forn_path = str(folder / forn_candidates[0]) if forn_candidates else None

            # 4. Combined (forniceal + palpebral)
            comb_candidates = [
                f for f in files 
                if "forniceal" in f.lower() 
                and ("palp" in f.lower() or "papebral" in f.lower()) 
                and f.lower().endswith(".png")
                and "(1)" not in f
            ]
            comb_path = str(folder / comb_candidates[0]) if comb_candidates else None

            hgb_val = clean_hgb_value(row.get("Hgb"))
            gender_val = str(row.get("Gender")).strip().upper() if pd.notna(row.get("Gender")) else "Unknown"
            age_val = int(row.get("Age")) if pd.notna(row.get("Age")) else np.nan

            rec = {
                "patient_id": f"{country}_{subject_num:03d}",
                "country": country,
                "subject_number": subject_num,
                "gender": gender_val,
                "age": age_val,
                "hgb": hgb_val,
                "original_image_path": orig_path,
                "palpebral_mask_path": palp_path,
                "forniceal_mask_path": forn_path,
                "combined_mask_path": comb_path,
            }
            records.append(rec)

    full_df = pd.DataFrame(records)
    full_df["who_anemic"] = full_df.apply(determine_anemia_who, axis=1)
    full_df["severity"] = full_df.apply(determine_severity, axis=1)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SPLITS_DIR / "eyes_defy_anemia_metadata.csv"
    full_df.to_csv(out_csv, index=False)
    print(f"Successfully generated {out_csv} with {len(full_df)} total records.")
    return full_df

if __name__ == "__main__":
    df = process_eyes_defy_anemia()
    print("\nSummary of metadata:")
    print(df.groupby(["country", "who_anemic"], dropna=False).size().unstack(fill_value=0))
