import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare_metadata import process_eyes_defy_anemia
from src.data.create_splits import (
    create_conjunctiva_splits,
    create_cp_anemic_splits,
    create_palm_splits,
    create_fingernail_splits,
)

def main():
    print("=" * 60)
    print("Step 1: Parsing metadata for Eyes-Defy-Anemia...")
    print("=" * 60)
    process_eyes_defy_anemia()

    print("\n" + "=" * 60)
    print("Step 2: Creating stratified splits for all modalities...")
    print("=" * 60)
    print("\n>>> 2.1 Conjunctiva (Eyes-Defy-Anemia)")
    create_conjunctiva_splits()

    print("\n>>> 2.2 Conjunctiva (CP-AnemiC Ghana)")
    create_cp_anemic_splits()

    print("\n>>> 2.3 Palm (Ghana)")
    create_palm_splits()

    print("\n>>> 2.4 Fingernails (Ghana)")
    create_fingernail_splits()

    print("\n" + "=" * 60)
    print("[SUCCESS] All metadata and splits successfully created in data/splits/")
    print("=" * 60)

if __name__ == "__main__":
    main()
