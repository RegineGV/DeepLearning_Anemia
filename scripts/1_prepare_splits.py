import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare_metadata import process_eyes_defy_anemia
from src.data.create_splits import create_conjunctiva_splits

def main():
    print("=" * 60)
    print("Step 1: Parsing metadata and generating clinical labels...")
    print("=" * 60)
    process_eyes_defy_anemia()

    print("\n" + "=" * 60)
    print("Step 2: Creating stratified 70/15/15 train/val/test splits...")
    print("=" * 60)
    create_conjunctiva_splits()

    print("\n[SUCCESS] Metadata and splits successfully created in data/splits/")

if __name__ == "__main__":
    main()
