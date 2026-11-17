"""Download the CDC Diabetes Health Indicators dataset from the UCI repository.

Usage (from the project root):
    python src/download_data.py

Source: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
The file is saved to data/raw/, which is not committed to Git.
"""

from pathlib import Path
from urllib.request import urlretrieve

URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "cdc_diabetes_health_indicators.csv"


def main() -> None:
    if DEST.exists():
        print(f"Already downloaded: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    urlretrieve(URL, DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
