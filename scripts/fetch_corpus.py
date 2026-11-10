"""Download a few public FAA advisory circulars into data/corpus/public/.

The synthetic files from make_synthetic_docs.py are enough for the demo.
These public PDFs make the corpus look like real source material. A failed
download is skipped, so a network problem does not break the demo.
"""
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "corpus" / "public"

# Small, stable FAA advisory circulars. Each URL was checked when this file was written.
SOURCES = [
    (
        "AC_20-62E.pdf",
        "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_20-62E.pdf",
    ),
    (
        "AC_43-9C.pdf",
        "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_43-9C.pdf",
    ),
    (
        "AC_120-16F.pdf",
        "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-16F.pdf",
    ),
]


def fetch(directory: Path = OUT_DIR, force: bool = False) -> list[Path]:
    """Download any missing files. Return the paths that are present afterwards."""
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, url in SOURCES:
        target = directory / name
        if target.exists() and not force:
            print(f"keep  {name} ({target.stat().st_size} bytes)")
            saved.append(target)
            continue
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": "docintel-lab/0.1"})
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"skip  {name}: {exc}")
            continue
        if "pdf" not in response.headers.get("content-type", "").lower() and not response.content.startswith(b"%PDF"):
            print(f"skip  {name}: response is not a PDF")
            continue
        target.write_bytes(response.content)
        print(f"saved {name} ({len(response.content)} bytes)")
        saved.append(target)
    return saved


def main() -> None:
    force = "--force" in sys.argv
    saved = fetch(force=force)
    print(f"{len(saved)} public file(s) in {OUT_DIR}")


if __name__ == "__main__":
    main()
