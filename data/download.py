"""Prepare official Kaggle competition files without mutating their contents."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

EXPECTED_FILES = {"train.csv", "test.csv", "test_labels.csv", "sample_submission.csv"}


def extract_archive(archive: str | Path, destination: str | Path) -> None:
    archive_path, destination_path = Path(archive), Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    destination_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as bundle:
        names = {Path(name).name for name in bundle.namelist()}
        missing = EXPECTED_FILES.difference(names)
        if missing:
            raise ValueError(f"Archive is not the official Jigsaw bundle; missing {sorted(missing)}")
        for member in bundle.infolist():
            name = Path(member.filename).name
            if name in EXPECTED_FILES:
                with bundle.open(member) as source, (destination_path / name).open("wb") as target:
                    shutil.copyfileobj(source, target)


def kaggle_download(destination: str | Path) -> None:
    """Download through Kaggle CLI when users have independently configured credentials."""
    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI is unavailable. Install `kaggle` or pass --archive.")
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "jigsaw-toxic-comment-classification-challenge", "-p", str(destination_path)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    if args.archive:
        extract_archive(args.archive, args.destination)
    else:
        kaggle_download(args.destination)


if __name__ == "__main__":
    main()

