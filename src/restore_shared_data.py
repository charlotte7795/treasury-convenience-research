"""Verify the published research snapshot and restore paths expected by its scripts."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for folder in ["cleaned", "tables", "figures", "report", "final"]:
        (ROOT / "output" / folder).mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        source = ROOT / item["path"]
        if file_hash(source) != item["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {source}")
        if not item.get("restore_to"):
            continue
        target = ROOT / item["restore_to"]
        if target.exists():
            if file_hash(target) == item["restored_sha256"]:
                continue
            raise FileExistsError(f"Refusing to replace different local data: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        if file_hash(target) != item["restored_sha256"]:
            raise ValueError(f"Restored checksum mismatch: {target}")
        print(f"Restored {item['restore_to']}")
    print(f"Verified {len(manifest['files'])} shared files; snapshot {manifest['research_snapshot']}.")


if __name__ == "__main__":
    main()
