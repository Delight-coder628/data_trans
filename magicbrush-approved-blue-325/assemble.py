"""Reassemble the MagicBrush transfer archive from GitHub-sized parts."""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="default: archive beside this script")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "parts.json").read_text(encoding="utf-8"))
    output = (args.output or root / manifest["archive_name"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    total = 0
    try:
        with temporary.open("wb") as target:
            for item in manifest["parts"]:
                part = root / item["name"]
                if not part.is_file() or part.stat().st_size != item["size_bytes"]:
                    raise ValueError(f"missing or wrong-sized part: {part}")
                with part.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                        total += len(chunk)
        if total != manifest["archive_size_bytes"]:
            raise ValueError(f"wrong assembled size: {total}")
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failed: {bad}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Assembled and ZIP-verified: {output} ({total} bytes)")


if __name__ == "__main__":
    main()
