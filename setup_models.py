"""Fetch the FahadPrimeX weights from the GitHub release and assemble them.

The full checkpoint is shipped as split parts (GitHub release asset size limit).
This script downloads the parts, concatenates them and verifies SHA-256 before
writing models/FahadPrimeX/model.safetensors.

Standard library only — no extra dependencies.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/Fahadub/-FahadPrimeX/releases/download/v1.0.0"
TARGET = Path(__file__).resolve().parent / "models" / "FahadPrimeX" / "model.safetensors"

PARTS = [
    # filename, sha256, size in bytes
    ("model.safetensors.part1", "77ff20853f041deb751f72b065a2ead5da8acd534f6711a60f23890345cca620", 1900000000),
    ("model.safetensors.part2", "ddbba4ee875e910844d64004fd994468c91c174477c37867ea321adb2233562f", 440697936),
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def fetch(url: str, dest: Path, expected_sha: str, expected_size: int) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"  {dest.name}: already downloaded, skipping")
        return
    print(f"  downloading {dest.name} ({human(expected_size)}) ...")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    hasher = hashlib.sha256()
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        done = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            hasher.update(chunk)
            done += len(chunk)
            if done % (50 * 1024 * 1024) < len(chunk):
                print(f"    {human(done)} / {human(expected_size)}")
    if done != expected_size:
        tmp.unlink(missing_ok=True)
        sys.exit(f"size mismatch for {dest.name}: got {done}, expected {expected_size}")
    digest = hasher.hexdigest()
    if digest != expected_sha:
        tmp.unlink(missing_ok=True)
        sys.exit(f"checksum mismatch for {dest.name}: got {digest}")
    tmp.replace(dest)
    print(f"  {dest.name}: ok")


def main() -> None:
    if TARGET.exists():
        print("model.safetensors already present — nothing to do.")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading FahadPrimeX weights (split release assets) ...")
    for name, sha, size in PARTS:
        fetch(f"{RELEASE}/{name}", TARGET.parent / name, sha, size)
    print("Assembling model.safetensors ...")
    hasher = hashlib.sha256()
    with open(TARGET, "wb") as out:
        for name, _, size in PARTS:
            part = TARGET.parent / name
            with open(part, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    hasher.update(chunk)
    total = sum(p[2] for p in PARTS)
    if TARGET.stat().st_size != total:
        sys.exit("assembled file size mismatch")
    print(f"Done: {TARGET} ({human(total)})  sha256={hasher.hexdigest()}")
    for name, _, _ in PARTS:
        (TARGET.parent / name).unlink(missing_ok=True)
    print("Cleaned up part files.")


if __name__ == "__main__":
    main()
