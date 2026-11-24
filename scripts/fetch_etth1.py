"""Fetch exactly one pinned public ETTh1 artifact, preserving its license locally."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.error
import urllib.request

import pandas as pd

COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
BASE_URL = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{COMMIT}/"
CSV_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
LICENSE_SHA256 = "83cbb5c31415b1d0ee1d20d35dae3fda3a045b85fae9364cc57facc9cd6e0946"


def verified_bytes(payload: bytes, expected: str, label: str) -> bytes:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return payload


def convert_csv(payload: bytes) -> pd.DataFrame:
    source = pd.read_csv(io.BytesIO(payload))
    if not {"date", "OT"}.issubset(source.columns):
        raise ValueError("ETTh1 must contain date and OT")
    return source.loc[:, ["date", "OT"]].rename(columns={"date": "timestamp", "OT": "y"})


def _read(path: Path | None, remote: str) -> bytes:
    if path:
        return path.read_bytes()
    request = urllib.request.Request(BASE_URL + remote, headers={"User-Agent": "chronos-bench/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download pinned ETTh1; CC BY-ND 4.0 data is kept local")
    parser.add_argument("--out", type=Path, default=Path("data/etth1"))
    parser.add_argument("--source", type=Path, help="Optional existing ORIGINAL CSV; checksum remains mandatory")
    parser.add_argument("--license-source", type=Path, help="Optional existing upstream license")
    args = parser.parse_args(argv)
    try:
        raw = verified_bytes(_read(args.source, "ETT-small/ETTh1.csv"), CSV_SHA256, "ETTh1.csv")
        license_bytes = verified_bytes(_read(args.license_source, "LICENSE"), LICENSE_SHA256, "LICENSE")
        converted = convert_csv(raw)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ETTh1.original.csv").write_bytes(raw)
        (args.out / "ETDataset.LICENSE.txt").write_bytes(license_bytes)
        converted.to_csv(args.out / "load.csv", index=False)
        manifest = {"dataset": "ETTh1", "target": "OT", "upstream": "https://github.com/zhouhaoyi/ETDataset",
                    "commit": COMMIT, "raw_url": BASE_URL + "ETT-small/ETTh1.csv", "raw_sha256": CSV_SHA256,
                    "license": "CC BY-ND 4.0", "license_url": BASE_URL + "LICENSE",
                    "license_sha256": LICENSE_SHA256, "rows": len(converted),
                    "transformation": "Local-only: retain date and OT, rename to timestamp and y; no row filtering",
                    "timezone": "Upstream timestamps have no zone; benchmark preserves clock values and interprets them as UTC",
                    "attribution": "Haoyi Zhou et al., Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting, AAAI 2021",
                    "publication_note": "Do not commit original/converted data or pointwise traces. Repository publishes aggregate evaluation metrics only."}
        (args.out / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Verified {len(converted)} rows; local evaluation CSV: {args.out / 'load.csv'}")
        print(f"Raw SHA256: {CSV_SHA256}; upstream license: CC BY-ND 4.0")
    except (ValueError, OSError, urllib.error.URLError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
