import hashlib

import pytest

from scripts.fetch_etth1 import convert_csv, verified_bytes


def test_upstream_checksum_is_enforced_before_processing():
    original = b"date,OT,other\n2025-01-01 00:00:00,42,999\n"
    digest = hashlib.sha256(original).hexdigest()
    assert verified_bytes(original, digest, "fixture") == original
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verified_bytes(original + b"corrupted", digest, "fixture")


def test_conversion_does_not_filter_or_modify_target_values():
    original = b"date,OT,other\n2025-01-01 00:00:00,42,999\n2025-01-01 01:00:00,43,888\n"
    converted = convert_csv(original)
    assert list(converted.columns) == ["timestamp", "y"]
    assert converted.y.tolist() == [42, 43]
    assert len(converted) == 2
    with pytest.raises(ValueError, match="date and OT"):
        convert_csv(b"date,load\n2025-01-01,5\n")
