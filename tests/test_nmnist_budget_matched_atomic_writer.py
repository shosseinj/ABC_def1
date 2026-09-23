from __future__ import annotations

import numpy as np

import scripts.run_nmnist_budget_matched_comparison_v2_seed42 as runner


def payload(audit_passed=True, value=1):
    return {"audit_passed": np.asarray(audit_passed), "value": np.asarray(value)}


def validator(path, expected=None):
    with np.load(path, allow_pickle=False) as z:
        if not bool(z["audit_passed"]):
            return False, None, "audit failed"
        return True, {"value": int(z["value"])}, "valid"


def test_npz_suffix_is_not_silently_doubled(tmp_path):
    requested = tmp_path / "record.npz.tmp"
    np.savez_compressed(requested, value=np.asarray(1))
    assert not requested.exists()
    assert (tmp_path / "record.npz.tmp.npz").exists()


def test_atomic_writer_uses_npz_name_and_reopens_without_pickle(tmp_path):
    destination = tmp_path / "record.npz"
    runner.atomic_save_npz(destination, payload(), expected=(1,), validator=validator)
    assert destination.exists()
    assert not (tmp_path / "record.tmp.npz").exists()
    with np.load(destination, allow_pickle=False) as z:
        assert int(z["value"]) == 1


def test_validation_happens_before_replacement(tmp_path):
    destination = tmp_path / "record.npz"
    destination.write_bytes(b"old")
    seen = []

    def checking_validator(path, expected=None):
        seen.append(path)
        assert destination.read_bytes() == b"old"
        return validator(path, expected)

    runner.atomic_save_npz(destination, payload(value=2), validator=checking_validator)
    assert seen and destination.exists()
    with np.load(destination, allow_pickle=False) as z:
        assert int(z["value"]) == 2


def test_failed_validation_preserves_destination_and_cleans_temp(tmp_path):
    destination = tmp_path / "record.npz"
    destination.write_bytes(b"old")
    try:
        runner.atomic_save_npz(destination, payload(audit_passed=False), validator=validator)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected validation failure")
    assert destination.read_bytes() == b"old"
    assert not (tmp_path / "record.tmp.npz").exists()


def test_valid_destination_is_never_overwritten(tmp_path):
    destination = tmp_path / "record.npz"
    runner.atomic_save_npz(destination, payload(value=7), validator=validator)
    try:
        runner.atomic_save_npz(destination, payload(value=9), validator=validator)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected refusal to overwrite valid record")
    with np.load(destination, allow_pickle=False) as z:
        assert int(z["value"]) == 7
    assert not (tmp_path / "record.tmp.npz").exists()


def test_partial_temp_file_is_ignored_by_resume_classification(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DEST", tmp_path)
    monkeypatch.setattr(runner, "RECORDS", tmp_path / "records")
    key = (2502, "SNN", "PGD", "wallclock_snn", 0.1)
    path = runner.expected_npz_path(*key)
    path.parent.mkdir(parents=True)
    path.with_name(path.stem + ".tmp.npz").write_bytes(b"partial")
    valid, invalid = runner.load_existing_progress([key])
    assert valid == []
    assert invalid == 0
