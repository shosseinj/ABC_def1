import hashlib
import numpy as np

from scripts.audit_nmnist_attack_protocol_v3_auditable_seed42 import (
    canonical_frames_numpy,
    reference_frames_numpy,
    validate_record_payload,
)


DTYPE = [("x", "i2"), ("y", "i2"), ("t", "i8"), ("p", "i1")]


def events(x=None, y=None, p=None, t=None):
    return np.array(
        list(zip(
            [1, 2, 3] if x is None else x,
            [1, 1, 2] if y is None else y,
            [10, 20, 30] if t is None else t,
            [0, 1, 0] if p is None else p,
        )),
        dtype=DTYPE,
    )


def payload(epsilon_fraction=0.1, adversarial_t=None, adversarial_x=None,
            stored_adversarial_prediction=3, stored_success=True,
            clean_prediction=2, objective_name="true_class_margin"):
    clean = events()
    clean_t = clean["t"].astype(np.float64)
    adv_t = clean_t.copy() if adversarial_t is None else np.asarray(adversarial_t, dtype=np.float64)
    adv_x = clean["x"].copy() if adversarial_x is None else np.asarray(adversarial_x, dtype=np.int16)
    duration = float(clean["t"][-1] - clean["t"][0] + 1)
    epsilon_absolute = epsilon_fraction * duration
    adv = events(x=adv_x.tolist(), t=adv_t.astype(np.int64).tolist())
    clean_margin = 1.0
    adv_margin = -0.5 if stored_success else 1.0
    clean_bins = np.minimum(((clean_t - 10) * 10 / duration).astype(np.int64), 9)
    adv_bins = np.minimum(((adv_t - 10) * 10 / duration).astype(np.int64), 9)
    return {
        "protocol_version": np.asarray("test"),
        "record_key": np.asarray("sample=7;label=2;model=SNN;attack=TEMP-DRIFT-v2;epsilon=0.100000"),
        "sample_id": np.asarray(7, dtype=np.int64),
        "label": np.asarray(2, dtype=np.int64),
        "model": np.asarray("SNN"),
        "attack": np.asarray("TEMP-DRIFT-v2"),
        "epsilon_fraction": np.asarray(epsilon_fraction, dtype=np.float64),
        "epsilon_absolute": np.asarray(epsilon_absolute, dtype=np.float64),
        "bound_tolerance": np.asarray(1e-9, dtype=np.float64),
        "duration": np.asarray(duration, dtype=np.float64),
        "timestamp_domain_min": np.asarray(float(clean["t"][0]), dtype=np.float64),
        "timestamp_domain_max": np.asarray(float(clean["t"][-1]), dtype=np.float64),
        "clean_event_count": np.asarray(len(clean), dtype=np.int64),
        "adversarial_event_count": np.asarray(len(adv), dtype=np.int64),
        "clean_x": clean["x"],
        "clean_y": clean["y"],
        "clean_p": clean["p"],
        "clean_t_native": clean["t"],
        "clean_t": clean_t,
        "adversarial_x": adv_x,
        "adversarial_y": adv["y"],
        "adversarial_p": adv["p"],
        "adversarial_t": adv_t,
        "clean_bins": clean_bins,
        "adversarial_bins": adv_bins,
        "clean_prediction": np.asarray(clean_prediction, dtype=np.int64),
        "adversarial_prediction": np.asarray(stored_adversarial_prediction, dtype=np.int64),
        "stored_attack_success": np.asarray(stored_success),
        "clean_objective": np.asarray(clean_margin, dtype=np.float64),
        "adversarial_objective": np.asarray(adv_margin, dtype=np.float64),
        "clean_margin": np.asarray(clean_margin, dtype=np.float64),
        "adversarial_margin": np.asarray(adv_margin, dtype=np.float64),
        "attack_returned_objective": np.asarray(adv_margin, dtype=np.float64),
        "attack_returned_prediction": np.asarray(stored_adversarial_prediction, dtype=np.int64),
        "objective_name": np.asarray(objective_name),
        "rng_seed": np.asarray(42, dtype=np.int64),
        "runtime_seconds": np.asarray(0.1, dtype=np.float64),
        "model_forward_evaluations": np.asarray(1600, dtype=np.int64),
        "backward_evaluations": np.asarray(0, dtype=np.int64),
        "candidate_evaluations": np.asarray(1600, dtype=np.int64),
        "timestamps_changed": np.asarray(np.count_nonzero(adv_t != clean_t), dtype=np.int64),
        "events_changing_bin": np.asarray(0, dtype=np.int64),
        "frame_l0": np.asarray(0, dtype=np.int64),
        "frame_l1": np.asarray(0.0, dtype=np.float64),
        "frame_l2": np.asarray(0.0, dtype=np.float64),
        "frame_linf": np.asarray(0.0, dtype=np.float64),
        "changed_frame_elements": np.asarray(0, dtype=np.int64),
        "clean_frame_sha256": np.asarray("unused"),
        "adversarial_frame_sha256": np.asarray("unused"),
        "clean_event_sha256": np.asarray("unused"),
        "adversarial_event_sha256": np.asarray("unused"),
    }


def observer_sequence(clean=(2, 1.0, 1.0), adversarial=(3, -0.5, -0.5)):
    responses = iter((clean, adversarial))
    return lambda frame: next(responses)


def test_canonical_numpy_frame_matches_reference():
    sample = events()
    timestamps = sample["t"].astype(np.float64)
    assert np.array_equal(canonical_frames_numpy(sample, timestamps), reference_frames_numpy(sample, timestamps))


def test_valid_record_passes_independent_feasibility(tmp_path):
    record_path = tmp_path / "record.npz"
    values = payload(adversarial_t=[10, 22, 30])
    np.savez_compressed(record_path, **values)
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    result = validate_record_payload(
        values,
        expected_sample_id=7,
        expected_label=2,
        dataset_events=events(),
        dataset_label=2,
        observer=observer_sequence(),
        expected_record_hash=digest,
        record_path=record_path,
    )
    assert result["feasible"]
    assert result["checks"]["epsilon_bound"]
    assert result["checks"]["canonical_frames"]
    assert result["checks"]["stored_predictions"]


def test_zero_control_requires_exact_identity():
    values = payload(epsilon_fraction=0.0, adversarial_t=[10, 20, 30],
                     stored_adversarial_prediction=2, stored_success=False,
                     clean_prediction=2)
    result = validate_record_payload(
        values,
        expected_sample_id=7,
        expected_label=2,
        dataset_events=events(),
        dataset_label=2,
        observer=observer_sequence(adversarial=(2, 1.0, 1.0)),
    )
    assert result["feasible"]
    assert result["checks"]["epsilon_zero_identity"]


def test_auditor_detects_independent_violations():
    values = payload(adversarial_t=[10, 30, 25], adversarial_x=[1, 9, 3],
                     stored_adversarial_prediction=2, stored_success=False)
    result = validate_record_payload(
        values,
        expected_sample_id=7,
        expected_label=2,
        dataset_events=events(),
        dataset_label=2,
        observer=observer_sequence(),
    )
    assert not result["feasible"]
    assert not result["checks"]["x_coordinates"]
    assert not result["checks"]["timestamp_order"]
    assert not result["checks"]["epsilon_bound"]


def test_stored_prediction_mismatch_is_not_trusted():
    values = payload(stored_adversarial_prediction=4, stored_success=False)
    result = validate_record_payload(
        values,
        expected_sample_id=7,
        expected_label=2,
        dataset_events=events(),
        dataset_label=2,
        observer=observer_sequence(),
    )
    assert not result["feasible"]
    assert not result["checks"]["stored_predictions"]
    assert not result["checks"]["attack_success_consistency"]
