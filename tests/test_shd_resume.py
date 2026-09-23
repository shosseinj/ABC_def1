from scripts.resume_shd_snn_longer import incumbent_from_history


def test_resume_incumbent_uses_accuracy_loss_then_earliest_epoch():
    rows = [
        {"epoch": "1", "validation_accuracy": "0.5", "validation_loss": "1.0"},
        {"epoch": "2", "validation_accuracy": "0.6", "validation_loss": "1.2"},
        {"epoch": "3", "validation_accuracy": "0.6", "validation_loss": "1.1"},
        {"epoch": "4", "validation_accuracy": "0.5", "validation_loss": "0.9"},
    ]
    best, stale = incumbent_from_history(rows)
    assert best["epoch"] == "3"
    assert stale == 1
