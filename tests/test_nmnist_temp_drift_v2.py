import numpy as np
import torch

from scripts.run_nmnist_temp_drift_v2_seed42 import objective_values, true_class_margin
from scripts.run_nmnist_common_attack_protocol_seed42 import project


def test_lower_objective_means_lower_true_class_margin():
    label=0
    high_confidence=torch.tensor([[5.0,1.0,0.0]])
    low_confidence=torch.tensor([[1.2,1.0,0.0]])
    misclassified=torch.tensor([[0.5,1.0,0.0]])
    values=torch.cat([objective_values(x,label) for x in (high_confidence,low_confidence,misclassified)])
    assert values.tolist()==[4.0,pytest.approx(0.2),-0.5]
    assert values[2] < values[1] < values[0]


def test_negative_objective_implies_competitor_exceeds_true_class():
    logits=torch.tensor([[0.5,1.0,-2.0],[2.0,1.0,0.0]])
    margins=true_class_margin(logits,0)
    assert margins[0] < 0 and logits[0].argmax().item()!=0
    assert margins[1] > 0 and logits[1].argmax().item()==0


def test_argmin_selects_most_adversarial_candidate():
    logits=torch.tensor([[4.0,0.0],[1.1,1.0],[0.2,1.0]])
    assert int(torch.argmin(objective_values(logits,0)))==2


def test_timestamp_projection_preserves_bound_and_order():
    clean=np.asarray([0.0,2.0,4.0,6.0]); epsilon=1.0
    attacked=project(np.asarray([1.0,-5.0,9.0,5.5]),clean,epsilon)
    assert np.all(np.diff(attacked)>=0)
    assert np.all(np.abs(attacked-clean)<=epsilon)


import pytest
