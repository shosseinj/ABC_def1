import pytest, torch
pytest.importorskip("pennylane")
from models.qsnn import IrisQSNN

def test_qsnn_shape_and_params():
    m=IrisQSNN(4,4,3)
    x=torch.zeros((2,4),dtype=torch.float32)
    y=m(x)
    assert tuple(y.shape)==(2,3)
    assert m.trainable_parameter_count()==47
