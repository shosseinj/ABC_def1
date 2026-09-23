import numpy as np
from experiments.iris.data import load_iris_splits

def test_iris_shapes_and_range():
    Xtr,Xv,Xte,ytr,yv,yte,scaler=load_iris_splits()
    assert Xtr.shape[1] == 4
    assert len(set(ytr)) == 3
    for split in (Xtr, Xv, Xte):
        assert np.all(split >= 0.0)
        assert np.all(split <= 1.0)


def test_scaler_is_fitted_on_training_split_only():
    Xtr, Xv, Xte, *_ , scaler = load_iris_splits()
    assert np.allclose(scaler.data_min_, [4.3, 2.0, 1.1, 0.1])
    assert np.allclose(scaler.data_max_, [7.9, 3.9, 6.9, 2.5])
    assert Xtr.min() == 0.0 and Xtr.max() == 1.0
