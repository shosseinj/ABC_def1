import numpy as np
from encoding.ttfs import ttfs_encode

def test_ttfs_exact():
    x=np.array([0.2,0.5,0.8,1.0])
    assert np.allclose(ttfs_encode(x,T=100),[80,50,20,0])

def test_ttfs_bounds():
    out=ttfs_encode([0,1],T=10)
    assert np.all((out>=0)&(out<=10))
