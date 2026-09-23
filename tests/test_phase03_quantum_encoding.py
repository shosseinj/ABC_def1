import numpy as np
from encoding.quantum import angle_encode, phase_encode

def test_angle_encoding():
    t=np.array([0,25,50,100],float)
    assert np.allclose(angle_encode(t,100),[0,np.pi/8,np.pi/4,np.pi/2])

def test_phase_encoding():
    theta,phi=phase_encode([0,50,100],100)
    assert np.allclose(theta,np.pi/4)
    assert np.allclose(phi,[0,np.pi,2*np.pi])
