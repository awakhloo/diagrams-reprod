from scs.backend import xp, asnumpy, to_device
import numpy as np


def test_roundtrip():
    a = np.arange(6.0).reshape(2, 3)
    b = asnumpy(to_device(a))
    assert np.array_equal(a, b)
    assert xp is not None
