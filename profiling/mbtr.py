"""
Used to profile MBTR.
"""
# import pyximport; pyximport.install()
import numpy as np
from pprint import pprint
from ase.build import bulk
from ase.visualize import view

from describe.descriptors import MBTR

import chronic


def test_bulk_k2():
    system = bulk("NaCl", "rocksalt", 5.64, cubic=True)
    system.wrap()
    # system *= (3, 3, 3)
    # view(system)

    mbtr = MBTR(
        k=[2],
        atomic_numbers=system.get_atomic_numbers(),
        periodic=True,
        grid={"k2": {"min": 0, "max": 3, "n": 20, "sigma": 0.1}},
        weighting={"k2": {"function": lambda x: np.exp(-0.5*x), "threshold": 1e-3}},
        flatten=True
    )
    feat = mbtr.create(system)
    pprint(chronic.timings)

if __name__ == "__main__":
    test_bulk_k2()
