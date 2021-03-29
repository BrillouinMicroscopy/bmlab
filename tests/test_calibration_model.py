import numpy as np

from bmlab.models.calibration_model import CalibrationModel


def test_calibration_model_add_brillouin_region():
    cm = CalibrationModel()
    cm.add_brillouin_region(0, [1, 3])
    cm.add_brillouin_region(0, [2, 5])

    brillouin_region_0 = cm.get_brillouin_regions(0)

    assert brillouin_region_0 == [(1, 5)]

    cm.add_brillouin_region(0, (7, 9))

    assert brillouin_region_0 == [(1, 5), (7, 9)]