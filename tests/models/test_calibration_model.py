from bmlab.models.calibration_model import CalibrationModel

import numpy as np


def test_calibration_model_add_brillouin_region():
    cm = CalibrationModel()
    cm.add_brillouin_region(0, [0.9, 3])
    cm.add_brillouin_region(0, (2, 5))

    brillouin_region_0 = cm.get_brillouin_regions(0)

    assert brillouin_region_0 == [(1, 5)]

    cm.add_brillouin_region(0, (7, 9))

    assert brillouin_region_0 == [(1, 5), (7, 9)]

    cm.add_brillouin_region(0, (2, 3))

    assert brillouin_region_0 == [(1, 5), (7, 9)]


def test_calibration_model_set_brillouin_region():
    cm = CalibrationModel()
    cm.add_brillouin_region(0, [0.9, 3])

    brillouin_region_0 = cm.get_brillouin_regions(0)

    assert brillouin_region_0 == [(1, 3)]


def test_calibration_model_add_rayleigh_region():
    cm = CalibrationModel()
    cm.add_rayleigh_region(0, [0.9, 3])
    cm.add_rayleigh_region(0, (2, 5))

    rayleigh_region_0 = cm.get_rayleigh_regions(0)

    assert rayleigh_region_0 == [(1, 5)]

    cm.add_rayleigh_region(0, (7, 9))

    assert rayleigh_region_0 == [(1, 5), (7, 9)]

    cm.add_rayleigh_region(0, (2, 3))

    assert rayleigh_region_0 == [(1, 5), (7, 9)]


def test_calibration_model_set_rayleigh_region():
    cm = CalibrationModel()
    cm.add_rayleigh_region(0, [0.9, 3])

    rayleigh_region_0 = cm.get_rayleigh_regions(0)

    assert rayleigh_region_0 == [(1, 3)]


def test_get_frequency_by_time():
    cm = CalibrationModel()

    """
    Fill calibration model with values for calibrations
    """
    cm.set_frequencies('0', 0, np.arange(100).reshape(1, -1))
    cm.set_frequencies('1', 10, np.arange(100).reshape(1, -1) + 10)
    cm.set_frequencies('2', 30, np.arange(100).reshape(1, -1) + 30)

    frequency = cm.get_frequency_by_time(0, 0)
    assert frequency == 0

    frequency = cm.get_frequency_by_time(10, 0)
    assert frequency == 10

    frequency = cm.get_frequency_by_time(20, 0)
    assert frequency == 20

    frequency = cm.get_frequency_by_time(20, 11)
    assert frequency == 31
