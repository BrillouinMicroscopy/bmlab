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
    # Test that one calibration works
    cm.set_frequencies('0', 0, np.arange(100).reshape(1, -1))

    frequency = cm.get_frequency_by_time(0, 0)
    np.testing.assert_allclose(frequency, 0, atol=1.E-8)
    frequency = cm.get_frequency_by_time(10, 0)
    np.testing.assert_allclose(frequency, 0, atol=1.E-8)

    times = 10 * np.ones((5, 5, 5, 2, 2, 2))
    positions = np.zeros((5, 5, 5, 2, 2, 2))
    positions[:, :, :, 1, :, :] = 50
    frequencies = cm.get_frequency_by_time(times, positions)

    expected = np.zeros((5, 5, 5, 2, 2, 2))
    expected[:, :, :, 1, :, :] = 50
    np.testing.assert_allclose(frequencies, expected, atol=1.E-8)

    # Test that two calibrations work
    cm.set_frequencies('1', 10, np.arange(100).reshape(1, -1) + 10)

    frequency = cm.get_frequency_by_time(0, 0)
    np.testing.assert_allclose(frequency, 0, atol=1.E-8)
    frequency = cm.get_frequency_by_time(10, 0)
    np.testing.assert_allclose(frequency, 10, atol=1.E-8)

    times = 10 * np.ones((5, 5, 5, 2, 2, 2))
    positions = np.zeros((5, 5, 5, 2, 2, 2))
    positions[:, :, :, 1, :, :] = 50
    frequencies = cm.get_frequency_by_time(times, positions)

    expected = 10 * np.ones((5, 5, 5, 2, 2, 2))
    expected[:, :, :, 1, :, :] = 60
    np.testing.assert_allclose(frequencies, expected, atol=1.E-8)

    # Test that three calibrations work
    cm.set_frequencies('2', 30, np.arange(100).reshape(1, -1) + 30)

    frequency = cm.get_frequency_by_time(0, 0)
    np.testing.assert_allclose(frequency, 0, atol=1.E-8)

    frequency = cm.get_frequency_by_time(10, 0)
    np.testing.assert_allclose(frequency, 10, atol=1.E-8)

    frequency = cm.get_frequency_by_time(20, 0)
    np.testing.assert_allclose(frequency, 20, atol=1.E-8)

    frequency = cm.get_frequency_by_time(20, 11)
    np.testing.assert_allclose(frequency, 31, atol=1.E-8)

    # Test that the function also accepts arrays
    times = 10 * np.ones((5, 5, 5, 2, 2, 2))
    times[:, :, :, :, :, 1] = 20
    positions = np.zeros((5, 5, 5, 2, 2, 2))
    positions[:, :, :, 1, :, :] = 50
    frequencies = cm.get_frequency_by_time(times, positions)

    expected = 10 * np.ones((5, 5, 5, 2, 2, 2))
    expected[:, :, :, 0, :, 1] = 20
    expected[:, :, :, 1, :, 0] = 60
    expected[:, :, :, 1, :, 1] = 70
    np.testing.assert_allclose(frequencies, expected, atol=1.E-8)
