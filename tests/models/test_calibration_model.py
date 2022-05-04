from bmlab.models.calibration_model import CalibrationModel,\
    FitSet, BrillouinFit

import numpy as np


def test_calibration_model_add_brillouin_region():
    cm = CalibrationModel()

    brillouin_region = cm.get_brillouin_regions(0)
    assert brillouin_region == []

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

    cm.set_brillouin_region(0, 0, (2, 5))

    assert brillouin_region_0 == [(2, 5)]

    cm.set_brillouin_region(0, 1, (4, 7))

    assert brillouin_region_0 == [(2, 5), (4, 7)]

    cm.set_brillouin_region(0, 3, (8, 9))

    assert brillouin_region_0 == [(2, 5), (4, 7), (8, 9)]


def test_calibration_model_clear_brillouin_region():
    cm = CalibrationModel()
    cm.add_brillouin_region(0, [0.9, 3])

    brillouin_region_0 = cm.get_brillouin_regions(0)

    assert brillouin_region_0 == [(1, 3)]

    cm.clear_brillouin_regions(0)
    brillouin_region_0 = cm.get_brillouin_regions(0)

    assert brillouin_region_0 == []


def test_calibration_model_add_rayleigh_region():
    cm = CalibrationModel()

    rayleigh_region = cm.get_rayleigh_regions(0)
    assert rayleigh_region == []

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

    cm.set_rayleigh_region(0, 0, (2, 5))

    assert rayleigh_region_0 == [(2, 5)]

    cm.set_rayleigh_region(0, 1, (4, 7))

    assert rayleigh_region_0 == [(2, 5), (4, 7)]

    cm.set_rayleigh_region(0, 3, (8, 9))

    assert rayleigh_region_0 == [(2, 5), (4, 7), (8, 9)]


def test_calibration_model_clear_rayleigh_region():
    cm = CalibrationModel()
    cm.add_rayleigh_region(0, [0.9, 3])

    rayleigh_region_0 = cm.get_rayleigh_regions(0)

    assert rayleigh_region_0 == [(1, 3)]

    cm.clear_rayleigh_regions(0)
    rayleigh_region_0 = cm.get_rayleigh_regions(0)

    assert rayleigh_region_0 == []


def test_get_frequencies_by_calib_key():
    cm = CalibrationModel()

    frequencies0 = np.arange(100).reshape(1, -1)
    cm.set_frequencies('0', 0, list(frequencies0))

    frequencies1 = np.arange(100).reshape(1, -1) + 10
    cm.set_frequencies('1', 10, list(frequencies1))

    assert (cm.get_frequencies_by_calib_key('0') == frequencies0).all()
    assert (cm.get_frequencies_by_calib_key('1') == frequencies1).all()

    cm.clear_frequencies('0')
    assert cm.get_frequencies_by_calib_key('0') is None


def test_get_frequency_by_calib_key():
    cm = CalibrationModel()

    frequencies0 = list(np.arange(100).reshape(1, -1))
    cm.set_frequencies('0', 0, frequencies0)

    frequencies1 = list(np.arange(100).reshape(1, -1) + 10)
    cm.set_frequencies('1', 10, frequencies1)

    assert cm.get_frequency_by_calib_key(70.7, '0') == 70.7
    assert cm.get_frequency_by_calib_key(80, '1') == 90

    cm.clear_frequencies('0')
    assert cm.get_frequency_by_calib_key(70.7, '0') is None


def test_get_frequencies_by_time():
    cm = CalibrationModel()

    """
    Fill calibration model with values for calibrations
    """
    # Test that one calibration works
    frequencies0 = np.arange(100)
    cm.set_frequencies('0', 0, list(frequencies0.reshape(1, -1)))

    frequencies = cm.get_frequencies_by_time(0)
    np.testing.assert_allclose(frequencies, frequencies0, atol=1.E-8)

    frequencies = cm.get_frequencies_by_time(10)
    np.testing.assert_allclose(frequencies, frequencies0, atol=1.E-8)

    frequencies1 = np.arange(100) + 10
    cm.set_frequencies('1', 10, list(frequencies1.reshape(1, -1)))

    frequencies = cm.get_frequencies_by_time(0)
    np.testing.assert_allclose(frequencies, frequencies0, atol=1.E-8)

    frequencies = cm.get_frequencies_by_time(10)
    np.testing.assert_allclose(frequencies, frequencies1, atol=1.E-8)


def test_get_frequency_by_time():
    cm = CalibrationModel()

    """
    Fill calibration model with values for calibrations
    """
    # Test that one calibration works
    cm.set_frequencies('0', 0, list(np.arange(100).reshape(1, -1)))

    frequency = cm.get_frequency_by_time(0, 0)
    np.testing.assert_allclose(frequency, 0, atol=1.E-8)
    frequency = cm.get_frequency_by_time(10, 0)
    np.testing.assert_allclose(frequency, 0, atol=1.E-8)

    # Ensure we don't throw a hard error in case our requested value
    # is outside the interpolation range
    frequency = cm.get_frequency_by_time(0, -1)
    np.testing.assert_array_equal(frequency, np.nan)
    frequency = cm.get_frequency_by_time(0, 100)
    np.testing.assert_array_equal(frequency, np.nan)

    times = 10 * np.ones((5, 5, 5, 2, 2, 2))
    positions = np.zeros((5, 5, 5, 2, 2, 2))
    positions[:, :, :, 1, :, :] = 50
    frequencies = cm.get_frequency_by_time(times, positions)

    expected = np.zeros((5, 5, 5, 2, 2, 2))
    expected[:, :, :, 1, :, :] = 50
    np.testing.assert_allclose(frequencies, expected, atol=1.E-8)

    # Test that two calibrations work
    cm.set_frequencies('1', 10, list(np.arange(100).reshape(1, -1) + 10))

    frequency = cm.get_frequency_by_time(0, 0)
    np.testing.assert_allclose(frequency, 0, atol=1.E-8)
    frequency = cm.get_frequency_by_time(10, 0)
    np.testing.assert_allclose(frequency, 10, atol=1.E-8)

    # Ensure we don't throw a hard error in case our requested value
    # is outside the interpolation range
    frequency = cm.get_frequency_by_time(0, -1)
    np.testing.assert_array_equal(frequency, np.nan)
    frequency = cm.get_frequency_by_time(0, 100)
    np.testing.assert_array_equal(frequency, np.nan)

    times = 10 * np.ones((5, 5, 5, 2, 2, 2))
    positions = np.zeros((5, 5, 5, 2, 2, 2))
    positions[:, :, :, 1, :, :] = 50
    frequencies = cm.get_frequency_by_time(times, positions)

    expected = 10 * np.ones((5, 5, 5, 2, 2, 2))
    expected[:, :, :, 1, :, :] = 60
    np.testing.assert_allclose(frequencies, expected, atol=1.E-8)

    # Test that three calibrations work
    cm.set_frequencies('2', 30, list(np.arange(100).reshape(1, -1) + 30))

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


def test_get_position_by_time():
    cm = CalibrationModel()

    frequencies0 = list(np.arange(100).reshape(1, -1))
    cm.set_frequencies('0', 0, frequencies0)

    frequencies1 = list(np.arange(100).reshape(1, -1) + 10)
    cm.set_frequencies('1', 10, frequencies1)

    positions = cm.get_position_by_time(0, [10])
    assert positions == [10]
    positions = cm.get_position_by_time(0, [10.5])
    assert positions == [10.5]
    positions = cm.get_position_by_time(5, [10])
    assert positions == [5]


def test_RayleighFitSet_average_fits():
    cm = CalibrationModel()

    cm.add_rayleigh_fit('0', 0, 0,
                        10, 1, 300, 100)
    cm.add_rayleigh_fit('0', 0, 1,
                        12, 1, 300, 100)

    w0_avg = cm.rayleigh_fits.average_fits('0', 0)

    assert w0_avg == 11.0


def test_BrillouinFitSet_average_fits():
    cm = CalibrationModel()

    cm.add_brillouin_fit('0', 0, 0,
                         10, 1, 300, 100)
    cm.add_brillouin_fit('0', 0, 1,
                         12, 1, 300, 100)

    w0_avg = cm.brillouin_fits.average_fits('0', 0)

    assert w0_avg == 11.0


def test_fitset():
    fit_set = FitSet()

    assert fit_set.make_key('1', 0, 0) == '1::0::0'
    assert fit_set.make_key('10', 0, 0) == '10::0::0'
    assert fit_set.split_key('1::0::0') == ['1', 0, 0]
    assert fit_set.split_key('10::0::0') == ['10', 0, 0]

    fit0 = BrillouinFit('1', 1, 4, 11., 12., 13., 14.)
    fit_set.add_fit(fit0)
    fit1 = BrillouinFit('10', 3, 4, 11., 12., 13., 14.)
    fit_set.add_fit(fit1)

    assert fit_set.get_fit('1', 1, 4) == fit0
    assert fit_set.get_fit('1', 2, 4) is None
    assert fit_set.get_fit('10', 3, 4) == fit1
    assert fit_set.get_fit('2', 3, 4) is None

    fit_set.clear('1')

    assert fit_set.get_fit('1', 1, 4) is None
    assert fit_set.get_fit('10', 3, 4) == fit1

    fit_set.clear('10')

    assert fit_set.get_fit('10', 3, 4) is None
