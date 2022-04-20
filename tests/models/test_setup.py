from bmlab.models.setup import Calibration
import numpy as np


def test_calibration_2_samples():
    shift_methanol = 3.6e9
    shift_water = 5.0e9

    calibration = Calibration(num_brillouin_samples=2,
                              shift_methanol=shift_methanol,
                              shift_water=shift_water)

    np.testing.assert_array_equal(
        calibration.shifts,
        [0,
         shift_methanol,
         shift_water,
         -1*shift_water,
         -1*shift_methanol,
         0]
    )
    np.testing.assert_array_equal(
        calibration.orders,
        [0, 0, 0, 1, 1, 1]
    )


def test_calibration_1_samples():
    shift_methanol = 3.6e9
    shift_water = 5.0e9

    calibration = Calibration(num_brillouin_samples=1,
                              shift_methanol=shift_methanol,
                              shift_water=shift_water)

    np.testing.assert_array_equal(
        calibration.shifts,
        [0,
         shift_methanol,
         -1*shift_methanol,
         0]
    )
    np.testing.assert_array_equal(
        calibration.orders,
        [0, 0, 1, 1]
    )


def test_calibration_set_shift():
    shift_methanol = 3.6e9
    shift_water = 5.0e9

    calibration = Calibration(num_brillouin_samples=2,
                              shift_methanol=shift_methanol,
                              shift_water=shift_water)

    shift_methanol_new = 3.0e9
    calibration.set_shift_methanol(shift_methanol_new)

    np.testing.assert_array_equal(
        calibration.shifts,
        [0,
         shift_methanol_new,
         shift_water,
         -1*shift_water,
         -1*shift_methanol_new,
         0]
    )
