import pathlib

import numpy as np

from bmlab.controllers import CalibrationController
from bmlab.controllers import EvaluationController, ExtractionController
from bmlab.controllers import calculate_derived_values
from bmlab.models import ExtractionModel


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_optimize_points_in_extraction_model(mocker):

    imgs = np.zeros((1, 100, 100), dtype=int)
    imgs[0, 19:22, 19:22] = 1
    imgs[0, 79:82, 79:82] = 1

    em = ExtractionModel()

    mocker.patch('bmlab.session.Session.extraction_model', return_value=em)
    mocker.patch('bmlab.session.Session.get_calibration_image',
                 return_value=imgs)
    mocker.patch('bmlab.session.Session.get_calibration_time',
                 return_value=0)

    ec = ExtractionController()
    ec.add_point('0', (15, 15))
    ec.add_point('0', (5, 85))

    opt_points = em.get_points('0')

    assert (15, 15) == opt_points[0]
    assert (5, 85) == opt_points[1]

    ec.set_point('0', 1, (75, 85))

    ec.optimize_points('0', radius=10)
    opt_points = em.get_points('0')

    assert (19, 20) == opt_points[0]
    assert (79, 80) == opt_points[1]


def test_distance_point_to_line():
    ec = ExtractionController()

    np.testing.assert_almost_equal(0, ec.distance_point_to_line(
        (0.5, 0.5), (1, 1), (0, 0)
    ))
    np.testing.assert_almost_equal(0.5, ec.distance_point_to_line(
        (0.5, 0.5), (0, 0), (0, 1)
    ))
    np.testing.assert_almost_equal(np.sqrt(0.5), ec.distance_point_to_line(
        (0, 1), (1, 1), (0, 0)
    ))


def test_find_points_in_extraction_model(mocker):

    imgs = np.ones((1, 400, 200), dtype=int)
    imgs[0, 19:22, 19:22] = 5
    imgs[0, 379:382, 19:22] = 5
    imgs[0, 19:22, 179:182] = 5
    imgs[0, 379:382, 179:182] = 5

    em = ExtractionModel()

    mocker.patch('bmlab.session.Session.extraction_model', return_value=em)
    mocker.patch('bmlab.session.Session.get_calibration_image',
                 return_value=imgs)
    mocker.patch('bmlab.session.Session.get_calibration_time',
                 return_value=0)

    ec = ExtractionController()

    ec.find_points('0', 1, 4, 10)
    opt_points = em.get_points('0')

    assert len(opt_points) == 2
    assert (19, 180) == opt_points[0]
    assert (379, 20) == opt_points[1]


def test_calibrate():

    calib_key = '0'
    calibration_controller = CalibrationController()
    calibration_controller.calibrate(calib_key)


def test_evaluate():
    evaluationcontroller = EvaluationController()
    evaluationcontroller.evaluate()


def test_calculate_derived_values_equal_region_count():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    """
    Test calculation in case of same number
    Brillouin and Rayleigh regions
    """
    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })

    evm.results['brillouin_peak_position'][:] = 1
    evm.results['rayleigh_peak_position'][:] = 3

    calculate_derived_values()

    assert (evm.results['brillouin_shift'] == 2).all()

    evm.results['brillouin_peak_position'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position'][:, :, :, :, 1, :] = 4
    evm.results['rayleigh_peak_position'][:, :, :, :, 0, :] = 3
    evm.results['rayleigh_peak_position'][:, :, :, :, 1, :] = 8

    calculate_derived_values()

    assert (evm.results['brillouin_shift'][:, :, :, :, 0, :] == 2).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 1, :] == 4).all()


def test_calculate_derived_values_equal_region_count_nr_peaks_2():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    """
    Test calculation in case of same number
    Brillouin and Rayleigh regions
    """
    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 2,
        'nr_rayleigh_regions': 2,
    })

    evm.results['brillouin_peak_position'][:] = 1
    evm.results['rayleigh_peak_position'][:] = 3

    calculate_derived_values()

    assert (evm.results['brillouin_shift'] == 2).all()

    evm.results['brillouin_peak_position'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position'][:, :, :, :, 1, :] = 4
    evm.results['brillouin_peak_position'][:, :, :, :, 0, 1] = 2
    evm.results['brillouin_peak_position'][:, :, :, :, 1, 1] = 5
    evm.results['rayleigh_peak_position'][:, :, :, :, 0, :] = 3
    evm.results['rayleigh_peak_position'][:, :, :, :, 1, :] = 8

    calculate_derived_values()

    assert (evm.results['brillouin_shift'][:, :, :, :, 0, 0] == 2).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 1, 0] == 4).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 0, 1] == 1).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 1, 1] == 3).all()


def test_calculate_derived_values_different_region_count():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    """
    Test calculation in case of more Rayleigh than
    Brillouin regions
    """
    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 3,
    })

    evm.results['brillouin_peak_position'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position'][:, :, :, :, 1, :] = 4
    evm.results['rayleigh_peak_position'][:, :, :, :, 0, :] = 5
    evm.results['rayleigh_peak_position'][:, :, :, :, 1, :] = 9
    evm.results['rayleigh_peak_position'][:, :, :, :, 2, :] = 10

    psm = evc.session.peak_selection_model()

    psm.add_brillouin_region((1, 2))
    psm.add_brillouin_region((4, 5))

    psm.add_rayleigh_region((0, 1))
    psm.add_rayleigh_region((6, 7))
    psm.add_rayleigh_region((8, 9))

    calculate_derived_values()

    assert (evm.results['brillouin_shift'][:, :, :, :, 0, :] == 4).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 1, :] == 5).all()


def test_calculate_derived_values_different_region_count_nr_peaks_2():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    """
    Test calculation in case of more Rayleigh than
    Brillouin regions
    """
    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 2,
        'nr_rayleigh_regions': 3,
    })

    evm.results['brillouin_peak_position'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position'][:, :, :, :, 1, :] = 4
    evm.results['brillouin_peak_position'][:, :, :, :, 0, 1] = 2
    evm.results['brillouin_peak_position'][:, :, :, :, 1, 1] = 5
    evm.results['rayleigh_peak_position'][:, :, :, :, 0, :] = 5
    evm.results['rayleigh_peak_position'][:, :, :, :, 1, :] = 9
    evm.results['rayleigh_peak_position'][:, :, :, :, 2, :] = 10

    psm = evc.session.peak_selection_model()

    psm.add_brillouin_region((1, 2))
    psm.add_brillouin_region((4, 5))

    psm.add_rayleigh_region((0, 1))
    psm.add_rayleigh_region((6, 7))
    psm.add_rayleigh_region((8, 9))

    calculate_derived_values()

    assert (evm.results['brillouin_shift'][:, :, :, :, 0, 0] == 4).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 1, 0] == 5).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 0, 1] == 3).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 1, 1] == 4).all()
