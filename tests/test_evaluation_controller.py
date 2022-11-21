import pathlib
import numpy as np

import pytest

from bmlab.controllers import EvaluationController, calculate_derived_values
from bmlab.models import CalibrationModel, EvaluationModel


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_get_key_from_indices():
    resolution = (7, 8, 9)
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 0, 0) == '0'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0.0, 0.0, 0.0) == '0'
    assert EvaluationController\
        .get_key_from_indices(resolution, 6, 0, 0) == '6'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 1, 0) == '7'
    assert EvaluationController\
        .get_key_from_indices(resolution, 1, 1, 0) == '8'
    assert EvaluationController\
        .get_key_from_indices(resolution, 6, 7, 0) == '55'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 0, 1) == '56'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 1, 1) == '63'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 0, 8) == '448'
    assert EvaluationController\
        .get_key_from_indices(resolution, 1, 0, 8) == '449'
    assert EvaluationController\
        .get_key_from_indices(resolution, 1, 1, 8) == '456'
    assert EvaluationController\
        .get_key_from_indices(resolution, 6, 7, 8) == '503'

    # Resolution is not three-dimensional
    with pytest.raises(ValueError) as err:
        EvaluationController \
            .get_key_from_indices((7, 8), 6, 7, 8)
    assert err.typename == 'ValueError'

    # x index is out or range
    with pytest.raises(IndexError) as err:
        EvaluationController \
            .get_key_from_indices((7, 8, 9), 7, 9, 9)
    assert err.typename == 'IndexError'


def test_get_indices_from_key():
    resolution = (7, 8, 9)
    assert EvaluationController\
        .get_indices_from_key(resolution, '0') == (0, 0, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '6') == (6, 0, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '7') == (0, 1, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '8') == (1, 1, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '55') == (6, 7, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '56') == (0, 0, 1)
    assert EvaluationController\
        .get_indices_from_key(resolution, '63') == (0, 1, 1)
    assert EvaluationController\
        .get_indices_from_key(resolution, '448') == (0, 0, 8)
    assert EvaluationController\
        .get_indices_from_key(resolution, '449') == (1, 0, 8)
    assert EvaluationController\
        .get_indices_from_key(resolution, '456') == (1, 1, 8)
    assert EvaluationController\
        .get_indices_from_key(resolution, '503') == (6, 7, 8)

    # Key is invalid
    with pytest.raises(ValueError) as err:
        EvaluationController \
            .get_indices_from_key((7, 8, 9), '504')
    assert err.typename == 'ValueError'


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

    evm.results['brillouin_peak_position_f'][:] = 1
    evm.results['rayleigh_peak_position_f'][:] = 3

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'] == 2).all()

    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 4
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 0, :] = -1
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 1, :] = 8

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, :] == 2).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 1, :] == 4).all()


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

    evm.results['brillouin_peak_position_f'][:] = 1
    evm.results['rayleigh_peak_position_f'][:] = 3

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'] == 2).all()

    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 4
    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, 1] = 2
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, 1] = 5
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 0, :] = -1
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 1, :] = 8

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, 0] == 2).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 1, 0] == 4).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, 1] == 3).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 1, 1] == 3).all()


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

    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 4
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 0, :] = -2
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 1, :] = 9
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 2, :] = 10

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, :] == 3).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 1, :] == 5).all()


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

    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 4
    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, 1] = 2
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, 1] = 5
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 0, :] = -2
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 1, :] = 9
    evm.results['rayleigh_peak_position_f'][0:4, :, :, :, 2, :] = 10
    evm.results['rayleigh_peak_position_f'][4, :, :, :, 2, :] = 8

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, 0] == 3).all()
    assert (evm.results['brillouin_shift_f'][0:4, :, :, :, 1, 0] == 5).all()
    assert (evm.results['brillouin_shift_f'][4, :, :, :, 1, 0] == 4).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, 1] == 4).all()
    assert (evm.results['brillouin_shift_f'][0:4, :, :, :, 1, 1] == 4).all()
    assert (evm.results['brillouin_shift_f'][4, :, :, :, 1, 1] == 3).all()


def test_get_data_0D(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('0D.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (1, 1, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    assert data == 1  # [GHz]
    assert positions[0][0, 0, 0] == -1
    assert positions[1][0, 0, 0] == -2
    assert positions[2][0, 0, 0] == -3
    assert dimensionality == 0
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_1D_x(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('1D-x.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (3, 1, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    positions_x = np.zeros(shape=resolution)
    positions_x[:, 0, 0] = [-1, 0, 1]
    positions_y = np.zeros(shape=resolution)
    positions_y[:] = -2
    positions_z = np.zeros(shape=resolution)
    positions_z[:] = -3

    assert (data == np.ones(shape=resolution)).all()  # [GHz]
    assert (positions[0] == positions_x).all()
    assert (positions[1] == positions_y).all()
    assert (positions[2] == positions_z).all()
    assert dimensionality == 1
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_1D_y(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('1D-y.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (1, 5, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    positions_x = np.zeros(shape=resolution)
    positions_x[:] = -1
    positions_y = np.zeros(shape=resolution)
    positions_y[0, :, 0] = [-2, -1, 0, 1, 2]
    positions_z = np.zeros(shape=resolution)
    positions_z[:] = -3

    assert (data == np.ones(shape=resolution)).all()  # [GHz]
    assert (positions[0] == positions_x).all()
    assert (positions[1] == positions_y).all()
    assert (positions[2] == positions_z).all()
    assert dimensionality == 1
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_1D_z(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('1D-z.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (1, 1, 7)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    positions_x = np.zeros(shape=resolution)
    positions_x[:] = -1
    positions_y = np.zeros(shape=resolution)
    positions_y[:] = -2
    positions_z = np.zeros(shape=resolution)
    positions_z[0, 0, :] = [-3, -2, -1, 0, 1, 2, 3]

    assert (data == np.ones(shape=resolution)).all()  # [GHz]
    assert (positions[0] == positions_x).all()
    assert (positions[1] == positions_y).all()
    assert (positions[2] == positions_z).all()
    assert dimensionality == 1
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_2D_xy(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('2D-xy.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (3, 5, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    data_expected = np.ones(shape=resolution)
    (pos_y, pos_x, pos_z) = np.meshgrid([-2, -1, 0, 1, 2], [-1, 0, 1], [-3])

    assert (data == data_expected).all()
    assert (positions[0] == pos_x).all()
    assert (positions[1] == pos_y).all()
    assert (positions[2] == pos_z).all()
    assert dimensionality == 2
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_2D_xz(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('2D-xz.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (3, 1, 7)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    data_expected = np.ones(shape=resolution)
    (pos_y, pos_x, pos_z) = np.meshgrid(
        [-2], [-1, 0, 1], [-3, -2, -1, 0, 1, 2, 3]
    )

    assert (data == data_expected).all()
    assert (positions[0] == pos_x).all()
    assert (positions[1] == pos_y).all()
    assert (positions[2] == pos_z).all()
    assert dimensionality == 2
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_2D_yz(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('2D-yz.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (1, 5, 7)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    data_expected = np.ones(shape=resolution)
    (pos_y, pos_x, pos_z) = np.meshgrid(
        [-2, -1, 0, 1, 2], [-1], [-3, -2, -1, 0, 1, 2, 3]
    )

    assert (data == data_expected).all()
    assert (positions[0] == pos_x).all()
    assert (positions[1] == pos_y).all()
    assert (positions[2] == pos_z).all()
    assert dimensionality == 2
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_3D(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('3D.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (3, 5, 7)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    data_expected = np.ones(resolution)
    (pos_y, pos_x, pos_z) = np.meshgrid(
        [-2, -1, 0, 1, 2],
        [-1, 0, 1],
        [-3, -2, -1, 0, 1, 2, 3]
    )

    assert dimensionality == 3
    assert (data == data_expected).all()  # [GHz]
    assert (positions[0] == pos_x).all()
    assert (positions[1] == pos_y).all()
    assert (positions[2] == pos_z).all()
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_3D_peak_index():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('3D.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 2,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:, :, :, :, :, 0] = 2e9  # [Hz]
    evm.results['brillouin_shift_f'][:, :, :, :, :, 1] = 4e9  # [Hz]
    evm.results['brillouin_shift_f'][:, :, :, :, :, 2] = 6e9  # [Hz]

    evm.results['brillouin_peak_fwhm_f'][:, :, :, :, :, 0] = 1e9  # [Hz]
    evm.results['brillouin_peak_fwhm_f'][:, :, :, :, :, 1] = 1e9  # [Hz]
    evm.results['brillouin_peak_fwhm_f'][:, :, :, :, :, 2] = 2e9  # [Hz]

    evm.results['brillouin_peak_intensity'][:, :, :, :, :, 0] = 1e9  # [Hz]
    evm.results['brillouin_peak_intensity'][:, :, :, :, :, 1] = 2e9  # [Hz]
    evm.results['brillouin_peak_intensity'][:, :, :, :, :, 2] = 3e9  # [Hz]

    # Get first peak
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f')
    assert data.shape == resolution
    assert (data == 2 * np.ones(resolution)).all()  # [GHz]

    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 0)
    assert data.shape == resolution
    assert (data == 2 * np.ones(resolution)).all()  # [GHz]

    # Get first peak of multi-peak fit
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 1)
    assert data.shape == resolution
    assert (data == 4 * np.ones(resolution)).all()  # [GHz]

    # Get second peak of multi-peak fit
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 2)
    assert data.shape == resolution
    assert (data == 6 * np.ones(resolution)).all()  # [GHz]

    # Get average of multi-peak fits
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 3)
    assert data.shape == resolution
    assert (data == 5 * np.ones(resolution)).all()  # [GHz]

    # Get weighted average of multi-peak fits
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 4)
    assert data.shape == resolution
    assert (data == 5.5 * np.ones(resolution)).all()  # [GHz]

    # Get single-peak fit when out of bounds
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 5)
    assert data.shape == resolution
    assert (data == 2 * np.ones(resolution)).all()  # [GHz]


def test_create_bounds(mocker):
    cm = CalibrationModel()
    evm = EvaluationModel()
    mocker.patch('bmlab.session.Session.calibration_model', return_value=cm)
    mocker.patch('bmlab.session.Session.evaluation_model', return_value=evm)
    evc = EvaluationController()

    # Set the calibration data
    cm.set_frequencies('0', 0,
                       list(1e9 * np.linspace(
                           -1, 16, 681).reshape(1, -1)))
    cm.set_frequencies('1', 1,
                       list(1e9 * np.linspace(
                           -0.975, 16.025, 681).reshape(1, -1)))
    cm.set_frequencies('2', 2,
                       list(1e9 * np.linspace(
                           -0.95, 16.05, 681).reshape(1, -1)))

    # If we don't supply regions and times, we get no bounds
    fit_bounds = EvaluationController().create_bounds(None, None)
    assert fit_bounds is None

    # Test that we get correct bounds for regions
    # that only contain one type of peaks (either Stokes or Anti-Stokes)
    evc.set_bounds([['min', '5'], ['5.5', 'Inf'], ['-inf', 'max']])

    brillouin_regions = [(3.3e9, 7.0e9), (8.0e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds = evc.create_bounds(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[3.3e9, 5.0e9], [5.5e9, np.inf], [-np.inf, 7.0e9]],
            [[3.3e9, 5.1e9], [5.6e9, np.inf], [-np.inf, 7.0e9]],
            [[3.3e9, 4.8e9], [5.3e9, np.inf], [-np.inf, 7.0e9]]
        ], [
            [[9.8e9, 12.1e9], [-np.inf, 9.3e9], [8.0e9, np.inf]],
            [[10.0e9, 12.1e9], [-np.inf, 9.5e9], [8.0e9, np.inf]],
            [[10.3e9, 12.1e9], [-np.inf, 9.8e9], [8.0e9, np.inf]]
        ]
    ], fit_bounds, atol=1e6)

    # Test that we get the correct bounds for regions
    # that contain both Stokes and Anti-Stokes peaks
    evc.set_bounds([['min', '5'], ['5.5', 'Inf'], ['-inf', 'max'],
                    ['min', '-5'], ['-5.5', 'Inf'], ['-inf', 'max']])

    brillouin_regions = [(2.3e9, 12.1e9), (3.3e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds = evc.create_bounds(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[2.3e9, 5.0e9], [5.5e9, np.inf], [-np.inf, 12.1e9],
             [9.8e9, 12.1e9], [-np.inf, 9.3e9], [-np.inf, 12.1e9]],
            [[2.3e9, 5.1e9], [5.6e9, np.inf], [-np.inf, 12.1e9],
             [10.0e9, 12.1e9], [-np.inf, 9.5e9], [-np.inf, 12.1e9]],
            [[2.3e9, 4.8e9], [5.3e9, np.inf], [-np.inf, 12.1e9],
             [10.3e9, 12.1e9], [-np.inf, 9.8e9], [-np.inf, 12.1e9]]
        ],
        [
            [[3.3e9, 5.0e9], [5.5e9, np.inf], [-np.inf, 12.1e9],
             [9.8e9, 12.1e9], [-np.inf, 9.3e9], [-np.inf, 12.1e9]],
            [[3.3e9, 5.1e9], [5.6e9, np.inf], [-np.inf, 12.1e9],
             [10.0e9, 12.1e9], [-np.inf, 9.5e9], [-np.inf, 12.1e9]],
            [[3.3e9, 4.8e9], [5.3e9, np.inf], [-np.inf, 12.1e9],
             [10.3e9, 12.1e9], [-np.inf, 9.8e9], [-np.inf, 12.1e9]]
        ]
    ], fit_bounds, atol=1e6)

    # Test real values for a two peak fit
    evc.set_bounds([['3.5', '5'], ['5.0', '7.5']])

    brillouin_regions = [(3.3e9, 7.5e9), (8.0e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds = evc.create_bounds(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[3.5e9, 5.0e9], [5.0e9, 7.5e9]],
            [[3.6e9, 5.1e9], [5.1e9, 7.6e9]],
            [[3.3e9, 4.8e9], [4.8e9, 7.3e9]]
        ], [
            [[9.8e9, 11.3e9], [7.3e9, 9.8e9]],
            [[10.0e9, 11.5e9], [7.5e9, 10.0e9]],
            [[10.3e9, 11.8e9], [7.8e9, 10.3e9]]
        ]
    ], fit_bounds, atol=1e6)

    # Test real values for a four peak fit over the whole spectrum
    evc.set_bounds([['3.5', '5'], ['5.0', '7.5'],
                    ['-7.5', '-5.0'], ['-5', '-3.5']])

    brillouin_regions = [(2.3e9, 12.1e9), (3.3e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds = evc.create_bounds(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[3.5e9, 5.0e9], [5.0e9, 7.5e9],
             [7.3e9, 9.8e9], [9.8e9, 11.3e9]],
            [[3.6e9, 5.1e9], [5.1e9, 7.6e9],
             [7.5e9, 10.0e9], [10.0e9, 11.5e9]],
            [[3.3e9, 4.8e9], [4.8e9, 7.3e9],
             [7.8e9, 10.3e9], [10.3e9, 11.8e9]]
        ], [
            [[3.5e9, 5.0e9], [5.0e9, 7.5e9],
             [7.3e9, 9.8e9], [9.8e9, 11.3e9]],
            [[3.6e9, 5.1e9], [5.1e9, 7.6e9],
             [7.5e9, 10.0e9], [10.0e9, 11.5e9]],
            [[3.3e9, 4.8e9], [4.8e9, 7.3e9],
             [7.8e9, 10.3e9], [10.3e9, 11.8e9]]
        ]
    ], fit_bounds, atol=1e6)


def test_create_bounds_fwhm(mocker):
    cm = CalibrationModel()
    evm = EvaluationModel()
    mocker.patch('bmlab.session.Session.calibration_model', return_value=cm)
    mocker.patch('bmlab.session.Session.evaluation_model', return_value=evm)
    evc = EvaluationController()

    # If we don't supply regions and times, we get no bounds
    fit_bounds = EvaluationController().create_bounds_fwhm(None, None)
    assert fit_bounds is None

    # Test that we get correct bounds for regions
    # that only contain one type of peaks (either Stokes or Anti-Stokes)
    evc.set_bounds_fwhm([['min', '-0.9'], ['2.0', 'Inf'], ['-inf', 'max']])

    brillouin_regions = [(3.3e9, 7.0e9), (8.0e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds_fwhm = evc.create_bounds_fwhm(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]],
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]],
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]]
        ], [
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]],
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]],
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]]
        ]
    ], fit_bounds_fwhm, atol=1e6)
