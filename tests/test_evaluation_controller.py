import pathlib
import numpy as np

from bmlab.controllers import EvaluationController, calculate_derived_values
from bmlab.models import CalibrationModel, EvaluationModel


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


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

    evm.results['brillouin_peak_fwhm'][:, :, :, :, :, 0] = 1e9  # [Hz]
    evm.results['brillouin_peak_fwhm'][:, :, :, :, :, 1] = 1e9  # [Hz]
    evm.results['brillouin_peak_fwhm'][:, :, :, :, :, 2] = 2e9  # [Hz]

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
    fit_bounds = EvaluationController().create_bounds(None, None, None)
    assert fit_bounds is None

    evm.bounds = [['min', '5'], ['5.5', 'Inf'], ['-inf', 'max']]

    brillouin_regions = [(200, 280), (400, 480)]
    times = [0, 1, 2]
    rayleigh_peaks = np.array([[40, 40, 41], [640, 640, 641]])
    fit_bounds = evc.create_bounds(brillouin_regions, times, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[200, 240], [260, np.inf], [-np.inf, 280]],
            [[200, 240], [260, np.inf], [-np.inf, 280]],
            [[200, 241], [261, np.inf], [-np.inf, 280]]
        ], [
            [[440, 480], [-np.inf, 420], [400, np.inf]],
            [[440, 480], [-np.inf, 420], [400, np.inf]],
            [[441, 480], [-np.inf, 421], [400, np.inf]]
        ]
    ], fit_bounds, atol=1.E-3)
