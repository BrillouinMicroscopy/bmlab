import pathlib
import numpy as np

from bmlab.controllers import EvaluationController, calculate_derived_values


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
