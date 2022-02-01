import pathlib

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
