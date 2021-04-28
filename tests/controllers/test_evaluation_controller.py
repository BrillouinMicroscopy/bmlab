import pathlib

from bmlab.controllers.evaluation_controller import EvaluationController


def data_file_path(file_name):
    return pathlib.Path(__file__).parent.parent / 'data' / file_name


def test_evaluate():

    evaluationcontroller = EvaluationController()
    evaluationcontroller.evaluate()


def test_calculate_derived_values():
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

    evc.calculate_derived_values()

    assert (evm.results['brillouin_shift'] == 2).all()

    evm.results['brillouin_peak_position'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position'][:, :, :, :, 1, :] = 4
    evm.results['rayleigh_peak_position'][:, :, :, :, 0, :] = 3
    evm.results['rayleigh_peak_position'][:, :, :, :, 1, :] = 8

    evc.calculate_derived_values()

    assert (evm.results['brillouin_shift'][:, :, :, :, 0, :] == 2).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 1, :] == 4).all()

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

    evc.calculate_derived_values()

    assert (evm.results['brillouin_shift'][:, :, :, :, 0, :] == 4).all()
    assert (evm.results['brillouin_shift'][:, :, :, :, 1, :] == 5).all()
