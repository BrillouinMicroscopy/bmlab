from bmlab.models.evaluation_model import EvaluationModel


def test_initialize_results_arrays():
    evm = EvaluationModel()
    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 1,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 1,
    })

    assert evm.results['intensity'].shape == (5, 5, 5, 2, 1, 1)
    assert evm.results['brillouin_peak_position'].shape == (5, 5, 5, 2, 1, 1)
    assert evm.results['rayleigh_peak_position'].shape == (5, 5, 5, 2, 1, 1)

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

    assert evm.results['intensity'].shape == (5, 5, 5, 2, 1, 1)
    assert evm.results['brillouin_peak_position'].shape == (5, 5, 5, 2, 2, 1)
    assert evm.results['rayleigh_peak_position'].shape == (5, 5, 5, 2, 2, 1)

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

    assert evm.results['intensity'].shape == (5, 5, 5, 2, 1, 1)
    # If we have more than one Brillouin peak,
    # we store the result of a single peak fit
    # and the result of a multi peak fit, adding up to nr_brillouin_peaks + 1
    assert evm.results['brillouin_peak_position'].shape == (5, 5, 5, 2, 2, 3)
    assert evm.results['rayleigh_peak_position'].shape == (5, 5, 5, 2, 2, 1)
