import pathlib

import numpy as np

from bmlab.controllers import Controller
from bmlab.models import Orientation
from bmlab.models.setup import AVAILABLE_SETUPS


def run_pipeline():

    filepath = pathlib.Path(__file__).parent / 'data' / 'Water.h5'
    setup = AVAILABLE_SETUPS[0]
    orientation = Orientation(rotation=1, reflection={
            'vertically': False, 'horizontally': False
        })

    # Frequency ranges to evaluate in GHz
    brillouin_regions = [(4.0e9, 6.0e9), (9.0e9, 11.0e9)]
    rayleigh_regions = [(-2.0e9, 2.0e9), (13.0e9, 17.0e9)]

    session = Controller().evaluate(
        filepath,
        setup,
        orientation,
        brillouin_regions,
        rayleigh_regions
    )

    return session


def test_run_pipeline():
    session = run_pipeline()
    evm = session.evaluation_model()
    shift = evm.results['brillouin_shift_f']
    assert shift.size != 0
    np.testing.assert_allclose(shift, 5.03e9, atol=50E6)
