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

    brillouin_regions = [(190, 250), (290, 350)]
    rayleigh_regions = [(110, 155), (370, 410)]

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
    np.testing.assert_allclose(
        evm.results['brillouin_shift_f'], 5.03e9, atol=50E6)
