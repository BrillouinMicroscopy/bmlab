===============
Getting started
===============

Installation
============

To install bmlab, use one of the following methods:
    
* from `PyPI <https://pypi.python.org/pypi/bmlab>`_:
    ``pip install bmlab``
* from `sources <https://github.com/BrillouinMicroscopy/bmlab>`_:
    ``pip install .``


Basic usage
===========

.. code-block:: python

    import pathlib

    from bmlab.controllers import Controller
    from bmlab.models import Orientation
    from bmlab.models.setup import AVAILABLE_SETUPS

    filepath = pathlib.Path(__file__).parent.parent / 'tests' / 'data' / 'Water.h5'
    setup = AVAILABLE_SETUPS[0]
    orientation = Orientation(rotation=1, reflection={
            'vertically': False, 'horizontally': False
        })

    # Frequency ranges to evaluate in Hz
    brillouin_regions = [(4.0e9, 6.0e9), (9.0e9, 11.0e9)]
    rayleigh_regions = [(-2.0e9, 2.0e9), (13.0e9, 17.0e9)]

    # The number of Brillouin peaks to fit. Can be set to 2
    # for samples where contributions to the Brillouin shift
    # from different materials are expected in the same focal
    # volume, e.g. droplets.
    nr_brillouin_peaks = 1

    # Bounds for the multi-peak fit, only considered for nr_brillouin_peaks = 2.
    # Possible options: 'min', 'max', '-inf', 'inf' or number in GHz.
    multi_peak_bounds = [
        ['min', 5.0],
        [5.5, 'inf']
    ]

    # This condition is necessary due to
    # bmlab using multiprocessing!
    if __name__ == '__main__':
        session = Controller().evaluate(
            filepath,
            setup,
            orientation,
            brillouin_regions,
            rayleigh_regions,
            nr_brillouin_peaks,
            multi_peak_bounds
        )
        session.save()


Citing bmlab
============
If you use bmlab in a scientific publication, please cite it with:

.. pull-quote::

   bmlab developers (2022), bmlab version X.X.X: Python library for the
   post-measurement analysis of Brillouin microscopy data sets
   [Software]. Available at https://github.com/BrillouinMicroscopy/bmlab.
