import logging
import numpy as np


logger = logging.getLogger(__name__)


class EvaluationModel(object):

    def __init__(self):

        self.nr_brillouin_peaks = 1

        self.results = {
            'brillouin_peak_position': [],  # [pix] Brillouin peak position
            'brillouin_shift_p': [],    # [pix] Brillouin frequency shift
            'brillouin_shift_f': [],    # [GHz] Brillouin frequency shift
            'brillouin_fwhm_p': [],     # [pix] Brillouin peak width
            'brillouin_fwhm_f': []      # [GHz] Brillouin peak width
        }

        self.parameters = {
            'brillouin_shift_p': {      # Brillouin frequency shift
                'unit', 'pix',
                'symbol', r'$\nu_\mathrm{B}$',
                'label', 'Brillouin frequency shift'
            },
            'brillouin_shift_f': {      # Brillouin frequency shift
                'unit', 'GHz',
                'symbol', r'$\nu_\mathrm{B}$',
                'label', 'Brillouin frequency shift'
            },
            'brillouin_fwhm_p': {       # Brillouin peak width
                'unit', 'pix',
                'symbol', r'$\Delta_\mathrm{B}$',
                'label', 'Brillouin peak width'
            },
            'brillouin_fwhm_f': {       # Brillouin peak width
                'unit', 'GHz',
                'symbol', r'$\Delta_\mathrm{B}$',
                'label', 'Brillouin peak width'
            }
        }

    def get_parameters(self):
        return self.parameters

    def initialize_results_arrays(self, shape):
        self.results['brillouin_peak_position'] = np.empty(shape)
        self.results['brillouin_peak_position'][:] = np.nan
