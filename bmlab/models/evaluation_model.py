import logging
import numpy as np
from collections import OrderedDict

from bmlab.serializer import Serializer

logger = logging.getLogger(__name__)


class EvaluationModel(Serializer):

    def __init__(self):

        self.nr_brillouin_peaks = 1
        self.spectra = {}

        # @since 0.1.0
        self.parameters = self.get_default_parameters()
        # @since 0.1.8
        self.bounds = None

        self.results = {}
        for key in self.parameters.keys():
            self.results[key] = np.empty((0,))

    def post_deserialize(self):
        # Migrations from 0.0.13 to 0.1.0
        # Check that the parameters attribute is present
        # @since 0.1.0
        if not hasattr(self, 'parameters')\
                or not isinstance(self, OrderedDict):
            self.parameters = self.get_default_parameters()
        # Migrations from 0.1.7 to 0.1.8
        # Check that the bounds attribute is present
        # @since 0.1.8
        if not hasattr(self, 'bounds'):
            self.bounds = None
        # Migrations from 0.3.0 to 0.4.0
        # Check that the results array also stores the peak offset
        # @since 0.4.0
        if 'brillouin_peak_offset' not in self.results:
            self.results['brillouin_peak_offset'] = np.empty(
                self.results['brillouin_peak_intensity'].shape
            )
            self.results['brillouin_peak_offset'][:] = np.nan
        if 'rayleigh_peak_offset' not in self.results:
            self.results['rayleigh_peak_offset'] = np.empty(
                self.results['rayleigh_peak_intensity'].shape
            )
            self.results['rayleigh_peak_offset'][:] = np.nan
        # Migrations from 0.4.0 to 0.5.0
        # @since 0.5.0
        if 'rayleigh_shift' not in self.results:
            # Stores how much the Rayleigh peak moved
            # Used for shifting the regions evaluated
            self.results['rayleigh_shift'] = np.empty(
                self.results['rayleigh_peak_intensity'].shape
            )

    @staticmethod
    def get_default_parameters():
        return OrderedDict({
            'brillouin_shift_f': {          # [GHz] Brillouin frequency shift
                'unit': 'GHz',
                'symbol': r'$\nu_\mathrm{B}$',
                'label': 'Brillouin frequency shift',
                'scaling': 1e-9,
            },
            'brillouin_shift': {            # [pix] Brillouin frequency shift
                'unit': 'pix',
                'symbol': r'$\nu_\mathrm{B}$',
                'label': 'Brillouin frequency shift',
                'scaling': 1,
            },
            'brillouin_peak_fwhm_f': {      # [GHz] Brillouin peak FWHM
                'unit': 'GHz',
                'symbol': r'$\Delta_\mathrm{B}$',
                'label': 'Brillouin peak width',
                'scaling': 1e-9,
            },
            'brillouin_peak_fwhm': {        # [pix] Brillouin peak FWHM
                'unit': 'pix',
                'symbol': r'$\Delta_\mathrm{B}$',
                'label': 'Brillouin peak width',
                'scaling': 1,
            },
            'brillouin_peak_position': {    # [pix] Brillouin peak position
                'unit': 'pix',
                'symbol': r'$s_\mathrm{B}$',
                'label': 'Brillouin peak position',
                'scaling': 1,
            },
            'brillouin_peak_intensity': {   # [a.u.] Brillouin peak intensity
                'unit': 'a.u.',
                'symbol': r'$I_\mathrm{B}$',
                'label': 'Brillouin peak intensity',
                'scaling': 1,
            },
            'brillouin_peak_offset': {   # [a.u.] Brillouin peak offset
                'unit': 'a.u.',
                'symbol': r'$I_{0,\mathrm{B}}$',
                'label': 'Brillouin peak offset',
                'scaling': 1,
            },
            'rayleigh_peak_fwhm_f': {       # [GHz] Rayleigh peak FWHM
                'unit': 'GHz',
                'symbol': r'$\Delta_\mathrm{R}$',
                'label': 'Rayleigh peak width',
                'scaling': 1e-9,
            },
            'rayleigh_peak_fwhm': {         # [pix] Rayleigh peak FWHM
                'unit': 'pix',
                'symbol': r'$\Delta_\mathrm{R}$',
                'label': 'Rayleigh peak width',
                'scaling': 1,
            },
            'rayleigh_peak_position': {     # [pix] Rayleigh peak position
                'unit': 'pix',
                'symbol': r'$s_\mathrm{R}$',
                'label': 'Rayleigh peak position',
                'scaling': 1,
            },
            'rayleigh_peak_intensity': {    # [a.u.] Rayleigh peak intensity
                'unit': 'a.u.',
                'symbol': r'$I_\mathrm{R}$',
                'label': 'Rayleigh peak intensity',
                'scaling': 1,
            },
            'rayleigh_peak_offset': {   # [a.u.] Rayleigh peak offset
                'unit': 'a.u.',
                'symbol': r'$I_{0,\mathrm{R}}$',
                'label': 'Rayleigh peak offset',
                'scaling': 1,
            },
            'intensity': {                  # [a.u.] Overall intensity of image
                'unit': 'a.u.',
                'symbol': r'$I_\mathrm{total}$',
                'label': 'Intensity',
                'scaling': 1,
            },
            'time': {                       # [s] The time the measurement
                'unit': 's',                # point was taken at
                'symbol': r'$t$',
                'label': 'Time',
                'scaling': 1,
            },
        })

    def initialize_results_arrays(self, dims):
        shape_general = (
            dims['dim_x'],
            dims['dim_y'],
            dims['dim_z'],
            dims['nr_images'],
            1,  # We just add this so it matches the ndims of the
            1,  # Brillouin array and reshapes are reduced
        )

        self.results['intensity'] = np.empty(shape_general)
        self.results['intensity'][:] = np.nan

        self.results['time'] = np.empty(shape_general)
        self.results['time'][:] = np.nan

        # We always do a single-peak fit, plus a multi-peak fit if requested.
        # Hence, we have to store
        # (nr_brillouin_peaks + 1) peaks, if nr_brillouin_peaks > 1.
        nr_brillouin_peaks_to_store = dims['nr_brillouin_peaks']
        if dims['nr_brillouin_peaks'] > 1:
            nr_brillouin_peaks_to_store = nr_brillouin_peaks_to_store + 1

        shape_brillouin = (
            dims['dim_x'],
            dims['dim_y'],
            dims['dim_z'],
            dims['nr_images'],
            dims['nr_brillouin_regions'],
            nr_brillouin_peaks_to_store,
        )

        self.results['brillouin_peak_position'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_position'][:] = np.nan

        self.results['brillouin_peak_fwhm'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_fwhm'][:] = np.nan

        self.results['brillouin_peak_intensity'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_intensity'][:] = np.nan

        self.results['brillouin_peak_offset'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_offset'][:] = np.nan

        self.results['brillouin_shift'] = np.empty(shape_brillouin)
        self.results['brillouin_shift'][:] = np.nan

        self.results['brillouin_shift_f'] = np.empty(shape_brillouin)
        self.results['brillouin_shift_f'][:] = np.nan

        self.results['brillouin_peak_fwhm_f'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_fwhm_f'][:] = np.nan

        shape_rayleigh = (
            dims['dim_x'],
            dims['dim_y'],
            dims['dim_z'],
            dims['nr_images'],
            dims['nr_rayleigh_regions'],
            1,  # We just add this so it matches the ndims of the
                #  Brillouin array and reshapes are reduced
        )

        self.results['rayleigh_peak_position'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_position'][:] = np.nan

        self.results['rayleigh_peak_fwhm'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_fwhm'][:] = np.nan

        self.results['rayleigh_peak_intensity'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_intensity'][:] = np.nan

        self.results['rayleigh_peak_offset'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_offset'][:] = np.nan

        self.results['rayleigh_peak_fwhm_f'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_fwhm_f'][:] = np.nan

        self.results['rayleigh_shift'] = np.empty(shape_rayleigh)
        self.results['rayleigh_shift'][:] = np.nan

    def set_spectra(self, image_key, spectra):
        self.spectra[image_key] = spectra

    def get_spectra(self, image_key):
        spectra = self.spectra.get(image_key)
        if spectra:
            return spectra
        return None

    def get_fits(self, ind_x, ind_y, ind_z):
        return (self.results['brillouin_peak_position'][
               ind_x, ind_y, ind_z, :, :, :],
               self.results['brillouin_peak_fwhm'][
               ind_x, ind_y, ind_z, :, :, :],
               self.results['brillouin_peak_intensity'][
               ind_x, ind_y, ind_z, :, :, :],
               self.results['brillouin_peak_offset'][
               ind_x, ind_y, ind_z, :, :, :]), \
               (self.results['rayleigh_peak_position'][
                ind_x, ind_y, ind_z, :, :, :],
                self.results['rayleigh_peak_fwhm'][
                ind_x, ind_y, ind_z, :, :, :],
                self.results['rayleigh_peak_intensity'][
                ind_x, ind_y, ind_z, :, :, :],
                self.results['rayleigh_peak_offset'][
                ind_x, ind_y, ind_z, :, :, :])

    def get_parameter_keys(self):
        return self.parameters

    def setNrBrillouinPeaks(self, nr_brillouin_peaks):
        self.nr_brillouin_peaks = nr_brillouin_peaks

        self.check_bounds()

    def set_bounds(self, bounds):
        self.bounds = bounds

        self.check_bounds()

    def check_bounds(self):
        # Check the bounds array for consistency
        # We don't need any bounds for a single-peak fit
        if self.nr_brillouin_peaks == 1:
            self.bounds = None

        # Initialize the bounds if necessary
        if self.nr_brillouin_peaks > 1 and\
                (self.bounds is None or
                 len(self.bounds) is not self.nr_brillouin_peaks):
            self.bounds = [['min', 'max'] for _ in
                           range(self.nr_brillouin_peaks)]
