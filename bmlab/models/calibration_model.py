import logging

import numpy as np
from scipy import interpolate

from bmlab.models.regions import regions_merge_add_region
from bmlab.serializer import Serializer

logger = logging.getLogger(__name__)


class CalibrationModel(Serializer):

    def __init__(self):
        self.calib_times = {}
        self.spectra = {}
        self.brillouin_regions = {}
        self.rayleigh_regions = {}
        self.brillouin_fits = BrillouinFitSet()
        self.rayleigh_fits = RayleighFitSet()

        self.vipa_params = {}
        self.frequencies = {}

        self.frequency_by_calib_key_interpolators = {}
        self.frequencies_by_time_interpolator = None
        self.frequency_by_time_interpolator = None

    def post_deserialize(self):
        self.refresh_frequency_interpolators()

    def add_brillouin_region(self, calib_key, region):
        if calib_key not in self.brillouin_regions:
            self.brillouin_regions[calib_key] = []

        region = tuple(round(x) for x in region)

        regions_merge_add_region(
            self.brillouin_regions[calib_key], region)

    def set_brillouin_region(self, calib_key, index, region):
        if calib_key not in self.brillouin_regions:
            self.brillouin_regions[calib_key] = []

        region = tuple(round(x) for x in region)

        if index < len(self.brillouin_regions[calib_key]):
            self.brillouin_regions[calib_key][index] = region
        else:
            self.brillouin_regions[calib_key].append(region)

    def get_brillouin_regions(self, calib_key):
        regions = self.brillouin_regions.get(calib_key)
        if regions is None:
            return []
        return regions

    def clear_brillouin_regions(self, calib_key):
        self.brillouin_regions[calib_key] = []

    def add_brillouin_fit(self, calib_key, region, frame_num,
                          w0s, fwhms, intensities, offset):
        fit = BrillouinFit(calib_key, region, frame_num,
                           w0s, fwhms, intensities, offset)
        self.brillouin_fits.add_fit(fit)

    def get_brillouin_fit(self, calib_key, region, frame_num):
        return self.brillouin_fits.get_fit(calib_key, region, frame_num)

    def clear_brillouin_fits(self, calib_key):
        self.brillouin_fits.clear(calib_key)

    def add_rayleigh_region(self, calib_key, region):
        if calib_key not in self.rayleigh_regions:
            self.rayleigh_regions[calib_key] = []

        region = tuple(round(x) for x in region)

        regions_merge_add_region(
            self.rayleigh_regions[calib_key], region)

    def set_rayleigh_region(self, calib_key, index, region):
        if calib_key not in self.rayleigh_regions:
            self.rayleigh_regions[calib_key] = []

        region = tuple(round(x) for x in region)

        if index < len(self.rayleigh_regions[calib_key]):
            self.rayleigh_regions[calib_key][index] = region
        else:
            self.rayleigh_regions[calib_key].append(region)

    def get_rayleigh_regions(self, calib_key):
        regions = self.rayleigh_regions.get(calib_key)
        if regions is None:
            return []
        return regions

    def clear_rayleigh_regions(self, calib_key):
        self.rayleigh_regions[calib_key] = []

    def add_rayleigh_fit(self, calib_key, region, frame_num,
                         w0, fwhm, intensity, offset):
        fit = RayleighFit(calib_key, region, frame_num,
                          w0, fwhm, intensity, offset)
        self.rayleigh_fits.add_fit(fit)

    def get_rayleigh_fit(self, calib_key, region, frame_num):
        return self.rayleigh_fits.get_fit(calib_key, region, frame_num)

    def clear_rayleigh_fits(self, calib_key):
        self.rayleigh_fits.clear(calib_key)

    def set_spectra(self, calib_key, spectra):
        self.spectra[calib_key] = spectra

    def get_spectra(self, calib_key):
        spectra = self.spectra.get(calib_key)
        if spectra:
            return spectra
        return None

    def get_sorted_peaks(self, calib_key, frame_num):
        """
        Returns the sorted centers of all fitted peaks
        of a given calibration and frame.

        Parameters
        ----------
        calib_key: The calibration key
        frame_num: The frame number

        Returns
        -------
        sorted np.array of all peaks
        """
        peaks = []
        # Search all fits for given calib_key and frame_num
        for key, fit in self.rayleigh_fits.fits.items():
            if (fit.calib_key == calib_key) and (fit.frame_num == frame_num):
                peaks.append(fit.w0)

        for key, fit in self.brillouin_fits.fits.items():
            if (fit.calib_key == calib_key) and (fit.frame_num == frame_num):
                for w0 in fit.w0s:
                    peaks.append(w0)

        return np.sort(np.array(peaks))

    def set_vipa_params(self, calib_key, vipa_params):
        self.vipa_params[calib_key] = vipa_params

    def clear_vipa_params(self, calib_key):
        if calib_key in self.vipa_params:
            del self.vipa_params[calib_key]

    def set_frequencies(self, calib_key, time, frequencies):
        self.frequencies[calib_key] = frequencies
        self.calib_times[calib_key] = time
        self.refresh_frequency_interpolators()

    def clear_frequencies(self, calib_key):
        if calib_key in self.frequencies:
            del self.frequencies[calib_key]
        if calib_key in self.calib_times:
            del self.calib_times[calib_key]
        self.refresh_frequency_interpolators()

    def refresh_frequency_interpolators(self):
        """
        This function creates the interpolator to get
        the frequencies of points in a spectrum

        Returns
        -------
        """
        # Reset all interpolators
        self.frequency_by_calib_key_interpolators = {}
        self.frequencies_by_time_interpolator = None
        self.frequency_by_time_interpolator = None

        sorted_keys = sorted(self.calib_times,
                             key=self.calib_times.get)
        # Don't do anything if there are not calibrations
        if len(sorted_keys) < 1:
            return

        """
        Create the interpolator for getting a frequency in a
        calibration spectrum
        """
        for calib_key in self.frequencies:
            frequencies = self.get_frequencies_by_calib_key(calib_key)
            if frequencies is None or not frequencies:
                return
            frequency = np.mean(np.array(frequencies), axis=0)

            xdata = np.arange(len(frequency))
            self.frequency_by_calib_key_interpolators[calib_key] =\
                interpolate.interp1d(xdata, frequency)

        """
        Create the interpolator for getting frequencies by time
        """
        if len(sorted_keys) < 2:
            self.frequencies_by_time_interpolator = \
                lambda time: np.nanmean(self.frequencies[sorted_keys[0]], 0)
        else:
            calib_times_array = []
            frequencies = []
            for key in sorted_keys:
                calib_times_array.append(self.calib_times[key])
                frequencies.append(np.nanmean(self.frequencies[key], 0))

            calib_times_array = np.array(calib_times_array)
            frequencies = np.squeeze(frequencies)

            self.frequencies_by_time_interpolator =\
                interpolate.interp1d(calib_times_array, frequencies, axis=0)

        """
        Create the interpolator for getting a frequency by position and time
        """
        calib_times_array = []
        frequencies = []
        for key in sorted_keys:
            calib_times_array.append(self.calib_times[key])
            frequencies.append(np.nanmean(self.frequencies[key], 0))

        calib_times_array = np.array(calib_times_array)
        frequencies = np.array(frequencies)

        indices = np.arange(frequencies.shape[1])

        # If we only have one time point, we
        # interpolate by peak position only
        if len(sorted_keys) < 2:
            frequencies = np.squeeze(frequencies)
            f = interpolate.interp1d(indices, frequencies)
            self.frequency_by_time_interpolator =\
                lambda time, position: f(position)
        # Otherwise we can interpolate by time as well
        elif len(calib_times_array) < 3:
            f = interpolate.RegularGridInterpolator(
                (calib_times_array, indices),
                frequencies,
                method='linear',
                bounds_error=False
            )
            self.frequency_by_time_interpolator = \
                lambda time, position: f((time, position))
        else:
            # If we only have three entries we cannot use a
            # third degree spline
            degree = 3 if len(calib_times_array) > 3 else 2
            f = interpolate.RectBivariateSpline(
                calib_times_array,
                indices,
                frequencies,
                kx=degree
            )
            self.frequency_by_time_interpolator = \
                lambda time, position: f(time, position, grid=False)

    def get_frequencies_by_calib_key(self, calib_key):
        """
        Returns the complete frequency axis for a given

        Parameters
        ----------
        calib_key: The key of the calibration

        Returns
        -------
        The frequency axis in Hz
        """
        if calib_key in self.frequencies:
            return self.frequencies[calib_key]

    def get_frequency_by_calib_key(self, position, calib_key):
        """
        Returns the frequency of a peak position on the
        spectrum for a given calibration

        Parameters
        ----------
        position: The position(s) of the peak(s)
        calib_key: The key of the calibration

        Returns
        -------
        The corresponding frequency in Hz
        """
        if calib_key in self.frequency_by_calib_key_interpolators:
            return self.frequency_by_calib_key_interpolators[
                calib_key](position)

    def get_frequencies_by_time(self, time):
        """
        Returns the complete frequency axis for a given

        Parameters
        ----------
        time: The time

        Returns
        -------
        The frequency axis in Hz
        """
        if self.frequencies_by_time_interpolator is not None:
            return self.frequencies_by_time_interpolator(time)

    def get_frequency_by_time(self, time, position):
        """
        Returns the frequency of a peak position on the
        spectrum for a given time

        Parameters
        ----------
        time: The time
        position: The position(s) of the peak(s)
        on the spectrum

        Returns
        -------
        The corresponding frequency in Hz
        """
        # Cannot execute if ndims are unequal
        if np.ndim(time) is not np.ndim(position):
            return None

        shape_time = np.shape(time)
        shape_position = np.shape(position)

        # In case the arrays don't have the same shape,
        # we repeat the time array
        if shape_time is not shape_position:
            time = np.tile(
                time,
                (np.array(shape_position) - np.array(shape_time)) + 1
            )

        if self.frequency_by_time_interpolator is not None:
            return self.frequency_by_time_interpolator(time, position)

    def get_position_by_time(self, time, frequencies):
        """
        Returns the position on the spectrum in pix for a given frequency
        at a certain time

        Parameters
        ----------
        time: The time
        frequencies: [GHz] The frequencies we want to convert to pixel

        Returns
        -------

        """
        spectrum = self.get_frequencies_by_time(time)
        f = interpolate.interp1d(spectrum, range(len(spectrum)))
        return f(frequencies)


class FitSet(Serializer):

    def __init__(self):
        self.fits = {}

    def make_key(self, calib_key, region_key, frame_num):
        return calib_key + '::' + str(region_key) + '::' + str(frame_num)

    def split_key(self, key):
        items = key.split('::')
        items[1] = int(items[1])
        items[2] = int(items[2])
        return items

    def add_fit(self, fit):
        key = self.make_key(fit.calib_key, fit.region_key, fit.frame_num)
        self.fits[key] = fit

    def get_fit(self, calib_key, region_key, frame_num=None):
        key = self.make_key(calib_key, region_key, frame_num)
        return self.fits.get(key)

    def clear(self, calib_key):
        keys = []
        for key, value in self.fits.items():
            if calib_key == self.split_key(key)[0]:
                keys.append(key)
        for key in keys:
            del self.fits[key]


class RayleighFitSet(FitSet, Serializer):

    def average_fits(self, calib_key, region_key):
        w0s = []
        for key, fit in self.fits.items():
            calib_key_, region_key_, _ = self.split_key(key)
            if calib_key == calib_key_ and region_key == region_key_:
                w0s.append(fit.w0)
        logger.debug('w0s = ', w0s)
        if w0s:
            return np.mean(w0s)
        return None


class BrillouinFitSet(FitSet, Serializer):

    def average_fits(self, calib_key, region_key):
        w0s = []
        for key, fit in self.fits.items():
            calib_key_, region_key_, _ = self.split_key(key)
            if calib_key == calib_key_ and region_key == region_key_:
                w0s.append(fit.w0s)
        logger.debug('w0s = ', w0s)
        if w0s:
            w0s = np.array(w0s)
            return np.mean(w0s, axis=0)
        return None


class RayleighFit(Serializer):

    def __init__(self, calib_key, region_key, frame_num,
                 w0, fwhm, intensity, offset):
        self.calib_key = calib_key
        self.region_key = region_key
        self.frame_num = frame_num
        self.w0 = w0
        self.fwhm = fwhm
        self.intensity = intensity
        self.offset = offset


class BrillouinFit(Serializer):

    def __init__(self, calib_key, region_key, frame_num,
                 w0s, fwhms, intensities, offset):
        self.calib_key = calib_key
        self.region_key = region_key
        self.frame_num = frame_num
        self.w0s = w0s
        self.fwhms = fwhms
        self.intensities = intensities
        self.offset = offset
