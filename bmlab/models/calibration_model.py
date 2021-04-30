import logging

import numpy as np
from scipy import interpolate

from bmlab.models.regions import regions_merge_add_region

logger = logging.getLogger(__name__)


class CalibrationModel(object):

    def __init__(self):
        self.calib_times = {}
        self.brillouin_regions = {}
        self.rayleigh_regions = {}
        self.brillouin_fits = BrillouinFitSet()
        self.rayleigh_fits = RayleighFitSet()

        self.vipa_params = {}
        self.frequencies = {}

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

        self.brillouin_regions[calib_key][index] = region

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

        self.rayleigh_regions[calib_key][index] = region

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

    def get_sorted_peaks(self, calib_key, frame_num):
        """
        Returns the sorted centers of all fitted peaks
        of a given calibration and frame.
        :param calib_key: The calibration key
        :param frame_num: The frame number
        :return: sorted np.array of all peaks
        """
        peaks = []
        # Search all fits for given calib_key and frame_num
        for key, fit in self.rayleigh_fits.fits.items():
            if (fit.calib_key == calib_key) & (fit.frame_num == frame_num):
                peaks.append(fit.w0)

        for key, fit in self.brillouin_fits.fits.items():
            if (fit.calib_key == calib_key) & (fit.frame_num == frame_num):
                for w0 in fit.w0s:
                    peaks.append(w0)

        return np.sort(np.array(peaks))

    def set_vipa_params(self, calib_key, vipa_params):
        self.vipa_params[calib_key] = vipa_params

    def set_frequencies(self, calib_key, time, frequencies):
        self.frequencies[calib_key] = frequencies
        self.calib_times[calib_key] = time

    def clear_frequencies(self, calib_key):
        del self.frequencies[calib_key]
        del self.calib_times[calib_key]

    def get_frequencies_by_calib_key(self, calib_key):
        """
        Returns the complete frequency axis for a given
        calibration
        :param calib_key: The key of the calibration
        :return: The frequency axis in Hz
        """
        if calib_key in self.frequencies:
            return self.frequencies[calib_key]

    def get_frequency_by_calib_key(self, position, calib_key):
        """
        Returns the frequency of a peak position on the
        spectrum for a given calibration
        :param position: The position(s) of the peak(s)
        on the spectrum
        :param calib_key: The key of the calibration
        :return: The corresponding frequency in Hz
        """
        # TODO Move the interpolation out of this function
        #  (only needs to be done once for a calibration key)
        #  (not so critical at the moment, since it's only done
        #  for calibrating)
        frequencies = self.get_frequencies_by_calib_key(calib_key)
        if not frequencies:
            return
        frequency = np.mean(np.array(frequencies), axis=0)

        xdata = np.arange(len(frequency))
        f = interpolate.interp1d(xdata, frequency)
        return f(position)

    def get_frequencies_by_time(self, time):
        """
        Returns the complete frequency axis for a given
        time
        :param time: The time
        :return: The frequency axis in Hz
        """

        # Sort calibration keys by time
        sorted_keys = sorted(self.calib_times,
                             key=self.calib_times.get)
        if not sorted_keys:
            return None

        # If we only have one time point, simply return
        #  the complete frequency axis
        if len(sorted_keys) < 2:
            return np.nanmean(self.frequencies[sorted_keys[0]], 0)
        else:
            # TODO Move the interpolation out of this function
            #  (only needs to be done once)
            #  (not so critical at the moment, since this
            #  function is only used for showing the frequency
            #  axis in the peak-selection panel)
            calib_times_array = []
            frequencies = []
            for key in sorted_keys:
                calib_times_array.append(self.calib_times[key])
                frequencies.append(np.nanmean(self.frequencies[key], 0))

            calib_times_array = np.array(calib_times_array)
            frequencies = np.squeeze(frequencies)

            f = interpolate.interp1d(calib_times_array, frequencies, axis=0)
            return f(time)

    def get_frequency_by_time(self, time, position):
        """
        Returns the frequency of a peak position on the
        spectrum for a given time
        :param time: The time
        :param position: The position(s) of the peak(s)
        on the spectrum
        :return: The corresponding frequency in Hz
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

        # TODO Move the interpolation out of this function
        #  (only needs to be done once)
        #  (!!! quite critical because this function is called
        #  for every measurement point)
        # Sort calibration keys by time
        sorted_keys = sorted(self.calib_times,
                             key=self.calib_times.get)
        if not sorted_keys:
            return None

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
            return f(position)
        # Otherwise we can interpolate by time as well
        elif len(calib_times_array) < 3:
            f = interpolate.RegularGridInterpolator(
                (calib_times_array, indices),
                frequencies,
                method='linear',
                bounds_error=False
            )
            return f((time, position))
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
            return f(time, position, grid=False)


class FitSet(object):

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
            if calib_key == key[0]:
                keys.append(key)
        for key in keys:
            del self.fits[key]


class RayleighFitSet(FitSet):

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


class BrillouinFitSet(FitSet):

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


class RayleighFit(object):

    def __init__(self, calib_key, region_key, frame_num,
                 w0, fwhm, intensity, offset):
        self.calib_key = calib_key
        self.region_key = region_key
        self.frame_num = frame_num
        self.w0 = w0
        self.fwhm = fwhm
        self.intensity = intensity
        self.offset = offset


class BrillouinFit(object):

    def __init__(self, calib_key, region_key, frame_num,
                 w0s, fwhms, intensities, offset):
        self.calib_key = calib_key
        self.region_key = region_key
        self.frame_num = frame_num
        self.w0s = w0s
        self.fwhms = fwhms
        self.intensities = intensities
        self.offset = offset
