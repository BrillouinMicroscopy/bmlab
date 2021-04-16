import logging

import numpy as np
from scipy import interpolate

logger = logging.getLogger(__name__)


class CalibrationModel(object):

    def __init__(self):
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

        self.regions_merge_add_region(
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

        self.regions_merge_add_region(
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
            if (key[0] == calib_key) & (key[2] == frame_num):
                peaks.append(fit.w0)

        for key, fit in self.brillouin_fits.fits.items():
            if (key[0] == calib_key) & (key[2] == frame_num):
                for w0 in fit.w0s:
                    peaks.append(w0)

        return np.sort(np.array(peaks))

    def set_vipa_params(self, calib_key, vipa_params):
        self.vipa_params[calib_key] = vipa_params

    def set_frequencies(self, calib_key, frequencies):
        self.frequencies[calib_key] = frequencies

    def get_frequencies_by_calib_key(self, calib_key):
        if calib_key in self.frequencies:
            return self.frequencies[calib_key]

    def get_frequency_by_calib_key(self, position, calib_key):
        """
        :param position:
        :param calib_key:
        :return:
        """
        # TODO Move the interpolation out of this function
        #  (only needs to be done once)
        frequencies = self.get_frequencies_by_calib_key(calib_key)
        frequency = np.mean(np.array(frequencies), axis=0)

        xdata = np.arange(len(frequency))
        f = interpolate.interp1d(xdata, frequency)
        return f(position)

    # TODO To be implemented
    def get_frequency_by_time(self, position, time):
        """
        Returns the frequency of a given peak position
        and time
        :param position: The position of the peak on the
        spectrum
        :param time: The time the peak was acquired
        :return: The corresponding frequency in Hz
        """
        return None

    @staticmethod
    def regions_merge_add_region(regions, region):
        regions_fused = False

        # check if the selected regions overlap
        for i, saved_region in enumerate(regions):
            if (np.min(region) < np.max(saved_region)
                    and (np.max(region) > np.min(saved_region))):
                # fuse overlapping regions
                regions[i] = (
                    np.min([region, saved_region]),
                    np.max([region, saved_region]))
                regions_fused = True

        if not regions_fused:
            regions.append(region)


class FitSet(object):

    def __init__(self):
        self.fits = {}

    def add_fit(self, fit):
        key = (fit.calib_key, fit.region_key, fit.frame_num)
        self.fits[key] = fit

    def get_fit(self, calib_key, region_key, frame_num=None):
        key = (calib_key, region_key, frame_num)
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
        w0s = [fit.w0
               for (calib_key_, region_key_, frame_num_), fit
               in self.fits.items()
               if calib_key == calib_key_ and region_key == region_key_]
        logger.debug('w0s = ', w0s)
        if w0s:
            return np.mean(w0s)
        return None


class BrillouinFitSet(FitSet):

    def average_fits(self, calib_key, region_key):
        w0s = [fit.w0s
               for (calib_key_, region_key_, frame_num_), fit
               in self.fits.items()
               if calib_key == calib_key_ and region_key == region_key_]
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
