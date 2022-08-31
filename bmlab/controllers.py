import logging

import numpy as np
from scipy.signal import medfilt2d, find_peaks
from skimage.measure import label
from skimage.morphology import closing, disk

from math import floor

import multiprocessing as mp
from itertools import repeat as irepeat

from bmlab import Session
from bmlab.fits import fit_vipa, VIPA, fit_lorentz_region
from bmlab.image import extract_lines_along_arc, find_max_in_radius
from bmlab.export import FluorescenceExport,\
    FluorescenceCombinedExport, BrillouinExport

import warnings

logger = logging.getLogger(__name__)


class ExtractionController(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def add_point(self, calib_key, point):
        time = self.session.get_calibration_time(calib_key)
        if time is None:
            return
        em = self.session.extraction_model()
        em.add_point(calib_key, time, *point)

    def set_point(self, calib_key, index, point):
        time = self.session.get_calibration_time(calib_key)
        if time is None:
            return
        em = self.session.extraction_model()
        em.set_point(calib_key, index, time, *point)

    def optimize_points(self, calib_key, radius=10):
        em = self.session.extraction_model()

        imgs = self.session.get_calibration_image(calib_key)
        if imgs is None:
            return
        img = np.nanmean(imgs, axis=0)
        img = medfilt2d(img)

        points = em.get_points(calib_key)
        time = em.get_time(calib_key)
        em.clear_points(calib_key)

        for p in points:
            new_point = find_max_in_radius(img, p, radius)
            # Warning: x-axis in imshow is 1-axis in img, y-axis is 0-axis
            em.add_point(
                calib_key, time, new_point[0], new_point[1])

    def find_points_all(self):
        calib_keys = self.session.get_calib_keys()

        if not calib_keys:
            return

        for calib_key in calib_keys:
            self.find_points(calib_key)

    def find_points(self, calib_key, min_height=10,
                    min_area=20, max_distance=50):
        em = self.session.extraction_model()

        imgs = self.session.get_calibration_image(calib_key)
        if imgs is None:
            return
        img = np.nanmean(imgs, axis=0)
        time = self.session.get_calibration_time(calib_key)
        if time is None:
            return

        # Account for binning for default values
        disc_size = 10
        binning = self.session.get_calibration_binning_factor(calib_key)
        if binning:
            max_distance = max_distance / binning
            disc_size = disc_size / binning
            min_area = min_area / (binning**2)

        img = medfilt2d(img)
        # This is the background level
        threshold = np.median(img)

        # Try to find a signal dependent estimate
        # for the minimal peak height.
        # Discard all values smaller than the background noise
        # plus a minimal peak height
        img_peaks = img[img > threshold + min_height]
        # Calculate the signal dependent peak threshold
        height = (np.nanmean(img_peaks) + threshold) / 2
        img_closed = closing(img > height, disk(disc_size))

        # Find all peaks higher than the min_height
        image_label, num = label(img_closed, return_num=True)

        all_peaks = []
        for region in range(1, num + 1):
            # Mask of everything but the peak
            mask = (image_label != region)
            # Set everything but the peak to zero
            tmp = img.copy()
            tmp[mask] = 0

            # Discard all peaks with an area that is too small
            if np.sum(np.logical_not(mask)) >= min_area:
                # Find indices of peak maximum
                ind = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)

                all_peaks.append(ind)

        # Filter found peaks
        p0 = (0, img.shape[1])
        p1 = (img.shape[0], 0)
        peaks = filter(
            lambda peak:
            self.distance_point_to_line(peak, p0, p1) < max_distance,
            all_peaks)

        # Add found peaks to model
        em.set_points(calib_key, time, list(peaks))

    def distance_point_to_line(self, point, line0, line1):
        return abs(
            (line1[1] - line0[1]) * (line0[0] - point[0]) -
            (line0[1] - point[1]) * (line1[0] - line0[0]))\
               / np.sqrt((line1[1] - line0[1])**2 + (line1[0] - line0[0])**2)


class ImageController(object):

    def __init__(self, model, get_image, get_time, get_exposure):
        self.session = Session.get_instance()
        self.model = model
        self.get_image = get_image
        self.get_time = get_time
        self.get_exposure = get_exposure

    def extract_spectra(self, image_key, frame_num=None):
        em = self.session.extraction_model()
        if not em:
            return None, None, None
        time = self.get_time(image_key)
        arc = em.get_arc_by_time(time)
        if arc.size == 0:
            return None, None, None

        imgs = self.get_image(image_key)
        if frame_num is not None:
            imgs = imgs[frame_num:frame_num+1]

        # Extract values from *all* frames in the current calibration
        spectra = []
        for img in imgs:
            values_by_img = extract_lines_along_arc(
                img,
                arc
            )
            spectra.append(values_by_img)

        exposure = self.get_exposure(image_key)
        times = exposure * np.arange(len(imgs)) + time

        intensities = np.nanmean(imgs, axis=(1, 2))

        # We only set the spectra if we extracted all
        if frame_num is None:
            self.model().set_spectra(image_key, spectra)
        return spectra, times, intensities


class CalibrationController(ImageController):

    def __init__(self, *args, **kwargs):
        session = Session.get_instance()
        super(CalibrationController, self).__init__(
            model=session.calibration_model,
            get_image=session.get_calibration_image,
            get_time=session.get_calibration_time,
            get_exposure=session.get_calibration_exposure
        )
        return

    def find_peaks(self, calib_key, min_prominence=15,
                   num_brillouin_samples=2, min_height=15):
        spectra, _, _ = self.extract_spectra(calib_key)
        if spectra is None:
            return
        spectrum = np.mean(spectra, axis=0)
        # This is the background value
        base = np.nanmedian(spectrum)
        peaks, properties = find_peaks(
            spectrum, prominence=min_prominence, width=True,
            height=min_height+base)

        # Number of peaks we are searching for
        # (2 Rayleigh + 2 times number calibration samples)
        num_peaks = 2 + 2 * num_brillouin_samples

        # Check if we found enough peak candidates
        if len(peaks) < num_peaks:
            # If we didn't find enough peaks, we try again
            # without a minimum height
            peaks, properties = find_peaks(
                spectrum, prominence=min_prominence, width=True)
            # If there a still too few, we give up
            if len(peaks) < num_peaks:
                return

        # We need to identify the position between the
        # Stokes and Anti-Stokes Brillouin peaks

        # In case there are just enough peaks,
        # we use the position in the middle:
        if len(peaks) == num_peaks:
            idx = int(num_peaks / 2)
            center = np.mean(peaks[idx - 1:idx + 1])
        # Otherwise we use the center of mass as the middle
        else:
            # Set everything below the background value to zero,
            # so it does not affect the center calculation
            spectrum[spectrum < base] = 0
            # Calculate the center of mass
            center = np.nansum(spectrum * range(1, len(spectrum) + 1))\
                / np.nansum(spectrum)

            # Check that we have enough peaks on both sides of the center
            num_peaks_right = len(peaks[peaks > center])
            num_peaks_left = len(peaks[peaks <= center])

            # If not enough peaks on the right, shift center to the left
            if num_peaks_right < (num_brillouin_samples + 1):
                center = np.mean(
                    peaks[-(num_brillouin_samples + 2):-num_brillouin_samples]
                )
            # If not enough peaks on the left, shift center to the right
            elif num_peaks_left < (num_brillouin_samples + 1):
                center = np.mean(
                    peaks[num_brillouin_samples:num_brillouin_samples + 2]
                )

        num_peaks_left = len(peaks[peaks <= center])

        indices_brillouin = range(
            num_peaks_left - num_brillouin_samples,
            num_peaks_left + num_brillouin_samples
        )
        indices_rayleigh = [
            num_peaks_left - num_brillouin_samples - 1,
            num_peaks_left + num_brillouin_samples
        ]

        def peak_to_region(i):
            r = (
                        peaks[i]
                        + properties['widths'][i] * np.array((-4, 4))
                ).astype(int)
            r[r > len(spectrum)] = len(spectrum)
            return tuple(r)

        regions_brillouin = list(map(peak_to_region, indices_brillouin))
        # Merge the Brillouin regions if necessary
        if num_brillouin_samples > 1:
            regions_brillouin = [
                (regions_brillouin[0][0],
                 regions_brillouin[num_brillouin_samples - 1][1]),
                (regions_brillouin[num_brillouin_samples][0],
                 regions_brillouin[-1][1]),
            ]

        regions_rayleigh = map(peak_to_region, indices_rayleigh)

        cm = self.session.calibration_model()
        # Add Brillouin regions
        for i, region in enumerate(regions_brillouin):
            # We use "set_brillouin_region" here so overlapping
            # regions don't get merged
            cm.set_brillouin_region(calib_key, i, region)
        # Add Rayleigh regions
        for i, region in enumerate(regions_rayleigh):
            # We use "set_brillouin_region" here so overlapping
            # regions don't get merged
            cm.set_rayleigh_region(calib_key, i, region)

    def calibrate(self, calib_key, count=None, max_count=None):

        setup = self.session.setup
        em = self.session.extraction_model()
        cm = self.session.calibration_model()

        if not calib_key\
                or not setup\
                or not em\
                or not cm:
            if max_count is not None:
                max_count.value = -1
            return

        spectra, _, _ = self.extract_spectra(calib_key)
        time = self.session.get_calibration_time(calib_key)

        if spectra is None or len(spectra) == 0:
            if max_count is not None:
                max_count.value = -1
            return

        self.fit_rayleigh_regions(calib_key)
        self.fit_brillouin_regions(calib_key)

        vipa_params = []
        frequencies = []

        if max_count is not None:
            max_count.value += len(spectra)

        for frame_num, spectrum in enumerate(spectra):
            peaks = cm.get_sorted_peaks(calib_key, frame_num)

            params = fit_vipa(peaks, setup)
            if params is None:
                continue
            vipa_params.append(params)
            xdata = np.arange(len(spectrum))

            frequencies.append(VIPA(xdata, params) - setup.f0)
            if count is not None:
                count.value += 1

        cm.set_vipa_params(calib_key, vipa_params)
        cm.set_frequencies(calib_key, time, frequencies)

        calculate_derived_values()

    def clear_calibration(self, calib_key):
        cm = self.session.calibration_model()
        if not cm:
            return

        cm.clear_brillouin_fits(calib_key)
        cm.clear_rayleigh_fits(calib_key)
        cm.clear_frequencies(calib_key)
        cm.clear_vipa_params(calib_key)

        calculate_derived_values()

    def fit_rayleigh_regions(self, calib_key):
        cm = self.session.calibration_model()
        spectra = cm.get_spectra(calib_key)
        regions = cm.get_rayleigh_regions(calib_key)

        cm.clear_rayleigh_fits(calib_key)
        for frame_num, spectrum in enumerate(spectra):
            for region_key, region in enumerate(regions):
                xdata = np.arange(len(spectrum))
                w0, fwhm, intensity, offset = \
                    fit_lorentz_region(region, xdata, spectrum)
                cm.add_rayleigh_fit(calib_key, region_key, frame_num,
                                    w0, fwhm, intensity, offset)

    def fit_brillouin_regions(self, calib_key):
        cm = self.session.calibration_model()
        spectra = cm.get_spectra(calib_key)
        regions = cm.get_brillouin_regions(calib_key)
        setup = self.session.setup
        if not setup:
            return

        cm.clear_brillouin_fits(calib_key)
        for frame_num, spectrum in enumerate(spectra):
            for region_key, region in enumerate(regions):
                xdata = np.arange(len(spectrum))
                w0s, fwhms, intensities, offset = \
                    fit_lorentz_region(
                        region,
                        xdata,
                        spectrum,
                        setup.calibration.num_brillouin_samples
                    )
                cm.add_brillouin_fit(calib_key, region_key, frame_num,
                                     w0s, fwhms, intensities, offset)

    def expected_frequencies(self, calib_key=None, current_frame=None):
        cm = self.session.calibration_model()

        if calib_key not in cm.vipa_params or \
                current_frame > len(cm.vipa_params[calib_key]) - 1:
            return None

        return self.session.setup.calibration.shifts \
            + self.session.setup.calibration.orders \
            * cm.vipa_params[calib_key][current_frame][3]


class PeakSelectionController(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def add_brillouin_region_frequency(self, region_frequency):
        self.add_region_frequency(region_frequency, 'Brillouin')

    def add_rayleigh_region_frequency(self, region_frequency):
        self.add_region_frequency(region_frequency, 'Rayleigh')

    def add_region_frequency(self, region_frequency, peak_type):
        cm = self.session.calibration_model()
        # We use the first measurement image here
        time = self.session.get_payload_time('0')
        region_pix = cm.get_position_by_time(time, list(region_frequency))
        if region_pix is None:
            return

        psm = self.session.peak_selection_model()
        if peak_type == 'Brillouin':
            psm.add_brillouin_region(tuple(region_pix))
        if peak_type == 'Rayleigh':
            psm.add_rayleigh_region(tuple(region_pix))


class EvaluationController(ImageController):

    def __init__(self, *args, **kwargs):
        session = Session.get_instance()
        super(EvaluationController, self).__init__(
            model=session.evaluation_model,
            get_image=session.get_payload_image,
            get_time=session.get_payload_time,
            get_exposure=session.get_payload_exposure
        )
        return

    def set_nr_brillouin_peaks(self, nr_brillouin_peaks):
        evm = self.session.evaluation_model()
        if not evm:
            return
        evm.setNrBrillouinPeaks(nr_brillouin_peaks)

    def set_bounds(self, bounds):
        evm = self.session.evaluation_model()
        if not evm:
            return
        evm.bounds = bounds

    def evaluate(self, abort=None, count=None, max_count=None):
        em = self.session.extraction_model()
        if not em:
            if max_count is not None:
                max_count.value = -1
            return

        cm = self.session.calibration_model()
        if not cm:
            if max_count is not None:
                max_count.value = -1
            return

        pm = self.session.peak_selection_model()
        if not pm:
            if max_count is not None:
                max_count.value = -1
            return

        evm = self.session.evaluation_model()
        if not evm:
            if max_count is not None:
                max_count.value = -1
            return

        image_keys = self.session.get_image_keys(True)

        if max_count is not None:
            max_count.value = len(image_keys)

        brillouin_regions = pm.get_brillouin_regions()
        rayleigh_regions = pm.get_rayleigh_regions()

        resolution = self.session.get_payload_resolution()

        # Get first spectrum to find number of images
        spectra, _, _ = self.extract_spectra('0')

        if not spectra:
            if max_count is not None:
                max_count.value = -1
            return

        # We create a variable for this value here,
        # so changing nr_brillouin_peaks during evaluation
        # does not create issues
        nr_brillouin_peaks = evm.nr_brillouin_peaks
        evm.initialize_results_arrays({
            # measurement points in x direction
            'dim_x': resolution[0],
            # measurement points in y direction
            'dim_y': resolution[1],
            # measurement points in z direction
            'dim_z': resolution[2],
            # number of images per measurement point
            'nr_images': len(spectra),
            # number of Brillouin regions
            'nr_brillouin_regions': len(brillouin_regions),
            # number of peaks to fit per region
            'nr_brillouin_peaks': nr_brillouin_peaks,
            # number of Rayleigh regions
            'nr_rayleigh_regions': len(rayleigh_regions),
        })

        pool_size = mp.cpu_count() * 2
        pool = mp.Pool(processes=pool_size)
        # Initialize the Rayleigh shift
        # used for compensating drifts
        rayleigh_peak_initial =\
            np.nan * np.ones((len(spectra), len(rayleigh_regions), 1))
        rayleigh_shift = 0
        # Loop over all measurement positions
        for idx, image_key in enumerate(image_keys):
            # Calculate the indices for the given key
            (ind_x, ind_y, ind_z) =\
                self.get_indices_from_key(resolution, image_key)

            if (abort is not None) and abort.value:
                calculate_derived_values()
                if max_count is not None:
                    max_count.value = -1
                return
            spectra, times, intensities =\
                self.extract_spectra(image_key)
            if spectra is None:
                continue
            evm.results['time'][ind_x, ind_y, ind_z, :, 0, 0] =\
                times
            evm.results['intensity'][ind_x, ind_y, ind_z, :, 0, 0] =\
                intensities

            # We shift the evaluated regions here
            # to compensate for eventual drift
            brillouin_regions_shifted = [
                tuple([val + rayleigh_shift for val in region])
                for region in brillouin_regions]
            rayleigh_regions_shifted = [
                tuple([val + rayleigh_shift for val in region])
                for region in rayleigh_regions]

            # Pack the data for parallel processing
            regions = brillouin_regions_shifted + rayleigh_regions_shifted
            packed_data = zip(irepeat(spectra), regions)
            # Process it
            results = pool.starmap(self.fit_spectra, packed_data)

            # Unpack the results
            for frame_num, spectrum in enumerate(spectra):
                for region_key, _ in enumerate(brillouin_regions):
                    ind = (ind_x, ind_y, ind_z,
                           frame_num, region_key, 0)
                    evm.results['brillouin_peak_position'][ind] =\
                        results[region_key][frame_num][0]
                    evm.results['brillouin_peak_fwhm'][ind] =\
                        results[region_key][frame_num][1]
                    evm.results['brillouin_peak_intensity'][ind] =\
                        results[region_key][frame_num][2]
                    evm.results['brillouin_peak_offset'][ind] =\
                        results[region_key][frame_num][3]

                for region_key, _ \
                        in enumerate(rayleigh_regions,
                                     start=len(brillouin_regions)):
                    ind = (ind_x, ind_y, ind_z, frame_num,
                           region_key - len(brillouin_regions))
                    evm.results['rayleigh_peak_position'][ind] =\
                        results[region_key][frame_num][0]
                    evm.results['rayleigh_peak_fwhm'][ind] =\
                        results[region_key][frame_num][1]
                    evm.results['rayleigh_peak_intensity'][ind] =\
                        results[region_key][frame_num][2]
                    evm.results['rayleigh_peak_offset'][ind] =\
                        results[region_key][frame_num][3]

            # We can only do a multi-peak fit after the single-peak
            # Rayleigh fit is done, because we have to know the
            # Rayleigh peak positions in GHz in order to convert
            # the multi-peak fit bounds given in GHz into the position
            # in pixels.
            if nr_brillouin_peaks > 1:
                ind =\
                    (ind_x, ind_y, ind_z, slice(None), slice(None), 0)
                rayleigh_peaks = np.transpose(
                    evm.results['rayleigh_peak_position'][ind]
                )
                bounds = self.create_bounds(
                    brillouin_regions_shifted,
                    times,
                    rayleigh_peaks
                )
                if bounds is not None:
                    packed_data_multi_peak =\
                        zip(irepeat(spectra),
                            brillouin_regions_shifted,
                            irepeat(nr_brillouin_peaks),
                            bounds)
                else:
                    packed_data_multi_peak =\
                        zip(irepeat(spectra),
                            brillouin_regions_shifted,
                            irepeat(nr_brillouin_peaks),
                            irepeat(bounds))
                # Process it
                results_multi_peak = pool.starmap(
                    self.fit_spectra, packed_data_multi_peak)

                for frame_num, spectrum in enumerate(spectra):
                    for region_key, _ in enumerate(
                            brillouin_regions):
                        ind = (ind_x, ind_y, ind_z,
                               frame_num, region_key,
                               slice(1, nr_brillouin_peaks+1))
                        evm.results[
                            'brillouin_peak_position'][ind] = \
                            results_multi_peak[
                                region_key][frame_num][0]
                        evm.results[
                            'brillouin_peak_fwhm'][ind] = \
                            results_multi_peak[
                                region_key][frame_num][1]
                        evm.results[
                            'brillouin_peak_intensity'][ind] = \
                            results_multi_peak[
                                region_key][frame_num][2]
                        evm.results[
                            'brillouin_peak_offset'][ind] = \
                            results_multi_peak[
                                region_key][frame_num][3]

            # Calculate the shift of the Rayleigh peaks,
            # in order to follow the peaks in case of a drift
            rayleigh_peak_current = evm.results['rayleigh_peak_position'][
                    ind_x, ind_y, ind_z, :, :, :]
            # If we haven't found a valid Rayleigh peak position,
            # but the current one is valid, use it
            if not np.isnan(rayleigh_peak_current).all()\
                    and np.isnan(rayleigh_peak_initial).all():
                rayleigh_peak_initial = rayleigh_peak_current
            shift = rayleigh_peak_current - rayleigh_peak_initial
            evm.results['rayleigh_shift'][ind_x, ind_y, ind_z, :, :, :] = shift
            if not np.isnan(shift).all():
                rayleigh_shift = round(np.nanmean(shift))

            if count is not None:
                count.value += 1

            # Calculate the derived values every ten steps
            if not (idx % 10):
                calculate_derived_values()

        pool.close()
        pool.join()
        calculate_derived_values()

        return

    @staticmethod
    def fit_spectra(spectra, region, nr_peaks=1, bounds_w0=None):
        fits = []
        for frame_num, spectrum in enumerate(spectra):
            xdata = np.arange(len(spectrum))
            if bounds_w0 is None:
                fit = fit_lorentz_region(region, xdata, spectrum,
                                         nr_peaks)
            else:
                fit = fit_lorentz_region(region, xdata, spectrum,
                                         nr_peaks, bounds_w0[frame_num])
            fits.append(fit)
        return fits

    def create_bounds(self, brillouin_regions, times, rayleigh_peaks):
        """
        This function converts the bounds settings into
        a bounds object for the fitting function
        Allowed parameters for the bounds settings are
        - 'min'/'max'   -> Will be converted to the respective
            lower or upper limit of the given region
        - '-Inf', 'Inf' -> Will be converted to -np.Inf or np.Inf
        - number [GHz]  -> Will be converted into the pixel
            position of the given frequency
        Parameters
        ----------
        brillouin_regions
        times
        rayleigh_peaks

        Returns
        -------

        """
        cm = self.session.calibration_model()
        evm = self.session.evaluation_model()
        bounds = evm.bounds
        if bounds is None:
            return None

        # We need a Rayleigh peak position for
        # every combination of brillouin_region and time
        if rayleigh_peaks.shape != (len(brillouin_regions), len(times)):
            return None

        w0_bounds = []
        # We have to create a separate bound for every region
        for region_idx, region in enumerate(brillouin_regions):
            local_time = []
            for time_idx, time in enumerate(times):
                # In case this is an Anti-Stokes peak, we find the peak
                # with the higher frequency on the left hand side and
                # have to flip the bounds
                is_anti_stokes = np.mean(region) <\
                                 rayleigh_peaks[region_idx][time_idx]
                f_rayleigh = cm.get_frequency_by_time(
                    time,
                    rayleigh_peaks[region_idx][time_idx]
                )[()]

                local_bound = []
                for bound in bounds:
                    local_limit = []
                    for limit in bound:
                        if limit.lower() == 'min':
                            val = region[is_anti_stokes]
                        elif limit.lower() == 'max':
                            val = region[not is_anti_stokes]
                        elif limit.lower() == '-inf':
                            val = -((-1) ** is_anti_stokes)\
                                          * np.Inf
                        elif limit.lower() == 'inf':
                            val = ((-1) ** is_anti_stokes)\
                                          * np.Inf
                        else:
                            # Try to convert the value in GHz into
                            # a value in pixel depending on the time
                            try:
                                f = ((-1) ** is_anti_stokes)\
                                          * 1e9 * float(limit) + f_rayleigh
                                val = cm.get_position_by_time(time, f)[()]
                            except BaseException:
                                val = np.Inf

                        local_limit.append(val)
                    # Check that the bounds are sorted ascendingly
                    # (for anti-stokes, they might not).
                    local_limit.sort()
                    local_bound.append(local_limit)
                local_time.append(local_bound)
            w0_bounds.append(local_time)

        return w0_bounds

    def get_data(self, parameter_key, brillouin_peak_index=0):
        """
        This function returns the evaluated data,
        its positions, dimensionality and labels
        Parameters
        ----------
        parameter_key: str
            The key of the parameter requested.
            See bmlab.model.evaluation_model.get_parameter_keys()
        brillouin_peak_index: int
            The index of the Brillouin peak to show in case
            we did a multi-peak fit
            0: the single-peak fit
            1:nr_brillouin_peaks: the multi-peak fits
            nr_brillouin_peaks+1: all multi-peak fits average
            nr_brillouin_peaks+2: all multi-peak fits weighted average

        Returns
        -------
        data: np.ndarray
            The data to show. This is always a 3-dimensional array.
        positions: list
            This is a list of length 3 containing ndarrays with
            the spatial positions of the data points.
        dimensionality: int
            Whether it's a 0, 1, 2, or 3D measurement
        labels: list
            The labels of the positions
        """
        resolution = self.session.get_payload_resolution()

        dimensionality = sum(np.array(resolution) > 1)

        # Get the positions and squeeze them
        pos = self.session.get_payload_positions()

        positions = list(pos.values())
        labels = list(map(lambda l: r'$' + l + '$ [$\\mu$m]', ['x', 'y', 'z']))

        evm = self.session.evaluation_model()
        data = evm.results[parameter_key]

        # Ensure that we always get the expected shape
        # (even if the array was not initialized yet)
        if data.size == 0:
            data = np.empty(resolution)
            data[:] = np.nan

        # Slice the appropriate Brillouin peak if necessary and possible
        if data.ndim >= 6:
            nr_peaks_stored = data.shape[5]
            if nr_peaks_stored > 1\
                    and brillouin_peak_index < nr_peaks_stored + 2:
                if brillouin_peak_index < nr_peaks_stored:
                    sliced = data[:, :, :, :, :, brillouin_peak_index]
                # Average all multi-peak fits
                if brillouin_peak_index == nr_peaks_stored:
                    sliced = data[:, :, :, :, :, 1:]
                # Weighted average of all multi-peak fits
                if brillouin_peak_index == nr_peaks_stored + 1:
                    weight =\
                        evm.results[
                            'brillouin_peak_intensity'][:, :, :, :, :, 1:]\
                        * evm.results['brillouin_peak_fwhm'][:, :, :, :, :, 1:]
                    sliced = \
                        np.nansum(data[:, :, :, :, :, 1:] * weight, axis=5)\
                        / np.nansum(weight, axis=5)
            else:
                sliced = data[:, :, :, :, :, 0]
        else:
            sliced = data

        # Average all non-spatial dimensions.
        # Do not show warning which occurs when a slice contains only NaNs.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message='Mean of empty slice'
            )
            data = np.nanmean(
                sliced,
                axis=tuple(range(3, sliced.ndim))
            )

        # Scale the date in case of GHz
        data = evm.parameters[parameter_key]['scaling'] * data

        return data, positions, dimensionality, labels

    def get_fits(self, image_key):
        resolution = self.session.get_payload_resolution()
        indices = self.get_indices_from_key(resolution, image_key)

        evm = self.session.evaluation_model()
        if not evm:
            return
        return evm.get_fits(*indices)

    @staticmethod
    def get_key_from_indices(resolution, ind_x, ind_y, ind_z):
        if len(resolution) != 3:
            raise ValueError('resolution has wrong dimension')
        if ind_x >= resolution[0]:
            raise IndexError('x index out of range')
        if ind_y >= resolution[1]:
            raise IndexError('y index out of range')
        if ind_z >= resolution[2]:
            raise IndexError('z index out of range')
        return str(int(ind_z * (resolution[0] * resolution[1])
                   + ind_y * resolution[0] + ind_x))

    @staticmethod
    def get_indices_from_key(resolution, key):
        key = int(key)
        ind_z = floor(key / (resolution[0] * resolution[1]))
        ind_y = floor(
            (key - ind_z * (resolution[0] * resolution[1])) / resolution[0])
        ind_x = (key % (resolution[0] * resolution[1])) % resolution[0]
        # ind_y = (key - ind_x) % resolution[0]
        if ind_x >= resolution[0]\
                or ind_y >= resolution[1]\
                or ind_z >= resolution[2]:
            raise ValueError('Invalid key')
        return ind_x, ind_y, ind_z


def calculate_derived_values():
    """
    We calculate the derived parameters here:
    - Brillouin shift [pix]
    - Brillouin shift [GHz]
    - Brillouin peak width [GHz]
    - Rayleigh peak width [GHz]
    """
    session = Session.get_instance()
    evm = session.evaluation_model()
    if not evm:
        return

    if len(evm.results['time']) == 0:
        return

    time = evm.results['time']
    shape_brillouin = evm.results['brillouin_peak_position'].shape
    shape_rayleigh = evm.results['rayleigh_peak_position'].shape
    # If we have the same number of Rayleigh and Brillouin regions,
    # we can simply subtract the two arrays (regions are always
    # sorted by center in the peak selection model, so corresponding
    # regions should be at the same array index)
    if shape_brillouin[4] == shape_rayleigh[4]:
        evm.results['brillouin_shift'] = abs(
            evm.results['brillouin_peak_position'] -
            evm.results['rayleigh_peak_position']
        )

        # Calculate shift in GHz
        shift = calculate_shift_f(
            time,
            evm.results['brillouin_peak_position'],
            evm.results['rayleigh_peak_position']
        )
        if shift is not None:
            evm.results['brillouin_shift_f'] = shift

    # Having a different number of Rayleigh and Brillouin regions
    # doesn't really make sense. But in case I am missing something
    # here, we assign each Brillouin region the nearest (by center)
    # Rayleigh region.
    else:
        psm = session.peak_selection_model()
        if not psm:
            return
        brillouin_centers = list(map(np.mean, psm.get_brillouin_regions()))
        rayleigh_centers = list(map(np.mean, psm.get_rayleigh_regions()))

        for idx in range(len(brillouin_centers)):
            # Find corresponding (nearest) Rayleigh region
            d = list(
                map(
                    lambda x: abs(x - brillouin_centers[idx]),
                    rayleigh_centers
                )
            )
            idx_r = d.index(min(d))
            evm.results['brillouin_shift'][:, :, :, :, idx, :] = abs(
                evm.results[
                    'brillouin_peak_position'][:, :, :, :, idx, :] -
                evm.results[
                    'rayleigh_peak_position'][:, :, :, :, idx_r, :]
            )

            # Calculate shift in GHz
            shift = calculate_shift_f(
                time[:, :, :, :, 0, :],
                evm.results['brillouin_peak_position'][:, :, :, :, idx, :],
                evm.results['rayleigh_peak_position'][:, :, :, :, idx_r, :]
            )
            if shift is not None:
                evm.results['brillouin_shift_f'][:, :, :, :, idx, :] =\
                    shift

    # Calculate FWHM in GHz
    fwhm_brillouin = calculate_fwhm_f(
        time,
        evm.results['brillouin_peak_position'],
        evm.results['brillouin_peak_fwhm']
    )
    if fwhm_brillouin is not None:
        evm.results['brillouin_peak_fwhm_f'] = fwhm_brillouin

    fwhm_rayleigh = calculate_fwhm_f(
        time,
        evm.results['rayleigh_peak_position'],
        evm.results['rayleigh_peak_fwhm']
    )
    if fwhm_brillouin is not None:
        evm.results['rayleigh_peak_fwhm_f'] = fwhm_rayleigh


def calculate_shift_f(time, brillouin_position, rayleigh_position):
    session = Session.get_instance()
    cm = session.calibration_model()
    if not cm:
        return np.full(np.shape(brillouin_position), np.nan)
    brillouin_peak_f = cm.get_frequency_by_time(
        time,
        brillouin_position
    )
    rayleigh_peak_f = cm.get_frequency_by_time(
        time,
        rayleigh_position
    )

    if (brillouin_peak_f is not None) and \
            (rayleigh_peak_f is not None):
        return abs(
            brillouin_peak_f -
            rayleigh_peak_f
        )
    return np.full(np.shape(brillouin_position), np.nan)


def calculate_fwhm_f(time, peak_position, peak_fwhm):
    session = Session.get_instance()
    cm = session.calibration_model()
    if not cm:
        return np.full(np.shape(peak_position), np.nan)
    brillouin_peak_right_slope_f = cm.get_frequency_by_time(
        time,
        peak_position + peak_fwhm/2
    )
    brillouin_peak_left_slope_f = cm.get_frequency_by_time(
        time,
        peak_position - peak_fwhm/2
    )

    if (brillouin_peak_right_slope_f is not None) and \
            (brillouin_peak_left_slope_f is not None):
        return abs(
            brillouin_peak_right_slope_f -
            brillouin_peak_left_slope_f
        )
    return np.full(np.shape(peak_position), np.nan)


class Controller(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def evaluate(self, filepath, setup, orientation,
                 brillouin_regions, rayleigh_regions,
                 repetitions=None, nr_brillouin_peaks=1,
                 multi_peak_bounds=None):
        # Load data file
        self.session.set_file(filepath)

        # Evaluate all repetitions if not requested differently
        if repetitions is None:
            repetitions = self.session.file.repetition_keys()

        for repetition in repetitions:
            # Select repetition
            self.session.set_current_repetition(repetition)
            self.session.set_setup(setup)

            # Set orientation
            self.session.orientation = orientation

            ec = ExtractionController()
            cc = CalibrationController()
            psc = PeakSelectionController()
            evc = EvaluationController()

            # First add all extraction points because this
            # can influence the extraction for other calibrations
            ec.find_points_all()

            # Then do the calibration
            for calib_key in self.session.get_calib_keys():
                cc.find_peaks(calib_key)

                cc.calibrate(calib_key)

            for region in brillouin_regions:
                psc.add_brillouin_region_frequency(region)

            for region in rayleigh_regions:
                psc.add_rayleigh_region_frequency(region)

            evc.set_nr_brillouin_peaks(nr_brillouin_peaks)
            evc.set_bounds(multi_peak_bounds)

            evc.evaluate()

        return self.session


class ExportController(object):

    def __init__(self):
        return

    @staticmethod
    def get_configuration():
        return {
            'fluorescence': {
                'export': True,
            },
            'fluorescenceCombined': {
                'export': True,
            },
            'brillouin': {
                'export': True,
                'parameters': ['brillouin_shift_f'],
                'brillouin_shift_f': {
                    'cax': ('min', 'max'),
                }
            },
        }

    def export(self, configuration=None):
        if not configuration:
            configuration = self.get_configuration()

        FluorescenceExport().export(configuration)
        FluorescenceCombinedExport().export(configuration)

        # BrillouinExport needs the EvaluationController
        # to nicely get the data, so we provide it here.
        # Not really nice, but importing it in BrillouinExport
        # leads to a circular dependency.
        BrillouinExport(EvaluationController()).export(configuration)
