import logging

import numpy as np

from bmlab import Session
from bmlab.fits import fit_vipa, VIPA, fit_lorentz_region
from bmlab.image import extract_lines_along_arc, find_max_in_radius

logger = logging.getLogger(__name__)


class CalibrationController(object):

    def __init__(self):
        self.session = Session.get_instance()
        self.setup = self.session.setup
        return

    def calibrate(self, calib_key, count=None, max_count=None):

        if not calib_key:
            if max_count is not None:
                max_count.value = -1
            return

        if not self.setup:
            if max_count is not None:
                max_count.value = -1
            return

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

        spectra = self.extract_calibration_spectra(calib_key)
        time = self.session.get_calibration_time(calib_key)

        if spectra is None:
            if max_count is not None:
                max_count.value = -1
            return

        if len(spectra) == 0:
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

            params = fit_vipa(peaks, self.setup)
            if params is None:
                continue
            vipa_params.append(params)
            xdata = np.arange(len(spectrum))

            frequencies.append(VIPA(xdata, params) - self.setup.f0)
            if count is not None:
                count.value += 1

        cm.set_vipa_params(calib_key, vipa_params)
        cm.set_frequencies(calib_key, time, frequencies)

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

        cm.clear_brillouin_fits(calib_key)
        for frame_num, spectrum in enumerate(spectra):
            for region_key, region in enumerate(regions):
                xdata = np.arange(len(spectrum))
                w0s, fwhms, intensities, offset = \
                    fit_lorentz_region(
                        region,
                        xdata,
                        spectrum,
                        self.setup.calibration.num_brillouin_samples
                    )
                cm.add_brillouin_fit(calib_key, region_key, frame_num,
                                     w0s, fwhms, intensities, offset)

    def extract_calibration_spectra(self, calib_key, frame_num=None):
        em = self.session.extraction_model()
        cm = self.session.calibration_model()
        if not em:
            return
        arc = em.get_arc_by_calib_key(calib_key)
        if arc.size == 0:
            return

        imgs = self.session.get_calibration_image(calib_key)
        if frame_num is not None:
            imgs = imgs[frame_num:1]

        # Extract values from *all* frames in the current calibration
        spectra = []
        for img in imgs:
            values_by_img = extract_lines_along_arc(
                img,
                arc
            )
            spectra.append(values_by_img)
        cm.set_spectra(calib_key, spectra)
        return spectra


class EvaluationController(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

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

        image_keys = self.session.get_image_keys()

        if max_count is not None:
            max_count.value += len(image_keys)

        brillouin_regions = pm.get_brillouin_regions()
        rayleigh_regions = pm.get_rayleigh_regions()

        resolution = self.session.current_repetition().payload.resolution

        # Get first spectrum to find number of images
        spectra, _, _ = self.extract_payload_spectra('0')

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
            'nr_brillouin_peaks': evm.nr_brillouin_peaks,
            # number of Rayleigh regions
            'nr_rayleigh_regions': len(rayleigh_regions),
        })

        # Loop over all measurement positions
        for ind_x in range(resolution[0]):
            for ind_y in range(resolution[1]):
                for ind_z in range(resolution[2]):
                    # Calculate the image key for the given position
                    image_key = str(ind_z * (resolution[0] * resolution[1])
                                    + ind_y * resolution[0] + ind_x)

                    if (abort is not None) and abort.value:
                        self.calculate_derived_values()
                        if max_count is not None:
                            max_count.value = -1
                        return
                    spectra, times, intensities =\
                        self.extract_payload_spectra(image_key)
                    evm.results['time'][ind_x, ind_y, ind_z, :, 0, 0] =\
                        times
                    evm.results['intensity'][ind_x, ind_y, ind_z, :, 0, 0] =\
                        intensities
                    # Loop over all frames per measurement position
                    for frame_num, spectrum in enumerate(spectra):
                        xdata = np.arange(len(spectrum))
                        # Evaluate all selected regions
                        for region_key, region in enumerate(brillouin_regions):
                            ind = (ind_x, ind_y, ind_z,
                                   frame_num, region_key, 0)
                            w0, fwhm, intensity, offset = \
                                fit_lorentz_region(region, xdata, spectrum)
                            # Save results into arrays
                            evm.results['brillouin_peak_position'][ind] = w0
                            evm.results['brillouin_peak_fwhm'][ind] = fwhm
                            evm.results['brillouin_peak_intensity'][ind] =\
                                intensity
                        for region_key, region in enumerate(rayleigh_regions):
                            ind = (ind_x, ind_y, ind_z,
                                   frame_num, region_key)
                            w0, fwhm, intensity, offset = \
                                fit_lorentz_region(region, xdata, spectrum)
                            # Save results into arrays
                            evm.results['rayleigh_peak_position'][ind] = w0
                            evm.results['rayleigh_peak_fwhm'][ind] = fwhm
                            evm.results['rayleigh_peak_intensity'][ind] =\
                                intensity

                    if count is not None:
                        count.value += 1

        self.calculate_derived_values()

        return

    def extract_payload_spectra(self, image_key):
        em = self.session.extraction_model()
        evm = self.session.evaluation_model()
        if not em:
            return
        time = self.session.get_payload_time(image_key)
        arc = em.get_arc_by_time(time)
        if arc.size == 0:
            return

        imgs = self.session.get_payload_image(image_key)

        # Extract values from *all* frames in the current payload
        spectra = []
        for img in imgs:
            values_by_img = extract_lines_along_arc(
                img,
                arc
            )
            spectra.append(values_by_img)

        exposure = self.session.current_repetition()\
            .payload.get_exposure(image_key)
        times = exposure * np.arange(len(imgs)) + time

        intensities = np.nanmean(imgs, axis=(1, 2))

        evm.set_spectra(image_key, spectra)
        return spectra, times, intensities

    def calculate_derived_values(self):
        """
        We calculate the derived parameters here:
        - Brillouin shift [pix]
        - Brillouin shift [GHz]
        - Brillouin peak width [GHz]
        - Rayleigh peak width [GHz]
        :return:
        """
        evm = self.session.evaluation_model()
        if not evm:
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
            shift = self.calculate_shift_f(
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
            psm = self.session.peak_selection_model()
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
                shift = self.calculate_shift_f(
                    time[:, :, :, :, 0, :],
                    evm.results['brillouin_peak_position'][:, :, :, :, idx, :],
                    evm.results['rayleigh_peak_position'][:, :, :, :, idx_r, :]
                )
                if shift is not None:
                    evm.results['brillouin_shift_f'][:, :, :, :, idx, :] =\
                        shift

        # Calculate FWHM in GHz
        fwhm_brillouin = self.calculate_fwhm_f(
            time,
            evm.results['brillouin_peak_position'],
            evm.results['brillouin_peak_fwhm']
        )
        if fwhm_brillouin is not None:
            evm.results['brillouin_peak_fwhm_f'] = fwhm_brillouin

        fwhm_rayleigh = self.calculate_fwhm_f(
            time,
            evm.results['rayleigh_peak_position'],
            evm.results['rayleigh_peak_fwhm']
        )
        if fwhm_brillouin is not None:
            evm.results['rayleigh_peak_fwhm_f'] = fwhm_rayleigh

    def calculate_shift_f(self, time, brillouin_position, rayleigh_position):
        cm = self.session.calibration_model()
        if not cm:
            return None
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
        return None

    def calculate_fwhm_f(self, time, peak_position, peak_fwhm):
        cm = self.session.calibration_model()
        if not cm:
            return None
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
        return None


class ExtractionController(object):

    def add_point(self, calib_key, point):
        session = Session.get_instance()
        time = session.get_calibration_time(calib_key)
        em = session.extraction_model()
        em.add_point(calib_key, time, *point)

    def optimize_points(self, calib_key, radius=10):
        session = Session.get_instance()
        em = session.extraction_model()

        img = session.get_calibration_image(calib_key, 0)

        points = em.get_points(calib_key)
        time = em.get_time(calib_key)
        em.clear_points(calib_key)

        for p in points:
            new_point = find_max_in_radius(img, p, radius)
            # Warning: x-axis in imshow is 1-axis in img, y-axis is 0-axis
            em.add_point(
                calib_key, time, new_point[0], new_point[1])
