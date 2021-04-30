import logging
import numpy as np

from bmlab.session import Session
from bmlab.fits import fit_lorentz_region

logger = logging.getLogger(__name__)


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
        spectra, _, _ = self.session.extract_payload_spectrum('0')

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

                    if (abort is not None) & abort.value:
                        self.calculate_derived_values()
                        if max_count is not None:
                            max_count.value = -1
                        return
                    spectra, times, intensities =\
                        self.session.extract_payload_spectrum(image_key)
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

        if (brillouin_peak_f is not None) & \
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

        if (brillouin_peak_right_slope_f is not None) & \
                (brillouin_peak_left_slope_f is not None):
            return abs(
                brillouin_peak_right_slope_f -
                brillouin_peak_left_slope_f
            )
        return None
