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
        spectra = self.session.extract_payload_spectrum('0')

        shape_brillouin = (
            resolution[0],           # measurement points in x direction
            resolution[1],           # measurement points in y direction
            resolution[2],           # measurement points in z direction
            len(spectra),            # number of images per measurement point
            len(brillouin_regions),  # number of Brillouin or Rayleigh regions
            evm.nr_brillouin_peaks   # number of peaks to fit per region
        )

        evm.initialize_results_arrays(shape_brillouin)

        # Loop over all measurement positions
        for ind_x in range(resolution[0]):
            for ind_y in range(resolution[1]):
                for ind_z in range(resolution[2]):
                    # Calculate the image key for the given position
                    image_key = str(ind_z * (resolution[0] * resolution[1])
                                    + ind_y * resolution[0] + ind_x)

                    if (abort is not None) & abort.value:
                        if max_count is not None:
                            max_count.value = -1
                        return
                    spectra = self.session.extract_payload_spectrum(
                        image_key
                    )
                    # Loop over all frames per measurement position
                    for frame_num, spectrum in enumerate(spectra):
                        xdata = np.arange(len(spectrum))
                        # Evaluate all selected regions
                        for region_key, region in enumerate(brillouin_regions):
                            w0, fwhm, intensity, offset = \
                                fit_lorentz_region(region, xdata, spectrum)
                            evm.results['brillouin_peak_position'][
                                ind_x, ind_y, ind_z, frame_num, region_key, :]\
                                = w0
                        for region_key, region in enumerate(rayleigh_regions):
                            w0, fwhm, intensity, offset = \
                                fit_lorentz_region(region, xdata, spectrum)

                    if count is not None:
                        count.value += 1

        return
