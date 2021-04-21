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

        image_keys = self.session.get_image_keys()

        if max_count is not None:
            max_count.value += len(image_keys)

        brillouin_regions = pm.get_brillouin_regions()
        rayleigh_regions = pm.get_rayleigh_regions()

        # Loop over all measurement positions
        for image_key in image_keys:
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
                for region_key, region in enumerate(rayleigh_regions):
                    w0, fwhm, intensity, offset = \
                        fit_lorentz_region(region, xdata, spectrum)

            if count is not None:
                count.value += 1

        return
