import logging
import numpy as np

from bmlab.fits import fit_vipa, VIPA
from bmlab.session import Session

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

        repetition = self.session.current_repetition()
        if repetition is None:
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

        spectra = self.session\
            .extract_calibration_spectrum(calib_key)
        time = repetition.calibration.get_time(calib_key)

        if spectra is None:
            if max_count is not None:
                max_count.value = -1
            return

        if len(spectra) == 0:
            if max_count is not None:
                max_count.value = -1
            return

        self.session.fit_rayleigh_regions(calib_key)
        self.session.fit_brillouin_regions(calib_key)

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
