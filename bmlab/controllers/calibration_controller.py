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

    def calibrate(self, calib_key):

        if not calib_key:
            return

        if not self.setup:
            return

        em = self.session.extraction_model()
        if not em:
            return

        cm = self.session.calibration_model()
        if not cm:
            return

        spectra = self.session.extract_calibration_spectrum(calib_key)
        time = self.session.current_repetition().calibration.get_time(calib_key)

        if spectra is None:
            return

        if len(spectra) == 0:
            return

        self.session.fit_rayleigh_regions(calib_key)
        self.session.fit_brillouin_regions(calib_key)

        vipa_params = []
        frequencies = []
        for frame_num, spectrum in enumerate(spectra):
            peaks = cm.get_sorted_peaks(calib_key, frame_num)

            params = fit_vipa(peaks, self.setup)
            if params is None:
                continue
            vipa_params.append(params)
            xdata = np.arange(len(spectrum))

            frequencies.append(VIPA(xdata, params) - self.setup.f0)

        cm.set_vipa_params(calib_key, vipa_params)
        cm.set_frequencies(calib_key, time, frequencies)
