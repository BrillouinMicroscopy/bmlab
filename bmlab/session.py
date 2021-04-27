import os

import numpy as np
import h5py

from bmlab.file import BrillouinFile
from bmlab.models.extraction_model import ExtractionModel
from bmlab.models.orientation import Orientation
from bmlab.models.calibration_model import CalibrationModel
from bmlab.models.peak_selection_model import PeakSelectionModel
from bmlab.models.evaluation_model import EvaluationModel
from bmlab.serializer import serialize, deserialize
from bmlab.image import extract_lines_along_arc
from bmlab.fits import fit_lorentz_region


class Session(object):
    """
    Session stores information about the current file
    to be processed.

    Session is a singleton. It can be accessed by
    calling the get_instance() method.
    """

    __instance = None

    def __init__(self):
        """
        Constructor of Session class. Since Session is a singleton,
        users should not call it but instead use the get_instance
        method.
        """
        if Session.__instance is not None:
            raise Exception('Session is a singleton!')
        else:
            Session.__instance = self
            self.clear()

    def current_repetition(self):
        """ Returns the repetition currently selected in data tab """
        if self.file and self._current_repetition_key:
            return self.file.get_repetition(self._current_repetition_key)
        return None

    def set_current_repetition(self, rep_key):
        self._current_repetition_key = rep_key

    def extraction_model(self):
        """
        Returns ExtractionModel instance for currently selected repetition
        """
        return self.extraction_models.get(self._current_repetition_key)

    def calibration_model(self):
        return self.calibration_models.get(self._current_repetition_key)

    def peak_selection_model(self):
        return self.peak_selection_models.get(self._current_repetition_key)

    def evaluation_model(self):
        return self.evaluation_models.get(self._current_repetition_key)

    @staticmethod
    def get_instance():
        """
        Returns the singleton instance of Session

        Returns
        -------
        out: Session
        """
        if Session.__instance is None:
            Session()
        return Session.__instance

    def set_file(self, file_name):
        """
        Set the file to be processed.

        Loads the corresponding data from HDF file.

        Parameters
        ----------
        file_name : str
            The file name.

        """
        try:
            file = BrillouinFile(file_name)
        except Exception as e:
            raise e

        """ Only load data if the file could be opened """
        self.file = file
        self.extraction_models = {key: ExtractionModel()
                                  for key in self.file.repetition_keys()}
        self.calibration_models = {key: CalibrationModel()
                                   for key in self.file.repetition_keys()}
        self.peak_selection_models = {key: PeakSelectionModel()
                                      for key in self.file.repetition_keys()}
        self.evaluation_models = {key: EvaluationModel()
                                  for key in self.file.repetition_keys()}

    def extract_calibration_spectrum(self, calib_key, frame_num=None):
        em = self.extraction_model()
        if not em:
            return
        arc = em.get_arc_by_calib_key(calib_key)
        if arc.size == 0:
            return

        imgs = self.current_repetition().calibration.get_image(calib_key)
        if frame_num is not None:
            imgs = imgs[frame_num:1]

        # Extract values from *all* frames in the current calibration
        extracted_values = []
        for img in imgs:
            values_by_img = extract_lines_along_arc(img,
                                                    self.orientation, arc)
            extracted_values.append(values_by_img)
        em.set_extracted_values(calib_key, extracted_values)
        return extracted_values

    def extract_payload_spectrum(self, image_key):
        em = self.extraction_model()
        if not em:
            return
        time = self.current_repetition().payload.get_time(image_key)
        arc = em.get_arc_by_time(time)
        if arc.size == 0:
            return

        imgs = self.current_repetition().payload.get_image(image_key)

        # Extract values from *all* frames in the current payload
        extracted_values = []
        for img in imgs:
            values_by_img = extract_lines_along_arc(img,
                                                    self.orientation, arc)
            extracted_values.append(values_by_img)

        exposure = self.current_repetition().payload.get_exposure(image_key)
        times = exposure * np.arange(len(imgs)) + time

        intensities = np.nanmean(imgs, axis=(1, 2))

        return extracted_values, times, intensities

    def fit_rayleigh_regions(self, calib_key):
        em = self.extraction_model()
        cm = self.calibration_model()
        extracted_values = em.get_extracted_values(calib_key)
        regions = cm.get_rayleigh_regions(calib_key)

        cm.clear_rayleigh_fits(calib_key)
        for frame_num, spectrum in enumerate(extracted_values):
            for region_key, region in enumerate(regions):
                spectrum = extracted_values[frame_num]
                xdata = np.arange(len(spectrum))
                w0, fwhm, intensity, offset = \
                    fit_lorentz_region(region, xdata, spectrum)
                cm.add_rayleigh_fit(calib_key, region_key, frame_num,
                                    w0, fwhm, intensity, offset)

    def fit_brillouin_regions(self, calib_key):
        em = self.extraction_model()
        cm = self.calibration_model()
        extracted_values = em.get_extracted_values(calib_key)
        regions = cm.get_brillouin_regions(calib_key)

        cm.clear_brillouin_fits(calib_key)
        for frame_num, spectrum in enumerate(extracted_values):
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

    def get_calib_keys(self):
        return self.current_repetition().calibration.image_keys()

    def get_image_keys(self):
        return self.current_repetition().payload.image_keys()

    def clear(self):
        """
        Close connection to loaded file.
        """

        # Global session data:
        self.file = None
        self.orientation = Orientation()
        self.setup = None

        # Session data by repetition:
        self.extraction_models = {}
        self.calibration_models = {}
        self.peak_selection_models = {}
        self.evaluation_models = {}

        self._current_repetition_key = None

    def set_setup(self, setup):
        self.setup = setup

    def save(self):
        if self.file is None:
            return

        h5_file_name = str(self.file.path)
        session_file_name = h5_file_name[:-3] + '.session.h5'

        with h5py.File(session_file_name, 'w') as f:
            serialize(self.orientation, f, 'orientation')
            serialize(self.extraction_models, f, 'extraction_models')
            serialize(self.calibration_models, f, 'calibration_models')
            # serialize(self.evaluation_models, f, 'evaluation_models')
            # serialize(self.peak_selection_models, f, 'peak_selection_models')

    def load(self, session_file_name):

        h5_file_name = str(session_file_name)[:-11] + '.h5'

        if not os.path.exists(session_file_name):
            return

        self.set_file(h5_file_name)

        with h5py.File(session_file_name, 'r') as f:
            self.orientation = deserialize(
                self.orientation.__class__, f['orientation'])
            self.extraction_models = deserialize(
                self.extraction_models.__class__, f['extraction_models'])
            self.calibration_models = deserialize(
                self.calibration_models.__class__, f['calibration_models']
            )
            # self.evaluation_models = deserialize(
            #     self.evaluation_models.__class__, f['evaluation_models']
            # )
            # self.peak_selection_models = deserialize(
            #     self.peak_selection_models.__class__,
            #     f['peak_selection_models']
            # )
