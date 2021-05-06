import os

import numpy as np
import h5py

from bmlab.file import BrillouinFile
from bmlab.models.extraction_model import ExtractionModel
from bmlab.models.orientation import Orientation
from bmlab.models.calibration_model import CalibrationModel
from bmlab.models.peak_selection_model import PeakSelectionModel
from bmlab.models.evaluation_model import EvaluationModel
from bmlab.serializer import Serializer
from bmlab.image import extract_lines_along_arc


class Session(Serializer):
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
        else:
            """ Only load data if the file could be opened """
            session = Session.get_instance()
            session.file = file
            session.extraction_models = {
                key: ExtractionModel()
                for key in self.file.repetition_keys()
            }
            session.calibration_models = {
                key: CalibrationModel()
                for key in self.file.repetition_keys()
            }
            session.peak_selection_models = {
                key: PeakSelectionModel()
                for key in self.file.repetition_keys()
            }
            session.evaluation_models = {
                key: EvaluationModel()
                for key in self.file.repetition_keys()
            }

        try:
            self.load(file_name)
        except Exception as e:
            raise e
        else:
            Session.get_instance().file = file

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

    def get_calib_keys(self):
        return self.current_repetition().calibration.image_keys()

    def get_image_keys(self):
        return self.current_repetition().payload.image_keys()

    def clear(self):
        """
        Close connection to loaded file.
        """

        # Global session data:
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()
            self.file = None
        else:
            self.file = None

        self.orientation = Orientation()
        self.setup = None

        # Session data by repetition:
        self.extraction_models = {}
        self.calibration_models = {}
        self.evaluation_models = {}
        self.peak_selection_models = {}

        self._current_repetition_key = None

    def set_setup(self, setup):
        self.setup = setup

    def save(self):
        if self.file is None:
            return

        session_file_name = self.get_session_file_name(self.file.path)

        with h5py.File(session_file_name, 'w') as f:
            self.serialize(f, 'session', skip=['file'])

    def load(self, h5_file_name):

        session_file_name = self.get_session_file_name(h5_file_name)

        if not os.path.exists(session_file_name):
            return

        with h5py.File(session_file_name, 'r') as f:
            new_session = Serializer.deserialize(f['session'])
            session = Session.get_instance()
            for var_name, var_value in new_session.__dict__.items():
                session.__dict__[var_name] = var_value

    @staticmethod
    def get_session_file_name(h5_file_name):
        return str(h5_file_name)[:-3] + '.session.h5'
