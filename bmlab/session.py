import os

from pathlib import Path

import h5py

from bmlab.file import BrillouinFile
from bmlab.models.extraction_model import ExtractionModel
from bmlab.models.orientation import Orientation
from bmlab.models.calibration_model import CalibrationModel
from bmlab.models.peak_selection_model import PeakSelectionModel
from bmlab.models.evaluation_model import EvaluationModel
from bmlab.serializer import Serializer


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
        """
        Returns the repetition currently selected in data tab
        """
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

    def set_image_shape(self):
        """
        We set the extraction model image shape for every repetition when
        - a file gets loaded
        - the image orientation changes
        """
        repetitions = self.file.repetition_keys()
        for repetition in repetitions:
            imgs = self.file.get_repetition(repetition).payload.get_image('0')
            img = self.orientation.apply(imgs[0, ...])
            self.extraction_models.get(repetition).set_image_shape(img.shape)

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
            self.set_image_shape()

        try:
            self.load(file_name)
        except Exception as e:
            raise e
        else:
            Session.get_instance().file = file

    def get_calib_keys(self):
        return self.current_repetition().calibration.image_keys()

    def get_calibration_image(self, calib_key, frame_num=None):
        imgs = self.current_repetition().calibration.get_image(calib_key)
        if frame_num is not None:
            imgs = imgs[frame_num, ...]
        return self.orientation.apply(imgs)

    def get_calibration_time(self, calib_key):
        return self.current_repetition().calibration.get_time(calib_key)

    def get_image_keys(self):
        return self.current_repetition().payload.image_keys()

    def get_payload_image(self, image_key, frame_num=None):
        imgs = self.current_repetition().payload.get_image(image_key)
        if frame_num is not None:
            imgs = imgs[frame_num, ...]
        return self.orientation.apply(imgs)

    def get_payload_time(self, image_key):
        return self.current_repetition().payload.get_time(image_key)

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

    def set_rotation(self, num_rots):
        self.orientation.set_rotation(num_rots)
        self.set_image_shape()

    def set_reflection(self, **kwargs):
        self.orientation.set_reflection(**kwargs)
        self.set_image_shape()

    def save(self):
        if self.file is None:
            return

        session_file_name = self.get_session_file_name(self.file.path,
                                                       create_folder=True)

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
    def get_session_file_name(h5_file_name, create_folder=False):
        # If the raw data file is located in a 'RawData' folder,
        # we put the eval data file in an 'EvalData' folder and
        # don't append the 'session' string.
        file = Path(h5_file_name)
        if file.parent.name == 'RawData':
            eval_folder = file.parents[1] / 'EvalData'
            # Create the evaluation folder if necessary
            if create_folder and not os.path.exists(eval_folder):
                os.mkdir(eval_folder)
            return str(eval_folder / (str(file.name)[:-3] + '.h5'))
        else:
            return str(h5_file_name)[:-3] + '.session.h5'
