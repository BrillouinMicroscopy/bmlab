"""
Module for interacting with files containing Brillouin microscopy
data.

NOTE: Ideally, the users of bmlab should not have to know about
the file format in which the data are stored. So, if possible,
do not expose HDF objects to the outside (like BMicro).
"""

import datetime

import numpy as np
import h5py

BRILLOUIN_GROUP = 'Brillouin'


class BrillouinFile(object):

    def __init__(self, path):
        """
        Load a HDF file with Brillouin microscopy data.

        Parameters
        ----------
        path : str
            path of the file to load

        Raises
        ------
        OSError
            when trying to open non-existing or bad file
        """
        self.path = path
        self.file = None
        self.file = h5py.File(self.path, 'r')
        if BRILLOUIN_GROUP not in self.file:
            raise BadFileException('File does not contain any Brillouin data')
        self.comment = self.file.attrs.get('comment')[0].decode('utf-8')

    def __del__(self):
        """
        Destructor. Closes hdf file when object runs out of scope.
        """
        try:
            self.file.close()
        except Exception:
            pass

    def repetition_count(self):
        """
        Get the number of repetitions in the data file.

        Returns
        -------
        out : int
            Number of repetitions in the data file
        """
        return len(self.file.get(BRILLOUIN_GROUP, 0))

    def repetition_keys(self):
        """
        Returns list of keys for the various repetitions in the file.

        Returns
        -------
        out: list of str
        """
        return list(self.file[BRILLOUIN_GROUP].keys())

    def get_repetition(self, repetition_key):
        """
        Get a repetition from the data file based on given key.

        Parameters
        ----------
        repetition_key : str
            key to identify the repetition in the Brillouin group

        Returns
        -------
        out : Repetition
            the repetition
        """
        return Repetition(self.file[BRILLOUIN_GROUP].get(repetition_key))


class Repetition(object):

    def __init__(self, repetition_group):
        """
        Creates a repetition from the corresponding group of a HDF file.

        Parameters
        ----------
        repetition_group : HDF group
            The HDF group representing a Repetition. Consists of payload,
            calibration and background.
        """
        self.date = self._get_datetime(repetition_group.attrs.get('date')[0])
        self.payload = Payload(repetition_group.get('payload'))
        calibration_group = repetition_group.get('calibration')
        self.calibration = Calibration(calibration_group)

    def _get_datetime(self, time_stamp):
        """ Convert the time stamp in the HDF file to Python datetime """
        time_stamp = time_stamp.decode('utf-8')
        try:
            return datetime.datetime.fromisoformat(time_stamp)
        except Exception:
            return None


class Payload(object):

    def __init__(self, payload_group):
        """
        Creates a payload representation from the corresponding group of a
        HDF file.

        Parameters
        ----------
        payload_group : HDF group
            The payload of a repetition, basically a set of images

        """
        self.resolution = tuple(payload_group.attrs.get(
            'resolution-%s' % axis)[0] for axis in ['x', 'y', 'z'])
        self.data = payload_group.get('data')

    def image_keys(self):
        """
        Returns the keys of the images stored in the payload.

        Returns
        -------
        out: list of str
            Keys of images in payload.
        """
        if self.data:
            return list(self.data.keys())
        return []

    def get_image(self, image_key):
        """
        Returns the image from the payload for given key.

        Parameters
        ----------
        image_key: str
            Key for the image.

        Returns
        -------
        out: numpy.ndarray
            Array representing the image.
        """
        return np.array(self.data.get(image_key))


class Calibration(object):

    def __init__(self, calibration_group):
        """
        Creates a calibration representation from the corresponding group of
        a HDF file.

        Parameters
        ----------
        calibration_group : HDF group
            Calibration data of a repetition from an HDF file.
        """
        self.data = calibration_group.get('data')

    def is_empty(self):
        return self.data is None or len(self.data) == 0


class BadFileException(Exception):
    pass
