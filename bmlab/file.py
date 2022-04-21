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

from packaging import version

BRILLOUIN_GROUP = 'Brillouin'


def _get_datetime(time_stamp):
    """ Convert the time stamp in the HDF file to Python datetime """
    try:
        return datetime.datetime.fromisoformat(time_stamp)
    except Exception:
        return None


def is_source_file(path):
    try:
        with h5py.File(path, "r") as h5:
            file_format = h5.attrs.get('version')
            if file_format is not None:
                # Is this a file from BrillouinAcquisition?
                # Then we get an ndarray with a bytes object inside
                # (because the version attribute is stored
                # as 'cset=H5T_CSET_ASCII').
                if isinstance(file_format, np.ndarray)\
                        and isinstance(file_format[0], bytes):
                    file_format = file_format[0].decode('ascii')
                # This is from BrillouinAcquisition!
                if file_format.startswith('H5BM'):
                    return True
        return False
    except Exception:
        return False


def is_session_file(path):
    try:
        with h5py.File(path, "r") as h5:
            file_format = h5.attrs.get('version')
            if file_format is not None:
                # This is a bmlab session file created
                # with bmlab>=0.0.14
                if isinstance(file_format, str)\
                        and file_format.startswith('bmlab'):
                    return True

            # This is a bmlab session file created with bmlab<0.0.14
            if 'session' in h5.keys()\
                    and h5['session'].attrs.get('type').startswith('bmlab'):
                return True
        return False
    except Exception:
        return False


class BrillouinFile(object):

    def __init__(self, path):
        """
        Load a HDF file with Brillouin microscopy data.

        Parameters
        ----------
        path : Path
            path of the file to load

        Raises
        ------
        OSError
            when trying to open non-existing or bad file
        """
        self.path = path
        self.file = None
        self.file = h5py.File(self.path, 'r')
        self.file_version_string = self.file.attrs.get('version')[
            0].decode('utf-8')
        if not self.file_version_string.startswith('H5BM'):
            raise BadFileException('File does not contain any Brillouin data')
        self.file_version = self.file_version_string[-5:]

        if version.parse(self.file_version) >= version.parse("0.0.4"):
            """"
            New Brillouin file format,
            supporting different modes and repetitions
            """
            if BRILLOUIN_GROUP not in self.file:
                raise BadFileException(
                    'File does not contain any Brillouin data')
            self.Brillouin_group = self.file[BRILLOUIN_GROUP]
        else:
            """ Old Brillouin file format """
            self.Brillouin_group = self.file
        self.comment = self.file.attrs.get('comment')[0].decode('utf-8')
        self.date = _get_datetime(
            self.file.attrs.get('date')[0].decode('utf-8'))

    def __del__(self):
        """
        Destructor. Closes hdf file when object runs out of scope.
        """
        self.close()

    def close(self):
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
        return len(self.repetition_keys())

    def repetition_keys(self):
        """
        Returns list of keys for the various repetitions in the file.

        Returns
        -------
        out: list of str
        """
        if version.parse(self.file_version) >= version.parse("0.0.4"):
            return list(self.Brillouin_group.keys())
        else:
            return list(['0'])

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
        if version.parse(self.file_version) >= version.parse("0.0.4"):
            return Repetition(self.Brillouin_group.get(repetition_key), self)
        else:
            return Repetition(self.file, self)


class Repetition(object):

    def __init__(self, repetition_group, file):
        """
        Creates a repetition from the corresponding group of a HDF file.

        Parameters
        ----------
        repetition_group : HDF group
            The HDF group representing a Repetition. Consists of payload,
            calibration and background.
        """
        self.date = _get_datetime(
            repetition_group.attrs.get('date')[0].decode('utf-8'))
        self.payload = Payload(repetition_group.get('payload'), self)
        calibration_group = repetition_group.get('calibration')
        self.calibration = Calibration(calibration_group, self)
        self.file = file


class Payload(object):

    def __init__(self, payload_group, repetition):
        """
        Creates a payload representation from the corresponding group of a
        HDF file.

        Parameters
        ----------
        payload_group : HDF group
            The payload of a repetition, basically a set of images

        """
        self.repetition = repetition
        self.resolution = tuple(int(payload_group.attrs.get(
            'resolution-%s' % axis)[0]) for axis in ['x', 'y', 'z'])
        self.positions = {
            'x': np.array(payload_group.get('positions-x')),
            'y': np.array(payload_group.get('positions-y')),
            'z': np.array(payload_group.get('positions-z')),
        }
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
        imgs = self.data.get(image_key)
        if imgs is None:
            return None
        return np.array(imgs)

    def get_image_count(self, image_key):
        imgs = self.data.get(image_key)
        if imgs is None:
            return 0
        return imgs.shape[0]

    def get_date(self, image_key):
        """"
        Returns the date of a payload image
        with the given key
        """
        try:
            return _get_datetime(
                self.data.get(image_key).attrs.get('date')[0].decode('utf-8'))
        except Exception:
            return ''

    def get_time(self, image_key):
        try:
            # Get date of the calibration
            date = self.get_date(image_key)
            # Get the reference date
            ref = self.repetition.file.date
            # return the difference in seconds
            return (date - ref).total_seconds()
        except Exception:
            return None

    def get_exposure(self, image_key):
        """"
        Returns the exposure time of a payload image
        with the given key
        """
        try:
            return self.data.get(image_key).attrs\
                .get('exposure')[0].decode('utf-8')
        except Exception:
            # For older files we return a default value
            return 0.5


class Calibration(object):

    def __init__(self, calibration_group, repetition):
        """
        Creates a calibration representation from the corresponding group of
        a HDF file.

        Parameters
        ----------
        calibration_group : HDF group
            Calibration data of a repetition from an HDF file.
        """
        self.repetition = repetition
        self.data = calibration_group.get('data')
        """
        For H5BM files < 0.0.4
        there was an inconsistency with the group naming
        """
        if self.data is None:
            self.data = calibration_group.get('calibrationData')

    def is_empty(self):
        return self.data is None or len(self.data) == 0

    def image_keys(self, sort_by_time=False):
        if self.data:
            keys = list(self.data.keys())
            if not sort_by_time:
                return keys

            dates = []
            for key in keys:
                dates.append(self.get_date(key))

            return [i for _, i in sorted(zip(dates, keys))]
        return []

    def get_image(self, image_key):
        """
        Returns the image from the calibration for given key.

        Parameters
        ----------
        image_key: str
            Key for the image.

        Returns
        -------
        out: numpy.ndarray
            Array representing the image.
        """
        imgs = self.data.get(image_key)
        if imgs is None:
            return None
        return np.array(imgs)

    def get_image_count(self, image_key):
        imgs = self.data.get(image_key)
        if imgs is None:
            return 0
        return imgs.shape[0]

    def get_date(self, image_key):
        """"
        Returns the date of a calibration image
        with the given key
        """
        try:
            return _get_datetime(
                self.data.get(image_key).attrs.get('date')[0].decode('utf-8'))
        except Exception:
            return ''

    def get_time(self, image_key):
        try:
            # Get date of the calibration
            date = self.get_date(image_key)
            # Get the reference date
            ref = self.repetition.file.date
            # return the difference in seconds
            return (date - ref).total_seconds()
        except Exception:
            return None


class BadFileException(Exception):
    pass
