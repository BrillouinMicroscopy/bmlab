import numpy as np

import bmlab.constants as constants
from bmlab.serializer import Serializer


class Setup(Serializer):

    def __init__(self, key, name, pixel_size, focal_length,
                 vipa, calibration):
        """

        Parameters
        ----------
        key: str
            ID for the setup
        name: str
            Name of setup
        pixel_size: float
            pixel size of the camera [m]
        lambda0: float
            laser wavelength [m]
        focal_length: float
            focal length of the lens behind the VIPA [m]
        vipa
        calibration
        """
        self.key = key
        self.name = name
        self.pixel_size = pixel_size
        self.lambda0 = vipa.lambda0
        self.f0 = constants.c / self.lambda0
        self.focal_length = focal_length
        self.vipa = vipa
        self.calibration = calibration
        self.VIPA_PARAMS = self.init_vipa_params()

    def init_vipa_params(self):
        p1 = (2 * np.pi * self.vipa.n * self.vipa.d *
              np.cos(self.vipa.theta)) / constants.c
        p2 = -(2 * np.pi * self.vipa.n * self.vipa.d *
               np.tan(self.vipa.theta)) /\
              (constants.c * self.focal_length) * np.sqrt(
            1 - (self.vipa.n * np.sin(self.vipa.theta)) ** 2)
        p3 = -np.pi / constants.c * self.vipa.d * np.cos(self.vipa.theta) /\
            (self.focal_length ** 2)

        return p1, p2, p3


class VIPA(Serializer):

    def __init__(self, d, n, theta, order, lambda0):
        """ Start values for VIPA fit

        Parameters
        ----------
        d : float
            width of the cavity [m]
        n : float
            refractive index of the cavity [-]
        theta : float
            angle [rad]
        order: int
            observed order of the VIPA spectrum
        """
        self.d = d
        self.n = n
        self.theta = theta
        self.order = order
        self.lambda0 = lambda0
        self.FSR = constants.c / (2 * self.n * self.d * np.cos(self.theta))
        self.m = round(constants.c/(self.lambda0 * self.FSR))


class Calibration(Serializer):

    def __init__(self, num_brillouin_samples, shift_methanol=None,
                 shift_water=None):
        """

        Parameters
        ----------
        num_brillouin_samples: int
            Number of samples
        shift_methanol: float
            ??
        shift_water: float
            ??
        """

        self.num_brillouin_samples = num_brillouin_samples
        self.shift_methanol = shift_methanol
        self.shift_water = shift_water

        # Construct array with the frequency shifts
        tmp = [self.shift_methanol, self.shift_water]
        self.shifts = np.full(2 + 2 * self.num_brillouin_samples, 0.0)
        for i in range(self.num_brillouin_samples):
            self.shifts[i + 1] = tmp[i]
            self.shifts[-1 * (i + 2)] = -1 * tmp[i]

        # The interference orders to which the peaks belong
        self.orders = np.array([0, 0, 0, 1, 1, 1])


AVAILABLE_SETUPS = [
    Setup(key='S0',
          name='780 nm @ Biotec R340',
          pixel_size=6.5e-6,
          focal_length=0.2,
          vipa=VIPA(d=0.006743,
                    n=1.45367,
                    theta=0.8 * 2 * np.pi / 360,
                    order=0,
                    lambda0=780.24e-9),
          calibration=Calibration(num_brillouin_samples=2,
                                  shift_methanol=3.78e9,
                                  shift_water=5.066e9)),
    Setup(key='S1',
          name='780 nm @ Biotec R340 old',
          pixel_size=6.5e-6,
          focal_length=0.2,
          vipa=VIPA(d=0.006743,
                    n=1.45367,
                    theta=0.8 * 2 * np.pi / 360,
                    order=0,
                    lambda0=780.24e-9),
          calibration=Calibration(num_brillouin_samples=1,
                                  shift_methanol=3.78e9)),
    Setup(key='S2',
          name='532 nm @ Biotec R314',
          pixel_size=6.5e-6,
          focal_length=0.2,
          vipa=VIPA(d=0.003371,
                    n=1.46071,
                    theta=0.8 * 2 * np.pi / 360,
                    order=0,
                    lambda0=532e-9),
          calibration=Calibration(num_brillouin_samples=2,
                                  shift_methanol=5.54e9,
                                  shift_water=7.43e9))
]
