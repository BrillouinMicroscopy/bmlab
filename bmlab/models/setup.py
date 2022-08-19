import numpy as np

from math import cos
from scipy import interpolate

import bmlab.constants as constants
from bmlab.serializer import Serializer


class Setup(Serializer):

    def __init__(self, key, name, pixel_size, focal_length,
                 vipa, calibration, temperature):
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
        # Default calibration temperature [°C]
        self.temperature = temperature

    def post_deserialize(self):
        # Migrations from 0.3.0 to 0.4.0
        if not hasattr(self, 'temperature'):
            self.temperature = 295.15

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

    def set_temperature(self, temperature):
        # Convert temperature from [°C] to [K]
        temperature = temperature + 273.15
        self.temperature = temperature
        # taken from
        # https://www.engineeringtoolbox.com/sound-speed-water-d_598.html
        # temperature water [K]
        water_t = [273.15, 278.15, 283.15, 293.15, 303.15, 313.15,
                   323.15, 333.15, 343.15, 353.15, 363.15, 373.15]
        # sound velocity water [m/s]
        water_vs = [1403, 1427, 1447, 1481, 1507, 1526,
                    1541, 1552, 1555, 1555, 1550, 1543]
        # Refractive index water [1]
        water_n = 1.3298
        water_f = interpolate.interp1d(water_t, water_vs)

        # taken from https://pubs.acs.org/doi/pdf/10.1021/je00054a002
        # temperature methanol [K]
        methanol_t = [274.74, 283.17, 293.15, 303.15, 313.11, 323.05, 332.95]
        # sound velocity methanol [m/s]
        methanol_vs = [1183.4, 1154.1, 1121.0, 1087.1, 1054.6, 1022.3, 990.3]
        # Refractive index methanol [1]
        methanol_n = 1.3234
        methanol_f = interpolate.interp1d(methanol_t, methanol_vs)

        water_shift = self.brillouin_shift(
            water_f(temperature), water_n)
        methanol_shift = self.brillouin_shift(
            methanol_f(temperature), methanol_n)

        self.calibration.shift_methanol = methanol_shift
        self.calibration.shift_water = water_shift
        self.calibration.update_calibration()

    def brillouin_shift(self, v, n):
        return 2 * cos(self.vipa.theta / 2) * n * v / self.vipa.lambda0


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
        self.shifts = np.array([])
        self.orders = np.array([])

        self.update_calibration()

    def set_shift_water(self, shift_water):
        self.shift_water = shift_water
        self.update_calibration()

    def set_shift_methanol(self, shift_methanol):
        self.shift_methanol = shift_methanol
        self.update_calibration()

    def update_calibration(self):
        # Construct array with the frequency shifts
        tmp = [self.shift_methanol, self.shift_water]
        self.shifts = np.full(2 + 2 * self.num_brillouin_samples, 0.0)
        for i in range(self.num_brillouin_samples):
            self.shifts[i + 1] = tmp[i]
            self.shifts[-1 * (i + 2)] = -1 * tmp[i]

        # The interference orders to which the peaks belong
        self.orders = np.full(2 + 2 * self.num_brillouin_samples, 0)
        self.orders[-(1 + self.num_brillouin_samples):] = 1


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
                                  shift_water=5.066e9),
          temperature=295.15),
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
                                  shift_methanol=3.78e9),
          temperature=295.15),
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
                                  shift_water=7.43e9),
          temperature=295.15)
]
