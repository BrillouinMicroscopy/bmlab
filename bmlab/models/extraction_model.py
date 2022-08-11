import numpy as np
from scipy import interpolate

from bmlab.fits import fit_circle
from bmlab.geometry import Circle, discretize_arc
from bmlab.serializer import Serializer


class ExtractionModel(Serializer):

    def __init__(self):
        self.image_shape = None
        self.arc_width = 2  # [pix] the width of the extraction arc
        self.points = {}
        self.calib_times = {}
        self.positions = {}
        self.positions_interpolation = None

    def add_point(self, calib_key, time, xdata, ydata):
        if calib_key not in self.points:
            self.points[calib_key] = []
        self.points[calib_key].append((xdata, ydata))
        self.calib_times[calib_key] = time
        self.update_positions(calib_key)

    def set_point(self, calib_key, index, time, xdata, ydata):
        if calib_key not in self.points:
            self.points[calib_key] = []
        if index < len(self.points[calib_key]):
            self.points[calib_key][index] = (xdata, ydata)
        else:
            self.points[calib_key].append((xdata, ydata))
        self.calib_times[calib_key] = time
        self.update_positions(calib_key)

    def set_points(self, calib_key, time, points):
        self.calib_times[calib_key] = time
        self.points[calib_key] = points
        self.update_positions(calib_key)

    def get_points(self, calib_key):
        if calib_key in self.points:
            return self.points[calib_key]
        return []

    def get_time(self, calib_key):
        if calib_key in self.calib_times:
            return self.calib_times[calib_key]
        return []

    def clear_points(self, calib_key):
        self.points.pop(calib_key, None)
        self.calib_times.pop(calib_key, None)
        self.update_positions(calib_key)

    def post_deserialize(self):
        # Migrations from 0.1.10 to 0.2.0
        # Check that correct attributes are present
        # @since 0.2.0
        attributes_to_remove = [
            'circle_fits',
            'circle_fits_index',
            'circle_fits_interpolation',
            'extraction_angles',
            'extraction_angles_index',
            'extraction_angles_interpolation'
        ]
        for attribute in attributes_to_remove:
            if hasattr(self, attribute):
                delattr(self, attribute)
        if not hasattr(self, 'positions'):
            self.positions = {}
        if not hasattr(self, 'positions_interpolation'):
            self.positions_interpolation = None
        self.update_positions()

    def update_positions(self, key=None):
        if not hasattr(self, 'image_shape') or self.image_shape is None:
            return
        if not key:
            points = self.points.items()
        elif key in self.points:
            points = [(key, self.points[key])]
        else:
            points = []

        for calib_key, points in points:
            if len(points) >= 3:
                center, radius = fit_circle(points)
                # Check that we got a valid circle before continuing
                circle = Circle(center, radius)
                if not circle.valid:
                    continue

                phis = discretize_arc(circle, self.image_shape, num_points=500)
                if phis is None:
                    continue
                arc = self.get_arc_from_circle_phis(
                    circle, phis, self.arc_width)
                self.positions[calib_key] = arc
            # If we don't have enough points but positions
            # already present for this key, we have probably removed points
            # and clear the positions then
            elif calib_key in self.positions:
                self.positions.pop(calib_key, None)

        self.refresh_positions_interpolation()

    def refresh_positions_interpolation(self):
        # If there are no entries, we reset the interpolation
        if not self.calib_times:
            self.positions_interpolation = None
            return

        # Sort calibration keys by time
        sorted_keys = sorted(self.calib_times,
                             key=self.calib_times.get)

        # Create arrays to interpolate
        calib_times_array = []
        positions = []
        for calib_key in sorted_keys:
            # We might not have positions for all present keys yet
            if calib_key in self.positions:
                calib_times_array.append(self.calib_times[calib_key])
                positions.append(self.positions[calib_key])
        positions_array = np.array(positions)
        calib_times_array = np.array(calib_times_array)

        # If we only have one entry we cannot interpolate
        # and always return the same value
        if len(calib_times_array) < 1:
            self.positions_interpolation = None
        elif len(calib_times_array) == 1:
            self.positions_interpolation =\
                lambda time: positions_array[0]
        else:
            self.positions_interpolation =\
                interpolate.interp1d(
                    calib_times_array,
                    positions_array,
                    axis=0,
                    bounds_error=False,
                    fill_value=(positions_array[0], positions_array[-1])
                )

    def get_arc_by_calib_key(self, calib_key):
        """
        Returns the arc at which to interpolate the 2D image for
        the 1D spectrum

        Parameters
        ----------
        calib_key: the calibration number

        Returns
        -------
        The arc with pixel positions
        """
        arc = np.empty(0)
        try:
            if calib_key in self.positions:
                arc = self.positions.get(calib_key)
        finally:
            return arc

    def get_arc_by_time(self, time):
        """
        Returns the arc at which to interpolate the 2D image for
        the 1D spectrum

        Parameters
        ----------
        time: time point

        Returns
        -------
        The arc with pixel positions
        """
        arc = np.empty(0)
        try:
            arc = self.positions_interpolation(time)
        finally:
            return arc

    @staticmethod
    def get_arc_from_circle_phis(circle, phis, arc_width):
        pos_e_r = np.arange(-arc_width, arc_width + 1)[..., None]
        arc = np.ndarray((len(phis), len(pos_e_r), 2))
        for i, phi in enumerate(phis):
            mid_point, e_r = circle.point(phi)
            arc[i, :, :] = mid_point + e_r * pos_e_r
        return arc

    # TODO: This needs to be called automatically
    #  on file load or when the image orientation is changed
    def set_image_shape(self, shape):
        if not hasattr(self, 'image_shape') or self.image_shape != shape:
            self.image_shape = shape
            self.update_positions()

    def set_arc_width(self, width):
        self.arc_width = width
        self.update_positions()


class CircleFit(Serializer):

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius


class ExtractionException(Exception):
    pass
