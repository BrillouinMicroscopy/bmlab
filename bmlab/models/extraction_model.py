import numpy as np
from scipy import interpolate

from bmlab.fits import fit_circle
from bmlab.geometry import Circle


class ExtractionModel(object):

    def __init__(self):
        self.arc_width = 2  # [pix] the width of the extraction arc
        self.points = {}
        self.calib_times = {}
        self.circle_fits = {}
        self.circle_fits_index = []
        self.circle_fits_interpolation = None
        self.extracted_values = {}
        self.extraction_angles = {}
        self.extraction_angles_index = None
        self.extraction_angles_interpolation = None

    def add_point(self, calib_key, time, xdata, ydata):
        if calib_key not in self.points:
            self.points[calib_key] = []
        self.points[calib_key].append((xdata, ydata))
        if len(self.points[calib_key]) >= 3:
            center, radius = fit_circle(self.points[calib_key])
            self.circle_fits[calib_key] = CircleFit(center, radius)
            self.calib_times[calib_key] = time
            self.refresh_circle_fits_interpolation()

    def get_points(self, calib_key):
        if calib_key in self.points:
            return self.points[calib_key]
        return []

    def get_time(self, calib_key):
        if calib_key in self.calib_times:
            return self.calib_times[calib_key]
        return []

    def optimize_points(self, calib_key, img, radius=10):

        from bmlab.image import find_max_in_radius
        # local import because to break circular dependency

        points = self.get_points(calib_key)
        time = self.get_time(calib_key)
        self.clear_points(calib_key)

        for p in points:
            new_point = find_max_in_radius(img, p, radius)
            # Warning: x-axis in imshow is 1-axis in img, y-axis is 0-axis
            self.add_point(
                calib_key, time, new_point[0], new_point[1])

    def clear_points(self, calib_key):
        self.points.pop(calib_key, None)
        self.circle_fits.pop(calib_key, None)
        self.calib_times.pop(calib_key, None)
        self.refresh_circle_fits_interpolation()

    def get_circle_fit(self, calib_key):
        circle_fit = self.circle_fits.get(calib_key)
        if circle_fit:
            return circle_fit.center, circle_fit.radius
        else:
            return None

    def get_circle_fit_by_time(self, time):
        if self.circle_fits_interpolation is None:
            return None
        fit = self.circle_fits_interpolation(self.circle_fits_index, time)
        return (fit[0], fit[1]), fit[2]

    def refresh_circle_fits_interpolation(self):
        # If there are no entries, we reset the interpolation
        if not self.calib_times:
            self.circle_fits_index = []
            self.circle_fits_interpolation = None
            return

        # Sort calibration keys by time
        sorted_keys = sorted(self.calib_times,
                             key=self.calib_times.get)
        # Create arrays to interpolate
        calib_times_array = []
        fits = []
        for key in sorted_keys:
            calib_times_array.append(self.calib_times[key])
            fit = []
            center, radius = self.get_circle_fit(key)
            fit.extend(center)
            fit.append(radius)
            fits.append(fit)
        circle_fits_array = np.array(fits)

        self.circle_fits_index = np.arange(circle_fits_array.shape[1])

        # If we only have one entry we always return the same value
        if len(sorted_keys) < 2:
            self.circle_fits_interpolation =\
                lambda idx, time: circle_fits_array[0]
        # Otherwise we can interpolate
        else:
            self.circle_fits_interpolation = interpolate.interp2d(
                self.circle_fits_index,
                calib_times_array,
                circle_fits_array)

    def get_arc_by_calib_key(self, calib_key):
        """
        Returns the arc at which to interpolate the 2D image for
        the 1D spectrum
        :param calib_key: the calibration number
        :return: the arc with pixel positions
        """
        arc = np.empty(0)
        try:
            center, radius = self.get_circle_fit(calib_key)
            circle = Circle(center, radius)
            phis = self.get_extraction_angles(calib_key)

            arc = self.get_arc_from_circle_phis(circle, phis)
        finally:
            return arc

    def get_arc_by_time(self, time):
        """
        Returns the arc at which to interpolate the 2D image for
        the 1D spectrum
        :param time: time point
        :return: the arc with pixel positions
        """
        arc = np.empty(0)
        try:
            center, radius = self.get_circle_fit_by_time(time)
            circle = Circle(center, radius)
            phis = self.get_extraction_angles_by_time(time)

            arc = self.get_arc_from_circle_phis(circle, phis)
        finally:
            return arc

    def get_arc_from_circle_phis(self, circle, phis):
        # ToDo refactor this, append to a list is slow
        arc = []
        for phi in phis:
            e_r = circle.e_r(phi)
            mid_point = circle.point(phi)
            points = [
                mid_point + e_r *
                k for k in np.arange(
                    -self.arc_width, self.arc_width + 1
                )
            ]
            arc.append(np.array(points))
        return np.array(arc)

    def set_extracted_values(self, calib_key, values):
        self.extracted_values[calib_key] = values

    def get_extracted_values(self, calib_key):
        values = self.extracted_values.get(calib_key)
        if values:
            return values
        return None

    def set_extraction_angles(self, calib_key, phis):
        self.extraction_angles[calib_key] = phis
        self.refresh_extraction_angles_interpolation()

    def get_extraction_angles(self, calib_key):
        if calib_key in self.extraction_angles:
            return self.extraction_angles[calib_key]
        return []

    def get_extraction_angles_by_time(self, time):
        return self.extraction_angles_interpolation(
            self.extraction_angles_index, time)

    def refresh_extraction_angles_interpolation(self):
        # If there are no entries, we reset the interpolation
        if not self.calib_times:
            self.extraction_angles_index = []
            self.extraction_angles_interpolation = None
            return

        # Sort calibration keys by time
        sorted_keys = sorted(self.calib_times,
                             key=self.calib_times.get)

        # Create arrays to interpolate
        calib_times_array = []
        angles = []
        for key in sorted_keys:
            calib_times_array.append(self.calib_times[key])
            angles.append(self.extraction_angles[key])
        extraction_angles_array = np.array(angles)

        self.extraction_angles_index = \
            np.arange(extraction_angles_array.shape[1])

        # If we only have one entry we always return the same value
        if len(sorted_keys) < 2:
            self.extraction_angles_interpolation =\
                lambda idx, time: extraction_angles_array[0]
        # Otherwise we can interpolate
        else:
            self.extraction_angles_interpolation = interpolate.interp2d(
                self.extraction_angles_index,
                calib_times_array,
                extraction_angles_array)

    def set_arc_width(self, width):
        self.arc_width = width


class CircleFit(object):

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius


class ExtractionException(Exception):
    pass
