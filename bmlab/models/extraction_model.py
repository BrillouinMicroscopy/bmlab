from bmlab.fits import fit_circle
from bmlab.geometry import Circle
import numpy as np


class ExtractionModel(object):

    def __init__(self):
        self.arc_width = 3      # [pix] the width of the extraction arc
        self.points = {}
        self.circle_fits = {}
        self.extracted_values = {}
        self.extraction_angles = {}

    def add_point(self, calib_key, xdata, ydata):
        if calib_key not in self.points:
            self.points[calib_key] = []
        self.points[calib_key].append((xdata, ydata))
        if len(self.points[calib_key]) >= 3:
            self.circle_fits[calib_key] = fit_circle(self.points[calib_key])

    def get_points(self, calib_key):
        if calib_key in self.points:
            return self.points[calib_key]
        return []

    def optimize_points(self, calib_key, img, radius=10):

        from bmlab.image import find_max_in_radius
        # local import because to break circular dependency

        points = self.get_points(calib_key)
        self.clear_points(calib_key)

        for p in points:
            new_point = find_max_in_radius(img, p, radius)
            # Warning: x-axis in imshow is 1-axis in img, y-axis is 0-axis
            self.add_point(
                calib_key, new_point[0], new_point[1])

    def clear_points(self, calib_key):
        self.points[calib_key] = []
        self.circle_fits[calib_key] = None

    def get_circle_fit(self, calib_key):
        return self.circle_fits.get(calib_key)

    def get_arc_by_calib_key(self, calib_key):
        """
        Returns the arc at which to interpolate the 2D image for
        the 1D spectrum
        :param calib_key: the calibration number
        :return: the arc with pixel positions
        """
        arc = []
        try:
            center, radius = self.get_circle_fit(calib_key)
            circle = Circle(center, radius)
            phis = self.get_extraction_angles(calib_key)

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
        finally:
            return arc

    def set_extracted_values(self, calib_key, values):
        self.extracted_values[calib_key] = values

    def get_extracted_values(self, calib_key):
        values = self.extracted_values.get(calib_key)
        if values:
            return values
        return None, None

    def set_extraction_angles(self, calib_key, phis):
        self.extraction_angles[calib_key] = phis

    def get_extraction_angles(self, calib_key):
        if calib_key in self.extraction_angles:
            return self.extraction_angles[calib_key]
        return []

    def set_arc_width(self, width):
        self.arc_width = width
