import numpy as np


class CalibrationModel(object):

    def __init__(self):
        self.brillouin_regions = {}
        self.rayleigh_regions = {}
        self.brillouin_fits = {}
        self.rayleigh_fits = {}

    def add_brillouin_region(self, calib_key, region):
        if calib_key not in self.brillouin_regions:
            self.brillouin_regions[calib_key] = []

        self.regions_merge_add_region(
            self.brillouin_regions[calib_key], region)

    def set_brillouin_region(self, calib_key, index, region):
        if calib_key not in self.brillouin_regions:
            self.brillouin_regions[calib_key] = []

        self.brillouin_regions[calib_key][index] = region

    def get_brillouin_regions(self, calib_key):
        regions = self.brillouin_regions.get(calib_key)
        if regions is None:
            return []
        return regions

    def clear_brillouin_regions(self, calib_key):
        self.brillouin_regions[calib_key] = []

    def add_brillouin_fit(self, calib_key, w0, gam, offset):
        if calib_key not in self.brillouin_fits:
            self.brillouin_fits[calib_key] = []
        self.brillouin_fits[calib_key].append({
            'w0': w0, 'gam': gam,
            'offset': offset})

    def get_brillouin_fits(self, calib_key):
        if calib_key in self.brillouin_fits:
            return self.brillouin_fits[calib_key]
        return {}

    def clear_brillouin_fits(self, calib_key):
        self.brillouin_fits[calib_key] = []

    def add_rayleigh_region(self, calib_key, region):
        if calib_key not in self.rayleigh_regions:
            self.rayleigh_regions[calib_key] = []

        self.regions_merge_add_region(
            self.rayleigh_regions[calib_key], region)

    def set_rayleigh_region(self, calib_key, index, region):
        if calib_key not in self.rayleigh_regions:
            self.rayleigh_regions[calib_key] = []

        self.rayleigh_regions[calib_key][index] = region

    def get_rayleigh_regions(self, calib_key):
        regions = self.rayleigh_regions.get(calib_key)
        if regions is None:
            return []
        return regions

    def clear_rayleigh_regions(self, calib_key):
        self.rayleigh_regions[calib_key] = []

    def add_rayleigh_fit(self, calib_key, w0, gam, offset):
        if calib_key not in self.rayleigh_fits:
            self.rayleigh_fits[calib_key] = []
        self.rayleigh_fits[calib_key].append({
            'w0': w0, 'gam': gam,
            'offset': offset})

    def get_rayleigh_fits(self, calib_key):
        if calib_key in self.rayleigh_fits:
            return self.rayleigh_fits[calib_key]
        return {}

    def clear_rayleigh_fits(self, calib_key):
        self.rayleigh_fits[calib_key] = []

    @staticmethod
    def regions_merge_add_region(regions, region):
        regions_fused = False

        # check if the selected regions overlap
        for i, saved_region in enumerate(regions):
            if (np.min(region) < np.max(saved_region)
                    and (np.max(region) > np.min(saved_region))):
                # fuse overlapping regions
                regions[i] = (
                    np.min([region, saved_region]),
                    np.max([region, saved_region]))
                regions_fused = True

        if not regions_fused:
            regions.append(region)
