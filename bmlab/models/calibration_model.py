class CalibrationModel(object):

    def __init__(self):
        self.brillouin_regions = {}
        self.rayleigh_regions = {}
        self.brillouin_fits = {}
        self.rayleigh_fits = {}

    def add_brillouin_region(self, calib_key, region):
        if calib_key not in self.brillouin_regions:
            self.brillouin_regions[calib_key] = []

        self.brillouin_regions[calib_key].append(region)

    def set_brillouin_region(self, calib_key, index, region):
        if calib_key not in self.brillouin_regions:
            self.brillouin_regions[calib_key] = []

        self.brillouin_regions[calib_key][index] = region

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

    def clear_brillouin_fits(self, calib_key):
        self.brillouin_fits[calib_key] = []

    def clear_rayleigh_fits(self, calib_key):
        self.rayleigh_fits[calib_key] = []

    def clear_brillouin_regions(self, calib_key):
        self.brillouin_regions[calib_key] = []

    def clear_rayleigh_regions(self, calib_key):
        self.rayleigh_regions[calib_key] = []

    def add_rayleigh_region(self, calib_key, region):
        if calib_key not in self.rayleigh_regions:
            self.rayleigh_regions[calib_key] = []

        self.rayleigh_regions[calib_key].append(region)

    def set_rayleigh_region(self, calib_key, index, region):
        if calib_key not in self.rayleigh_regions:
            self.rayleigh_regions[calib_key] = []

        self.rayleigh_regions[calib_key][index] = region

    def get_brillouin_regions(self, calib_key):
        regions = self.brillouin_regions.get(calib_key)
        if regions is None:
            return []
        return regions

    def get_rayleigh_regions(self, calib_key):
        regions = self.rayleigh_regions.get(calib_key)
        if regions is None:
            return []
        return regions
