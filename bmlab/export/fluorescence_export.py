import os
import numpy as np
from skimage import transform
from PIL import Image

from bmlab import Session


class FluorescenceExport(object):

    def __init__(self):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = 'Fluorescence'
        return

    def export(self, configuration):
        if not self.file:
            return

        config = configuration['fluorescence']
        if not config['export']:
            return

        fluorescence_repetitions = self.file.repetition_keys(self.mode)
        brillouin_repetitions = self.file.repetition_keys()

        # Loop over all fluorescence repetitions
        for fluorescence_repetition in fluorescence_repetitions:
            # Get the repetition
            repetition = self.file.get_repetition(
                fluorescence_repetition, self.mode)
            # Get the keys for all images in this repetition
            image_keys = repetition.payload.image_keys()
            # Get the scale calibration
            scale_calibration = repetition.payload.get_scale_calibration()
            if scale_calibration is None:
                tmatrix = None
            else:
                # Create the transform matrix for the affine transformation
                n = np.linalg.norm(
                    np.array(scale_calibration['micrometerToPixX']))
                tmatrix = np.matrix([
                    [-1 * scale_calibration['micrometerToPixY'][1],
                     -1 * scale_calibration['micrometerToPixY'][0], 0],
                    [-1 * scale_calibration['micrometerToPixX'][1],
                     -1 * scale_calibration['micrometerToPixX'][0], 0],
                    [0, 0, n]
                ]) / n

            # Loop over all images in this repetition
            for image_key in image_keys:
                channel = repetition.payload.get_channel(image_key)
                img_data = repetition.payload.get_image(image_key)
                # Average all images acquired
                img_data = np.nanmean(img_data, axis=0).astype(np.ubyte)

                # Construct export path and create it if necessary
                if self.file.path.parent.name == 'RawData':
                    path = self.file.path.parents[1] / 'Plots'
                else:
                    path = self.file.path.parent
                if not os.path.exists(path):
                    os.mkdir(path)
                filename = path / f"{self.file.path.stem}" \
                                  f"_FLrep{fluorescence_repetition}" \
                                  f"_channel{channel}.png"
                image = Image.fromarray(img_data)
                if channel.casefold() == 'red':
                    blank = Image.new("L", image.size)
                    image = Image.merge("RGB", (image, blank, blank))

                if channel.casefold() == 'green':
                    blank = Image.new("L", image.size)
                    image = Image.merge("RGB", (blank, image, blank))

                if channel.casefold() == 'blue':
                    blank = Image.new("L", image.size)
                    image = Image.merge("RGB", (blank, blank, image))

                image.save(filename)

                # Also export the image with axes parallel to the stage
                if tmatrix is None:
                    continue

                # Get the region of interest of the repetition
                roi = repetition.payload.get_ROI(image_key)

                if roi is None:
                    continue

                # Create an x-y grid for the image with positions in µm
                x_pix, y_pix = np.meshgrid(
                    np.arange(roi['width_physical']) + roi['left'],
                    np.flip(np.arange(roi['height_physical']) + roi['bottom'])
                )
                x_pix = x_pix - scale_calibration['origin'][0]
                y_pix = y_pix - scale_calibration['origin'][1]

                x_mm =\
                    x_pix * scale_calibration['pixToMicrometerX'][0] +\
                    y_pix * scale_calibration['pixToMicrometerY'][0] +\
                    scale_calibration['positionStage'][0]
                y_mm =\
                    x_pix * scale_calibration['pixToMicrometerX'][1] +\
                    y_pix * scale_calibration['pixToMicrometerY'][1] +\
                    scale_calibration['positionStage'][1]

                # Create translation matrix to move image back to ROI
                corners = [[0, image.size[0]-1, 0, image.size[0]-1],
                           [0, 0, image.size[1]-1, image.size[1]-1],
                           [1, 1, 1, 1]]
                corners_warped = tmatrix * corners

                # Necessary translation
                dx = corners_warped[0, :].min()
                dy = corners_warped[1, :].min()
                # New shape
                sx = corners_warped[0, :].max()\
                    - corners_warped[0, :].min()
                sy = corners_warped[1, :].max()\
                    - corners_warped[1, :].min()

                translate = np.matrix([
                    [1, 0, -dx],
                    [0, 1, -dy],
                    [0, 0, 1]
                ])

                tform = transform.AffineTransform(
                    matrix=np.linalg.inv(translate * tmatrix))
                shape = (int(np.ceil(sy)), int(np.ceil(sx)))

                # Warp the images and positions to align with a
                # standard x-y coordinate system
                image_data_warped = transform.warp(
                    img_data,
                    tform, output_shape=shape, cval=np.nan)
                x_mm_warped = transform.warp(
                    x_mm,
                    tform, output_shape=shape, cval=np.nan)
                y_mm_warped = transform.warp(
                    y_mm,
                    tform, output_shape=shape, cval=np.nan)

                # Export image with proper alpha channel
                image_warped = Image.fromarray(
                    (255 * image_data_warped).astype(np.ubyte))
                if channel.casefold() == 'red':
                    blank = Image.new("L", image_warped.size)
                    image_warped =\
                        Image.merge("RGB", (image_warped, blank, blank))

                if channel.casefold() == 'green':
                    blank = Image.new("L", image_warped.size)
                    image_warped =\
                        Image.merge("RGB", (blank, image_warped, blank))

                if channel.casefold() == 'blue':
                    blank = Image.new("L", image_warped.size)
                    image_warped =\
                        Image.merge("RGB", (blank, blank, image_warped))
                image_alpha = Image.fromarray(
                    (255 * np.logical_not(
                        np.isnan(image_data_warped))).astype(np.ubyte))
                image_warped.putalpha(image_alpha)

                filename = path / f"{self.file.path.stem}" \
                                  f"_FLrep{fluorescence_repetition}" \
                                  f"_channel{channel}_aligned.png"
                image_warped.save(filename)

                # Export the images with the ROI of the Brillouin
                # measurements
                for brillouin_repetition in brillouin_repetitions:
                    # Get the repetition
                    repetition_bm = self.file.get_repetition(
                        brillouin_repetition)
                    # Read the Brillouin positions
                    positions = repetition_bm.payload.positions
                    x_min = np.nanmin(positions['x'])
                    x_max = np.nanmax(positions['x'])
                    y_min = np.nanmin(positions['y'])
                    y_max = np.nanmax(positions['y'])

                    # We are only interested in x-y-maps
                    if not x_min < x_max or not y_min < y_max:
                        continue

                    # Find the indices delimiting the Brillouin ROI
                    idx_mask = (x_mm_warped >= x_min) &\
                               (x_mm_warped <= x_max) &\
                               (y_mm_warped >= y_min) &\
                               (y_mm_warped <= y_max)
                    idx = np.nonzero(idx_mask)

                    # Crop the Fluorescence image to the Brillouin ROI
                    image_warped_bm = image_warped.crop(
                        (
                            np.min(idx[1]),
                            np.min(idx[0]),
                            np.max(idx[1]),
                            np.max(idx[0])
                        )
                    )

                    filename = path / f"{self.file.path.stem}" \
                                      f"_FLrep{fluorescence_repetition}" \
                                      f"_channel{channel}" \
                                      f"_BMrep{brillouin_repetition}.png"
                    image_warped_bm.save(filename)
