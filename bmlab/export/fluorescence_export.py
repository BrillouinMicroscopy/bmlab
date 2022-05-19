from PIL import Image
import os

from bmlab import Session


class FluorescenceExport(object):

    def __init__(self):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = 'Fluorescence'
        return

    def export(self):
        fluorescence_repetitions = self.file.repetition_keys(self.mode)

        # Loop over all fluorescence repetitions
        for fluorescence_repetition in fluorescence_repetitions:
            # Get the repetition
            repetition = self.file.get_repetition(
                fluorescence_repetition, self.mode)
            # Get the keys for all images in this repetition
            image_keys = repetition.payload.image_keys()
            # Loop over all images in this repetition
            for image_key in image_keys:
                channel = repetition.payload.get_channel(image_key)
                img_data = repetition.payload.get_image(image_key)

                for acq in range(img_data.shape[0]):
                    # Construct export path and create it if necessary
                    if self.file.path.parent.name == 'RawData':
                        path = self.file.path.parents[1] / 'Plots'
                    else:
                        path = self.file.path.parent
                    if not os.path.exists(path):
                        os.mkdir(path)
                    postfix = ''
                    if img_data.shape[0] > 1:
                        postfix = f'_{acq}'
                    filename = f"{path}\\{self.file.path.stem}" \
                               f"_FLrep{fluorescence_repetition}" \
                               f"_channel{channel}{postfix}.png"
                    image = Image.fromarray(img_data[acq, :, :])
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
