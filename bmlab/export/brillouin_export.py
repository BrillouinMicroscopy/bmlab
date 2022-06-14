import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

from bmlab import Session


class BrillouinExport(object):

    def __init__(self, evc):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = 'Brillouin'
        self.evc = evc
        return

    def export(self):
        brillouin_repetitions = self.file.repetition_keys()

        for brillouin_repetition in brillouin_repetitions:
            self.session.set_current_repetition(brillouin_repetition)

            # Get a list of all parameters available
            parameters = self.session.evaluation_model().get_parameter_keys()

            # Get the data to plot
            # We select the Brillouin shift in GHz here
            parameter_key = 'brillouin_shift_f'
            data, positions, dimensionality, labels =\
                self.evc.get_data(parameter_key)

            # Subtract the mean value of the positions,
            # so they are centered around zero
            for position in positions:
                position -= np.nanmean(position)

            # This only works for two-dim data!
            if dimensionality != 2:
                continue

            # Create data necessary to correctly slice the data
            dslice = [slice(None) if dim > 1 else 0 for dim in data.shape]
            idx = [idx for idx, dim in enumerate(data.shape) if dim > 1]

            # We rotate the array, so the x axis is shown as the
            # horizontal axis
            image_map = data[tuple(dslice)]
            image_map = np.rot90(image_map)
            extent = np.nanmin(positions[idx[0]][tuple(dslice)]), \
                np.nanmax(positions[idx[0]][tuple(dslice)]), \
                np.nanmin(positions[idx[1]][tuple(dslice)]), \
                np.nanmax(positions[idx[1]][tuple(dslice)])

            # Actually plot the data
            fig = plt.figure()

            plot = fig.add_subplot(111)

            ims = plt.imshow(
                image_map, interpolation='nearest',
                extent=extent
            )
            plot.set_xlabel(labels[idx[0]])
            plot.set_ylabel(labels[idx[1]])
            cb_label = parameters[parameter_key]['symbol'] + \
                ' [' + parameters[parameter_key]['unit'] + ']'
            colorbar = fig.colorbar(ims)
            colorbar.ax.set_title(cb_label)
            plot.axis('scaled')
            plot.set_xlim(
                np.nanmin(positions[idx[0]][tuple(dslice)]),
                np.nanmax(positions[idx[0]][tuple(dslice)])
            )
            plot.set_ylim(
                np.nanmin(positions[idx[1]][tuple(dslice)]),
                np.nanmax(positions[idx[1]][tuple(dslice)])
            )
            value_min = np.nanmin(image_map)
            value_max = np.nanmax(image_map)
            ims.set_clim(value_min, value_max)

            # Export plot
            if self.file.path.parent.name == 'RawData':
                path = self.file.path.parents[1] / 'Plots' / 'WithAxis'
            else:
                path = self.file.path.parent
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            # Export as PDF
            plt.savefig(
                f"{path}\\{self.file.path.stem}"
                f"_BMrep{brillouin_repetition}"
                f"_{parameter_key}"
                ".pdf"
            )
            # Export as PNG
            plt.savefig(
                f"{path}\\{self.file.path.stem}"
                f"_BMrep{brillouin_repetition}"
                f"_{parameter_key}"
                ".png"
            )

            # Export as bare image without axes
            if self.file.path.parent.name == 'RawData':
                path = self.file.path.parents[1] / 'Plots' / 'Bare'
            else:
                path = self.file.path.parent
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

            # Convert indexed data into rgba map
            rgba = cm.ScalarMappable(cmap=cm.viridis).to_rgba(image_map)

            filename = f"{path}\\{self.file.path.stem}" \
                       f"_BMrep{brillouin_repetition}" \
                       f"_{parameter_key}" \
                       ".png"
            image = Image.fromarray((255 * rgba).astype(np.ubyte))
            image.save(filename)
