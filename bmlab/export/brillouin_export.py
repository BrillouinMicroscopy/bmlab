import os
import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages

from bmlab import Session


class BrillouinExport(object):

    def __init__(self, evc):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = 'Brillouin'
        self.evc = evc
        return

    def export(self, configuration):
        if not self.file:
            return

        config = configuration['brillouin']
        if not config['export']:
            return

        brillouin_repetitions = self.file.repetition_keys()

        for brillouin_repetition in brillouin_repetitions:
            self.session.set_current_repetition(brillouin_repetition)

            # Get a list of all parameters available
            parameters = self.session.evaluation_model().get_parameter_keys()
            nr_brillouin_peaks =\
                self.session.evaluation_model().nr_brillouin_peaks

            for parameter_key in config['parameters']:

                if parameter_key not in parameters:
                    continue

                for brillouin_peak_index in\
                        range(nr_brillouin_peaks +
                              (nr_brillouin_peaks > 1) * 3):

                    peak_postfix = ''
                    if nr_brillouin_peaks > 1:
                        if brillouin_peak_index == 0:
                            peak_postfix = '_peak-single'
                        elif brillouin_peak_index == nr_brillouin_peaks + 1:
                            peak_postfix = '_peak-average'
                        elif brillouin_peak_index == nr_brillouin_peaks + 2:
                            peak_postfix = '_peak-average-weighted'
                        else:
                            peak_postfix = f"_peak-{brillouin_peak_index}"

                    # Get the data to plot
                    data, positions, dimensionality, labels =\
                        self.evc.get_data(parameter_key, brillouin_peak_index)

                    # This only works for 2D and 3D data!
                    if dimensionality < 2:
                        continue

                    # Subtract the mean value of the positions,
                    # so they are centered around zero
                    for position in positions:
                        position -= np.nanmean(position)

                    filename_base = f"{self.file.path.stem}" \
                                    f"_BMrep{brillouin_repetition}" \
                                    f"_{parameter_key}" \
                                    f"{peak_postfix}" \

                    # We slice the array along the smallest dimension
                    idx_slice = np.argmin(data.shape)
                    slice_count = data.shape[idx_slice]

                    tiff_images = []
                    x_res = 0
                    y_res = 0
                    for slice_number in range(slice_count):
                        slice_postfix = ''
                        if slice_count > 1:
                            slice_postfix = f'_slice-{slice_number}'
                        # Create data necessary to correctly slice the data
                        dslice = [slice(None) if (idx != idx_slice)
                                  else slice_number
                                  for idx, _ in enumerate(data.shape)]
                        idx = [idx for idx, _ in
                               enumerate(data.shape) if idx != idx_slice]

                        # We rotate the array, so the x-axis is shown as the
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

                        # Set the colormap limits to min/max
                        # or provided boundaries
                        # Minimum limit
                        try:
                            value_min = float(config[parameter_key]['cax'][0])
                        except BaseException:
                            value_min = np.nanmin(data)
                        try:
                            value_max = float(config[parameter_key]['cax'][1])
                        except BaseException:
                            value_max = np.nanmax(data)
                        ims.set_clim(value_min, value_max)

                        # Export plot
                        if self.file.path.parent.name == 'RawData':
                            path = self.file.path.parents[1]\
                                   / 'Plots' / 'WithAxis'
                        else:
                            path = self.file.path.parent
                        if not os.path.exists(path):
                            os.makedirs(path, exist_ok=True)

                        # Export as PDF
                        pdf_path = path / f"{filename_base}{slice_postfix}.pdf"
                        with PdfPages(pdf_path) as pdf:
                            pdf.savefig()

                        # Export as PNG
                        png_path = path / f"{filename_base}{slice_postfix}.png"
                        plt.savefig(png_path)

                        plt.close(fig)

                        # Export as bare image without axes
                        if self.file.path.parent.name == 'RawData':
                            path = self.file.path.parents[1] / 'Plots' / 'Bare'
                        else:
                            path = self.file.path.parent
                        if not os.path.exists(path):
                            os.makedirs(path, exist_ok=True)

                        # Convert indexed data into rgba map
                        rgba = cm.ScalarMappable(
                            norm=Normalize(vmin=value_min, vmax=value_max),
                            cmap=cm.viridis
                        ).to_rgba(image_map)

                        filename = path / f"{filename_base}{slice_postfix}.png"
                        image = Image.fromarray((255 * rgba).astype(np.ubyte))
                        image.save(filename)

                        x_res = (positions[idx[0]].shape[idx[0]] - 1) / abs(
                            np.nanmax(positions[idx[0]][tuple(dslice)]) -
                            np.nanmin(positions[idx[0]][tuple(dslice)]))
                        y_res = (positions[idx[1]].shape[idx[1]] - 1) / abs(
                            np.nanmax(positions[idx[1]][tuple(dslice)]) -
                            np.nanmin(positions[idx[1]][tuple(dslice)]))

                        # Create list of TIFF images
                        tiff_images.append(Image.fromarray(10000 * image_map))

                        # Export data as CSV file
                        if self.file.path.parent.name == 'RawData':
                            csv_path = self.file.path.parents[1] / 'Export'
                        else:
                            csv_path = self.file.path.parent
                        if not os.path.exists(csv_path):
                            os.makedirs(csv_path, exist_ok=True)
                        csv_filename =\
                            csv_path / f"{filename_base}{slice_postfix}.csv"
                        with open(csv_filename, 'w', newline='') as csvfile:
                            csv_writer = csv.writer(
                                csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                            label = f"{parameters[parameter_key]['label']}" \
                                    f" [{parameters[parameter_key]['unit']}]"
                            csv_writer.writerow([label])
                            csv_writer.writerow([])
                            csv_writer.writerows(data[tuple(dslice)])

                    if len(tiff_images) > 0:
                        # Export as bare image without axes
                        if self.file.path.parent.name == 'RawData':
                            path = self.file.path.parents[1] / 'Plots' / 'Bare'
                        else:
                            path = self.file.path.parent
                        # Export as TIFF files
                        filename = path / f"{filename_base}.tiff"
                        tiffinfo = dict({
                            270: "ImageJ=1.53f\nunit=micron",
                            282: x_res,  # XResolution [pixel/cm]
                            283: y_res,  # YResolution [pixel/cm]
                            296: 3  # Resolution unit [cm]
                        })
                        if len(tiff_images) > 2:
                            to_append = tiff_images[1:]
                        else:
                            to_append = []
                        tiff_images[0].save(
                            filename,
                            save_all=True,
                            tiffinfo=tiffinfo,
                            append_images=to_append
                        )
