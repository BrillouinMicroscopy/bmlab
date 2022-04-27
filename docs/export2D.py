import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os

from bmlab.session import Session
from bmlab.controllers import EvaluationController

"""
Place this file in the same directory as your RawData/EvalData folders
and adjust the filename.
"""
filename = 'Brillouin.h5'


# Construct path to file
file_path = pathlib.Path(__file__).parent / 'RawData' / filename

session = Session.get_instance()

# Load the file to be used
session.set_file(file_path)
# Export the first repetition
session.set_current_repetition('0')

evc = EvaluationController()

# Get a list of all parameters available
parameters = session.evaluation_model().get_parameter_keys()

# Get the data to plot
# We select the Brillouin shift in GHz here
parameter_key = list(parameters)[0]
data, positions, dimensionality, labels =\
                    evc.get_data(parameter_key)

# Subtract the mean value of the positions,
# so they are centered around zero
for position in positions:
    position -= np.nanmean(position)

# This only works for two-dim data!
if dimensionality != 2:
    exit()

# Create data necessary to correctly slice the data
dslice = [slice(None) if dim > 1 else 0 for dim in data.shape]
idx = [idx for idx, dim in enumerate(data.shape) if dim > 1]

# We rotate the array so the x axis is shown as the
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

# Export plot
plot_dir = pathlib.Path(__file__).parent / 'Plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# Export as PDF
plt.savefig(
    plot_dir /
    (file_path.stem + '_' + parameter_key + '.pdf')
)
# Export as PNG
plt.savefig(
    plot_dir /
    (file_path.stem + '_' + parameter_key + '.png')
)

# Show the plot
plt.show()
