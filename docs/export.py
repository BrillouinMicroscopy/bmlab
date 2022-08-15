import pathlib

from bmlab.session import Session
from bmlab.controllers import ExportController

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

ec = ExportController()
# Get the default configuration to adjust
export_config = ec.get_configuration()
# Disable exporting fluorescence data
export_config['fluorescence']['export'] = False
# Set caxis of Brillouin data
export_config['brillouin']['brillouin_shift_f']['cax'] = (5.0, 5.8)
# Export the data
ec.export(export_config)
