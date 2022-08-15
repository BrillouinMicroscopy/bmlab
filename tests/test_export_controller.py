from pathlib import Path
import pytest
import shutil
import uuid
import os

from bmlab.session import Session
from bmlab.controllers import ExportController


@pytest.fixture()
def tmp_dir():
    current_path = Path.cwd()
    tmp_dir = Path.cwd() / f"tmp{str(uuid.uuid4())}" / 'RawData'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)
    yield tmp_dir
    os.chdir(current_path)
    try:
        shutil.rmtree(tmp_dir.parent)
    # Windows sometimes does not correctly close the HDF file. Then we have a
    # file access conflict.
    except Exception as e:
        print(e)


def data_file_path(file_name):
    return Path(__file__).parent / 'data' / file_name


def test_export_fluorescence(tmp_dir):
    shutil.copy(
        data_file_path('Fluorescence.h5'), Path.cwd() / 'Fluorescence.h5')

    session = Session.get_instance()
    session.set_file(Path('Fluorescence.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['export'] = False
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    images = [
        'Fluorescence_FLrep0_channelBlue.png',
        'Fluorescence_FLrep0_channelBlue_aligned.png',
        'Fluorescence_FLrep0_channelBrightfield.png',
        'Fluorescence_FLrep0_channelBrightfield_aligned.png',
        'Fluorescence_FLrep0_channelGreen.png',
        'Fluorescence_FLrep0_channelGreen_aligned.png',
        'Fluorescence_FLrep0_channelRed.png',
        'Fluorescence_FLrep0_channelRed_aligned.png',
        'Fluorescence_FLrep1_channelBlue.png',
        'Fluorescence_FLrep1_channelBlue_aligned.png',
        'Fluorescence_FLrep1_channelBrightfield.png',
        'Fluorescence_FLrep1_channelBrightfield_aligned.png',
        'Fluorescence_FLrep1_channelGreen.png',
        'Fluorescence_FLrep1_channelGreen_aligned.png',
        'Fluorescence_FLrep1_channelRed.png',
        'Fluorescence_FLrep1_channelRed_aligned.png',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)


def test_export_fluorescence_combined(tmp_dir):
    shutil.copy(
        data_file_path('Fluorescence.h5'), Path.cwd() / 'Fluorescence.h5')

    session = Session.get_instance()
    session.set_file(Path('Fluorescence.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['brillouin']['export'] = False
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots' / 'Bare'
    images = [
        'Fluorescence_FLrep0_fluorescenceCombined___b.png',
        'Fluorescence_FLrep0_fluorescenceCombined___b_aligned.png',
        'Fluorescence_FLrep0_fluorescenceCombined__g_.png',
        'Fluorescence_FLrep0_fluorescenceCombined__g__aligned.png',
        'Fluorescence_FLrep0_fluorescenceCombined__gb.png',
        'Fluorescence_FLrep0_fluorescenceCombined__gb_aligned.png',
        'Fluorescence_FLrep0_fluorescenceCombined_r__.png',
        'Fluorescence_FLrep0_fluorescenceCombined_r___aligned.png',
        'Fluorescence_FLrep0_fluorescenceCombined_r_b.png',
        'Fluorescence_FLrep0_fluorescenceCombined_r_b_aligned.png',
        'Fluorescence_FLrep0_fluorescenceCombined_rg_.png',
        'Fluorescence_FLrep0_fluorescenceCombined_rg__aligned.png',
        'Fluorescence_FLrep0_fluorescenceCombined_rgb.png',
        'Fluorescence_FLrep0_fluorescenceCombined_rgb_aligned.png',
        'Fluorescence_FLrep1_fluorescenceCombined___b.png',
        'Fluorescence_FLrep1_fluorescenceCombined___b_aligned.png',
        'Fluorescence_FLrep1_fluorescenceCombined__g_.png',
        'Fluorescence_FLrep1_fluorescenceCombined__g__aligned.png',
        'Fluorescence_FLrep1_fluorescenceCombined__gb.png',
        'Fluorescence_FLrep1_fluorescenceCombined__gb_aligned.png',
        'Fluorescence_FLrep1_fluorescenceCombined_r__.png',
        'Fluorescence_FLrep1_fluorescenceCombined_r___aligned.png',
        'Fluorescence_FLrep1_fluorescenceCombined_r_b.png',
        'Fluorescence_FLrep1_fluorescenceCombined_r_b_aligned.png',
        'Fluorescence_FLrep1_fluorescenceCombined_rg_.png',
        'Fluorescence_FLrep1_fluorescenceCombined_rg__aligned.png',
        'Fluorescence_FLrep1_fluorescenceCombined_rgb.png',
        'Fluorescence_FLrep1_fluorescenceCombined_rgb_aligned.png',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)


def test_export_brillouin(tmp_dir):
    shutil.copy(
        data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')

    session = Session.get_instance()
    session.set_file(Path('2D-xy.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['parameters'] =\
        ['brillouin_shift_f', 'brillouin_shift']
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots' / 'Bare'
    images = [
        '2D-xy_BMrep0_brillouin_shift_f.png',
        '2D-xy_BMrep0_brillouin_shift_f.tiff',
        '2D-xy_BMrep0_brillouin_shift.png',
        '2D-xy_BMrep0_brillouin_shift.tiff',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)

    plots_dir = tmp_dir.parent / 'Plots' / 'WithAxis'
    images = [
        '2D-xy_BMrep0_brillouin_shift_f.pdf',
        '2D-xy_BMrep0_brillouin_shift_f.png',
        '2D-xy_BMrep0_brillouin_shift.pdf',
        '2D-xy_BMrep0_brillouin_shift.png',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)

    csvs = [
        '2D-xy_BMrep0_brillouin_shift_f.csv',
        '2D-xy_BMrep0_brillouin_shift.csv',
    ]
    for csv in csvs:
        assert os.path.exists(
            tmp_dir.parent / 'Export' / csv)
