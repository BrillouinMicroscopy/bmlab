import pytest
import pathlib
import datetime

from bmlab.file import BrillouinFile, BadFileException


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_file_has_one_repetition():
    bf = BrillouinFile(data_file_path('Water.h5'))
    num_repetitions = bf.repetition_count()
    assert num_repetitions == 1

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    num_repetitions = bf.repetition_count()
    assert num_repetitions == 1


def test_non_existing_file_raises_exception():
    with pytest.raises(OSError):
        BrillouinFile('non_existing_file.h5')


def test_file_has_comment():
    bf = BrillouinFile(data_file_path('Water.h5'))
    assert bf.comment == 'Brillouin data'


def test_file_has_date():
    bf = BrillouinFile(data_file_path('Water.h5'))
    assert bf.date == datetime.datetime.fromisoformat(
        '2020-11-03T15:20:30.682+01:00')


def test_open_file_with_no_brillouin_data_raises_exception():
    with pytest.raises(BadFileException):
        BrillouinFile(data_file_path('empty_file.h5'))


def test_file_get_repetition_keys():
    bf = BrillouinFile(data_file_path('Water.h5'))
    assert bf.repetition_keys() == ['0']

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    assert bf.repetition_keys() == ['0']


def test_file_get_repetition_date():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.date == datetime.datetime.fromisoformat(
        '2020-11-03T15:20:52.852+01:00')


def test_file_get_resolution():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.resolution == (10, 1, 1)


def test_file_get_positions():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    # Test that the shape is correct
    assert rep.payload.positions['x'].shape == (1, 10, 1)
    assert rep.payload.positions['y'].shape == (1, 10, 1)
    assert rep.payload.positions['z'].shape == (1, 10, 1)
    # Test some values
    assert rep.payload.positions['x'][0, 0, 0] == -5652.5
    assert rep.payload.positions['y'][0, 0, 0] == -1963.0
    assert rep.payload.positions['z'][0, 0, 0] == 201.425


def test_file_repetition_has_calibration():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert not rep.calibration.is_empty()

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    rep = bf.get_repetition('0')
    assert not rep.calibration.is_empty()


def test_file_payload_image_keys():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.image_keys() == [str(k) for k in range(10)]

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.image_keys() == [str(k) for k in range(4)]


def test_file_payload_get_image():
    bf = BrillouinFile(data_file_path('Water.h5'))
    image = bf.get_repetition('0').payload.get_image('0')
    assert image.shape == (2, 400, 400)

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    image = bf.get_repetition('0').payload.get_image('0')
    assert image.shape == (2, 400, 400)


def test_file_payload_get_date():
    bf = BrillouinFile(data_file_path('Water.h5'))
    date = bf.get_repetition('0').payload.get_date('0')
    assert date == datetime.datetime.fromisoformat(
        '2020-11-03T15:21:10.568+01:00')


def test_file_payload_get_time():
    bf = BrillouinFile(data_file_path('Water.h5'))
    time = bf.get_repetition('0').payload.get_time('0')
    assert time == 39.886


def test_file_calibration_image_keys():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.calibration.image_keys() == [str(k+1) for k in range(2)]

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    rep = bf.get_repetition('0')
    assert rep.calibration.image_keys() == [str(k+1) for k in range(2)]


def test_file_calibration_get_date():
    bf = BrillouinFile(data_file_path('Water.h5'))
    date = bf.get_repetition('0').calibration.get_date('1')
    assert date == datetime.datetime.fromisoformat(
        '2020-11-03T15:21:07.484+01:00')


def test_file_calibration_get_time():
    bf = BrillouinFile(data_file_path('Water.h5'))
    time = bf.get_repetition('0').calibration.get_time('1')
    assert time == 36.802
