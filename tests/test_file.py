import pytest
import pathlib
import datetime

from bmlab.file import BrillouinFile,\
    BadFileException, is_source_file, is_session_file


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_is_source_file():
    # File from BrillouinAcquisition
    assert is_source_file(data_file_path('Water.h5'))
    # Session file from bmlab<0.0.14
    assert not is_source_file(data_file_path('1D-x.session.h5'))
    # Session file from bmlab>=0.0.14
    assert not is_source_file(data_file_path('1D-y.session.h5'))
    # Non-existing file
    assert not is_source_file(data_file_path('Unavailable.h5'))


def test_is_session_file():
    # Session file from bmlab<0.0.14
    assert is_session_file(data_file_path('1D-x.session.h5'))
    # Session file from bmlab>=0.0.14
    assert is_session_file(data_file_path('1D-y.session.h5'))
    # File from BrillouinAcquisition
    assert not is_session_file(data_file_path('Water.h5'))
    # Non-existing file
    assert not is_session_file(data_file_path('Unavailable.h5'))


def test_file_has_one_repetition():
    bf = BrillouinFile(data_file_path('Water.h5'))
    num_repetitions = bf.repetition_count()
    assert num_repetitions == 1

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    num_repetitions = bf.repetition_count()
    assert num_repetitions == 1


def test_non_existing_file_raises_exception():
    with pytest.raises(OSError):
        BrillouinFile(data_file_path('non_existing_file.h5'))


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

    assert bf.repetition_keys('Fluorescence') == []

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    assert bf.repetition_keys() == ['0']

    assert bf.repetition_keys('Fluorescence') == []


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


def test_file_calibration_get_exposure():
    bf = BrillouinFile(data_file_path('Binning.h5'))
    assert bf.get_repetition('0').calibration.get_exposure('1') == 0.8
    assert bf.get_repetition('0').payload.get_exposure('0') == 0.2


def test_file_calibration_get_binning():
    bf = BrillouinFile(data_file_path('Binning.h5'))
    assert bf.get_repetition('0').calibration.get_binning('1') == '8x8'
    assert bf.get_repetition('0').payload.get_binning('0') == '8x8'

    assert bf.get_repetition('0')\
        .payload.get_binning_factor('0') == 8


def test_file_repetition_count():
    bf = BrillouinFile(data_file_path('Fluorescence.h5'))
    mode = 'Fluorescence'

    rep_keys = bf.repetition_keys(mode)
    assert rep_keys == ['0', '1']

    with pytest.raises(NotImplementedError):
        bf.repetition_count('Raman')


def test_file_get_fluorescence_images():
    bf = BrillouinFile(data_file_path('Fluorescence.h5'))
    mode = 'Fluorescence'

    rep_keys = bf.repetition_keys(mode)
    assert rep_keys == ['0', '1']

    repetition = bf.get_repetition(rep_keys[0], mode)

    image_keys = repetition.payload.image_keys()
    assert image_keys == ['0', '1', '2', '3']

    time = repetition.payload.get_time(image_keys[0])
    assert time == 22.376

    channel = repetition.payload.get_channel(image_keys[0])
    assert channel == 'Blue'

    roi = repetition.payload.get_ROI(image_keys[0])
    assert roi == dict({
        'bottom': 152,
        'height_binned': 700,
        'height_physical': 700,
        'left': 200,
        'right': 202,
        'top': 150,
        'width_binned': 600,
        'width_physical': 600
    })

    image = repetition.payload.get_image(image_keys[0])
    assert image.shape == (1, 700, 600)


def test_file_get_scale_calibration():
    bf = BrillouinFile(data_file_path('Fluorescence.h5'))
    mode = 'Fluorescence'

    repetition = bf.get_repetition('0', mode)
    scale_calibration = repetition.payload.get_scale_calibration()

    assert scale_calibration == dict({
        'micrometerToPixX': (-7.95, 9.05),
        'micrometerToPixY': (9.45, 8.65),
        'pixToMicrometerX': (-0.05606325750210643, 0.05865577808023852),
        'pixToMicrometerY': (0.0612482986583706, 0.05152634649037527),
        'positionScanner': (0.0, 0.0),
        'positionStage': (0.0, 0.0),
        'origin': (0.0, 0.0)
    })
