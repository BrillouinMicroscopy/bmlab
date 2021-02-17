import pytest
import pathlib


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
    assert rep.date


def test_file_get_resolution():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.resolution == (10, 1, 1)


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


def test_file_calibration_image_keys():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.calibration.image_keys() == [str(k+1) for k in range(2)]

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    rep = bf.get_repetition('0')
    assert rep.calibration.image_keys() == [str(k+1) for k in range(2)]
