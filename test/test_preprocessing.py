import pytest
from preprocess import convert_to_sphinx
import os

def test_convert_to_sphinx():
    path = 'data/castor_2010_small_unpacked/11'
    try:
        os.mkdir('./tmp')
    except FileExistsError:
        os.rmdir('./tmp')
        os.mkdir('./tmp')
    out_path = convert_to_sphinx([path], output='./tmp')
    assert(all(['f.nii' in os.listdir(f) for f in os.listdir('./tmp')]) and len(os.listdir('./tmp')) > 0)
    print('tested sphinx')
