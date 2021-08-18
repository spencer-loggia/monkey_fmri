import pytest
from preprocess import convert_to_sphinx, motion_correction, plot_moco_rms_displacement
import os
import shutil


def test_convert_to_sphinx():
    path = 'data/castor_2010_small_unpacked/11'
    try:
        os.mkdir('./tmp')
    except FileExistsError:
        shutil.rmtree('./tmp')
        os.mkdir('./tmp')
    _ = convert_to_sphinx([path], output='./tmp')
    assert(all(['f.nii' in os.listdir(os.path.join('./tmp', f)) for f in os.listdir('./tmp')])
           and len(os.listdir('./tmp')) > 0)
    print('tested sphinx')


def test_motion_correction():
    path = 'data/castor_2010_small_sphinx/11'
    try:
        os.mkdir('./tmp')
    except FileExistsError:
        shutil.rmtree('./tmp')
        os.mkdir('./tmp')
    res = motion_correction([path], output='./tmp', check_rms=False)
    assert(all(['moco.nii.gz' in os.listdir(os.path.join('./tmp', f)) for f in os.listdir('./tmp')])
           and len(os.listdir('./tmp')) > 0)
    print('tested motion correction')


def test_plot_moco():
    path = 'data/castor_2010_small_moco/11'
    try:
        os.mkdir('./tmp')
    except FileExistsError:
        shutil.rmtree('./tmp')
        os.mkdir('./tmp')
    good_moco = plot_moco_rms_displacement([path], 'tmp/')
    assert (all(['moco_rms_displacement.eps' in os.listdir('./tmp')])
            and len(os.listdir('./tmp')) > 0)
    assert all(good_moco[k] for k in good_moco.keys())





