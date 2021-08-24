import pytest
from preprocess import convert_to_sphinx, motion_correction, plot_moco_rms_displacement, \
    linear_affine_registration, nonlinear_registration, fix_nii_headers, smooth
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


def test_linear_registration():
    path = 'data/castor_2010_small_moco/11'
    anatomical = 'data/castor_anatomical/castor.nii'
    try:
        os.mkdir('./tmp')
    except FileExistsError:
        shutil.rmtree('./tmp')
        os.mkdir('./tmp')
    res = linear_affine_registration([path], anatomical, output='tmp')
    assert(all(['moco_flirt.nii.gz' in os.listdir(os.path.join('./tmp', f)) for f in os.listdir('./tmp')])
           and len(os.listdir('./tmp')) > 0)
    print('tested linear affine registration')


def test_nonlinear_registration():
    path = 'data/castor_2010_small_flirt/11'
    anatomical = 'data/castor_anatomical/castor.nii'
    try:
        os.mkdir('./tmp')
    except FileExistsError:
        shutil.rmtree('./tmp')
        os.mkdir('./tmp')
    res = nonlinear_registration([path], anatomical, output='tmp')
    assert(all(['nirt.nii.gz' in os.listdir(os.path.join('./tmp', f)) for f in os.listdir('./tmp')])
           and len(os.listdir('./tmp')) > 0)
    print('tested nonlinear registration')


def test_fix_header():
    path = 'data/castor_2010_small_nirt/11'
    try:
        os.mkdir('./tmp')
    except FileExistsError:
        shutil.rmtree('./tmp')
        os.mkdir('./tmp')
    res = fix_nii_headers([path], output='tmp')
    assert(all(['fixed.nii' in os.listdir(os.path.join('./tmp', f)) for f in os.listdir('./tmp')])
           and len(os.listdir('./tmp')) > 0)


def test_smooth():
    path = 'data/castor_2010_small_fixed/11'
    try:
        os.mkdir('./tmp')
    except FileExistsError:
        shutil.rmtree('./tmp')
        os.mkdir('./tmp')
    res = smooth([path], output='tmp')
    assert(all(['smoothed.nii' in os.listdir(os.path.join('./tmp', f)) for f in os.listdir('./tmp')])
           and len(os.listdir('./tmp')) > 0)