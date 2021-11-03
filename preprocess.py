from __future__ import print_function
from __future__ import division

import ntpath
import shutil
import sys
from typing import List, Union, Tuple, Dict

import numpy as np
from nipype.interfaces import freesurfer
from builtins import str
from builtins import range

import os  # system functions
import subprocess

from scipy.ndimage import zoom, gaussian_filter

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as model  # model generation
import nipype.algorithms.rapidart as ra  # artifact detection
import nibabel as nib
from nilearn import image as nimg

from matplotlib import pyplot as plt
from multiprocessing import Pool

'''
TODO:
Slice timing correction
'''
def _create_dir_if_needed(base: str, name: str):
    out_dir = os.path.join(base, name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return out_dir


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    flat = arr.flatten()
    mean = np.mean(flat)
    std = np.std(flat)
    new_arr = (arr - mean) / std
    return new_arr


def _drift_correction(arr: np.ndarray) -> np.ndarray:
    """
    model temporal drift and correct for it. Assumes stimuli are evenly distributed throughout the run and will
    produce potentially wrong results otherwise.
    :param arr:
    :return:
    """
    mean = np.mean(np.mean(np.mean(arr, axis=2), axis=1), axis=0)
    filtered = gaussian_filter(mean, sigma=2)
    axis = np.arange(filtered.shape[0], dtype=float)
    cov = np.cov(filtered, axis)
    eig_val, eig_vec = np.linalg.eig(cov)
    ind = np.argsort(eig_val)
    vec = eig_vec[ind[-1]] * eig_val[ind[-1]]
    m = vec[0] / vec[1]
    est = m * axis
    arr = arr + np.tile(est, (arr.shape[0], arr.shape[1], arr.shape[2], 1))
    print('drift coefficient', m)
    return arr


def _apply_binary_mask(ts_in: nib.Nifti1Image, mask: nib.Nifti1Image) -> nib.Nifti1Image:
    n_data = np.array(ts_in.get_fdata())
    mask_arr = np.array(mask.get_fdata())[:, :, :, None]
    tiled_mask = np.tile(mask_arr, (1, 1, 1, n_data.shape[3]))
    n_data[tiled_mask == 0] = 0
    new_nifti = nib.Nifti1Image(n_data, affine=ts_in.affine, header=ts_in.header)
    return new_nifti


def _extract_frame(nii: nib.Nifti1Image, loc: Union[None, int] = None, norm: bool = True):
    """
    extracts a 3D reference frame from 4D nifti object and generates a 3D nifti object
    :param nii:
    :param loc:
    :return:
    """
    n_data = np.array(nii.get_fdata())
    if not loc:
        loc = int(n_data.shape[2] / 2)
    n_data = n_data[:, :, :, loc]
    if norm:
        n_data = _normalize_array(n_data)
    new_nifti = nib.Nifti1Image(n_data, affine=nii.affine, header=nii.header)
    return new_nifti


def _mri_convert_sphinx_wrapper(in_file, out_file, cvt):
    cvt.inputs.in_file = in_file
    cvt.inputs.out_file = out_file
    cvt.inputs.args = "--sphinx"
    cvt.inputs.out_type = 'nii'
    cvt.cmdline
    return cvt.run()


def _mri_convert_wrapper(in_file, out_file, in_orientation, out_orientation, cvt):
    cvt.inputs.in_file = in_file
    cvt.inputs.out_file = out_file
    cvt.inputs.args = "--in_orientation {} --out_orientation {}".format(in_orientation, out_orientation)
    cvt.inputs.out_type = 'nii'
    cvt.cmdline
    return cvt.run()


def _slice_time_wrapper(in_file, out_file, slice_dir, TR, st):
    st.inputs.in_file = in_file
    st.inputs.out_file = out_file
    st.inputs.slice_direction =  slice_dir
    st.inputs.interleaved = True
    st.inputs.time_repetition = TR
    return st.run()


def _clean_img_wrapper(in_file, out_file, low_pass, high_pass, TR):
    clean_img = nimg.clean_img(in_file, confounds=None, detrend=True, standardize=False, low_pass = low_pass, high_pass = high_pass, t_r = TR)
    nib.save(clean_img, out_file)


def convert_to_sphinx(input_dirs: List[str], output: Union[None, str] = None, fname='f.nii', scan_pos = 'HFP') -> str:
    """
    Convert to sphinx
    :param input_dirs: paths to dirs with input nii files, (likely in the MION or BOLD dir)
    :param fname: the expected name of unpacked nifitis
    :param output: output directory to create or populate (if None put in same dirs as input)
    :param scan_pos: String specifying the scanner position of the monkey. If 'HFS' (head first supine), the sphinx
    will be correct. If 'HFP' (head first prone), the sphinx transformation will not be correct, and the orientation
    will have be changed from RAS to RAI.
    :return: path to output directory
    """
    assert scan_pos == 'HFP' or scan_pos == 'HFS', "Parameter scan_pos must be either 'HFP' or 'HFS'"

    args_sphinx = []
    args_flip = []
    in_orientation = 'RAS' # Only applicable for flipping
    out_orientation = 'RAI' # Only applicable for flipping

    for scan_dir in input_dirs:
        os.environ.setdefault("SUBJECTS_DIR", scan_dir)
        cvt = freesurfer.MRIConvert()
        in_file = os.path.join(scan_dir, fname)
        if not output:
            local_out = os.path.join(scan_dir, 'f_sphinx.nii')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(scan_dir))
            local_out = os.path.join(out_dir, 'f_sphinx.nii')
        args_sphinx.append((in_file, local_out, cvt))
        args_flip.append((local_out, local_out, in_orientation, out_orientation, cvt))
    if scan_pos == 'HFS':
        with Pool() as p:
            res = p.starmap(_mri_convert_sphinx_wrapper, args_sphinx)
        return res
    elif scan_pos == 'HFP':
        with Pool() as p:
            res = p.starmap(_mri_convert_sphinx_wrapper, args_sphinx)
            res2 = p.starmap(_mri_convert_wrapper, args_flip)
        return res2


def _mcflt_wrapper(in_file, out_file, mcflt):
    mcflt.inputs.in_file = in_file
    mcflt.inputs.cost = 'mutualinfo'
    mcflt.inputs.out_file = out_file
    mcflt.inputs.save_plots = True
    mcflt.inputs.save_rms = True
    mcflt.cmdline
    return mcflt.run()


def motion_correction(input_dirs: List[str], output: Union[None, str] = None, fname='f.nii',
                      check_rms=True, abs_threshold=.8, var_threshold=.2) -> Union[List[str], None]:
    """
    preform fsl motion correction. If check rms is enabled will remove data where too much motion is detected.
    :param var_threshold:
    :param abs_threshold:
    :param output:
    :param input_dirs:
    :param fname:
    :param check_rms:
    :return:
    """
    args = []
    out_dirs = []
    for source_dir in input_dirs:
        outputs = []
        if os.path.isfile(os.path.join(source_dir, fname)):
            mcflt = fsl.MCFLIRT()
            if not output:
                out_dir = source_dir
                local_out = os.path.join(source_dir, 'moco.nii.gz')
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(out_dir, 'moco.nii.gz')
            path = os.path.join(source_dir, fname)
            args.append((path, local_out, mcflt))
            out_dirs.append(out_dir)
    with Pool() as p:
        res = p.starmap(_mcflt_wrapper, args)
    if check_rms:
        output = './'
        good_moco = plot_moco_rms_displacement(out_dirs, output, abs_threshold, var_threshold)
        return good_moco


def slice_time_correction(input_dirs: List[str], output: Union[None, str] = None, fname='moco.nii.gz', slice_dir = 3, TR = 3):
    '''
    slice_time_correction: Performs slice timing on motion corrected (or non motion corrected) data.
    :param input_dirs:
    :param output:
    :param fname:
    :param TR:
    :return:
    '''
    args = []
    for source_dir in input_dirs:
        outputs = []
        if os.path.isfile(os.path.join(source_dir, fname)):
            st = fsl.SliceTimer()
            if not output:
                out_dir = source_dir
                local_out = os.path.join(source_dir, 'slc.nii.gz')
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(source_dir, 'slc.nii.gz')
            path = os.path.join(source_dir, fname)
            args.append((path, local_out, slice_dir, TR, st))
    with Pool() as p:
        res = p.starmap(_slice_time_wrapper, args)


def image_cleaner(input_dirs: List[str], output: Union[None, str] = None, fname='slc.nii.gz', low_pass = 0.08, high_pass = 0.009, TR = 3):
    '''
    image_cleaner: Detrends image and applies high and low pass filtering
    :param input_dirs:
    :param output:
    :param fname:
    :param TR:
    :return:
    '''
    args = []
    for source_dir in input_dirs:
        if os.path.isfile(os.path.join(source_dir,fname)):
            if not output:
                out_dir = source_dir
                local_out = os.path.join(source_dir, 'clean.nii.gz')
            else:
                out_dir = _create_dir_if_needed(output,os.path.basename(source_dir))
                local_out = os.path.join(source_dir, 'clean.nii.gz')
            path = os.path.join(source_dir,fname)
            args.append((path,local_out,low_pass, high_pass, TR))
    with Pool() as p:
        res = p.starmap(_clean_img_wrapper, args)


def check_time_series_length(input_dirs: List[str], fname='f.nii.gz', expected_length: int=279):
    good = []
    for source_dir in input_dirs:
        path = os.path.join(source_dir, fname)
        file = nib.load(path)
        data = file.get_fdata()
        if data.shape[-1] == expected_length:
            good.append(source_dir)
    return good


def plot_moco_rms_displacement(transform_file_dirs: List[str], save_loc: str, abs_threshold, var_threshold) -> List[str]:
    fig, ax = plt.subplots(1)
    good_moco = []
    for transform_file_dir in transform_file_dirs:
        for f in os.listdir(transform_file_dir):
            if f == 'moco.nii.gz_abs.rms':
                disp_vec = np.loadtxt(os.path.join(transform_file_dir, f))
                if len(disp_vec.shape) == 0:
                    print(os.path.join(transform_file_dir, f))
                    continue
                label = os.path.basename(transform_file_dir)
                ax.plot(disp_vec, label=label)
                is_good_moco = max(disp_vec) <= abs_threshold and np.std(disp_vec) <= var_threshold
                if is_good_moco:
                    good_moco.append(transform_file_dir)
    ax.set_title("MOCO RMS Vector Translation (mm)")
    ax.set_xlabel("frame number")
    fig.legend(loc='upper left')
    fig.show()
    fig.savefig(save_loc + '/moco_rms_displacement.eps', format='eps')
    return good_moco


def _flirt_wrapper(in_file, out_file, mat_out_file, temp_file, flt, dof=12):
    flt.inputs.in_file = in_file
    flt.inputs.reference = temp_file
    flt.inputs.dof = dof
    flt.inputs.out_file = out_file
    flt.inputs.args = '-omat ' + mat_out_file
    flt.cmdline
    try:
        out = flt.run()
    except Exception:
        out = None
    return out


def linear_affine_registration(functional_input_dirs: List[str], template_file: str, fname: str = 'stripped.nii.gz', output: str = None):
    flt = fsl.FLIRT()
    args = []
    for source_dir in functional_input_dirs:
        if not os.path.isdir(source_dir):
            print("Failure", sys.stderr)
        chosen_name = fname.split('.')[0]
        if not output:
            local_out = os.path.join(source_dir, chosen_name + '_flirt.nii.gz')
            mat_out = os.path.join(source_dir, chosen_name + '_flirt.mat')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
            local_out = os.path.join(out_dir, chosen_name + '_flirt.nii.gz')
            mat_out = os.path.join(out_dir, chosen_name + '_flirt.mat')
        in_file = os.path.join(source_dir, fname)
        args.append((in_file, local_out, mat_out, template_file, flt, 12))
    with Pool() as p:
        return p.starmap(_flirt_wrapper, args)


def _nirt_wrapper(in_file, out_file, temp_file, affine_mat_file, fnt):
    fnt.inputs.in_file = in_file
    fnt.inputs.ref_file = temp_file
    fnt.inputs.affine_file = affine_mat_file
    fnt.cmdline
    out = fnt.run()

    # workaround fsl fnirt cout error
    source_dir = os.path.dirname(out_file)
    warp_file = os.path.join(source_dir, [s for s in os.listdir(source_dir) if '_warpcoef' in s][0])
    shutil.copy(warp_file, out_file)
    os.remove(warp_file)
    return out


def nonlinear_registration(functional_input_dirs: List[str], transform_input_dir: List[str], template_file: str,
                           source_fname: str = 'stripped.nii.gz', affine_fname: str = 'stripped_flirt.mat',
                           output: str = None):
    fnt = fsl.FNIRT()
    args = []
    try:
        sources = zip(functional_input_dirs, transform_input_dir)
    except Exception:
        print("function_input_dirs and transform_input_dirs must be lists of the same length.", sys.stderr)
        exit(-2)
    for source_dir, transform_dir in sources:
        try:
            files = os.listdir(source_dir)
            tfiles = os.listdir(transform_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        if source_fname in files and affine_fname in tfiles:
            if not output:
                local_out = os.path.join(source_dir, 'reg_tensor.nii.gz')
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(out_dir, 'reg_tensor.nii.gz')
            in_file = os.path.join(source_dir, source_fname)
            affine_mat_file = os.path.join(transform_dir, affine_fname)
            args.append((in_file, local_out, template_file, affine_mat_file, fnt))
        else:
            raise FileNotFoundError("The specified source nifti or affine transform matrix cannot be found.")
    with Pool() as p:
        return p.starmap(_nirt_wrapper, args)


def _apply_warp_wrapper(in_file, out_file, temp_file, warp_coef, apw):
    apw.inputs.in_file = in_file
    apw.inputs.ref_file = temp_file
    apw.inputs.field_file = warp_coef
    apw.inputs.out_file = out_file
    apw.cmdline
    return apw.run()


def preform_nifti_registration(functional_input_dirs: List[str], transform_input_dir: Union[None, List[str]] = None, template_file: str = None,
                    output: str = None, source_fname: str = 'stripped.nii.gz', transform_fname: str = 'reg_tensor.nii.gz'):
    """
    Apply a registration file to a 4D nifti and save the output.
    :param functional_input_dirs: Directories where we expect to find scan directories containing 4D input niftis.
    :param transform_input_dir: (default: None) Directories where we expect to find scan directories containing combined lin,
                            non-lin transforms. If None assumed to be the same as input directories.
    :param output:  (default: None) Parent folder to place scan directories containing transformed 4D niftis. If None
                    same as source.
    :param input_fname:  (default: 'moco.nii.gz') expected base filename for input 4D niftis.
    :param transform_fname:  (default: 'register.dat'):  expected base filename for registration transforms.
    :return:
    """
    apw = fsl.ApplyWarp()
    args = []
    try:
        sources = zip(functional_input_dirs, transform_input_dir)
    except Exception:
        print("function_input_dirs and transform_input_dirs must be lists of the same length.", sys.stderr)
        exit(-2)
    for source_dir, transform_dir in sources:
        try:
            files = os.listdir(source_dir)
            tfiles = os.listdir(transform_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        if source_fname in files and transform_fname in tfiles:
            if not output:
                local_out = os.path.join(source_dir, 'registered.nii.gz')
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(out_dir, 'registered.nii.gz')
            in_file = os.path.join(source_dir, source_fname)
            field_file = os.path.join(transform_dir, transform_fname)
            args.append((in_file, local_out, template_file, field_file, apw))
        else:
            raise FileNotFoundError("The specified source nifti or nonlinear transform tensor cannot be found.")
    with Pool() as p:
        return p.starmap(_apply_warp_wrapper, args)


def fix_nii_headers(input_dirs: List[str], output: str, fname: str = 'nirt.nii.gz', tr=2000):
    for scan_dir in input_dirs:
        files = os.listdir(scan_dir)
        os.environ.setdefault("SUBJECTS_DIR", scan_dir)
        cvt = freesurfer.MRIConvert()
        for f in files:
            if len(f) > 3 and f == fname:
                cvt.inputs.in_file = os.path.join(scan_dir, f)
                if not output:
                    local_out = os.path.join(scan_dir, 'fixed.nii')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(scan_dir))
                    local_out = os.path.join(out_dir, 'fixed.nii')
                cvt.inputs.out_file = local_out
                cvt.inputs.tr = tr
                cvt.inputs.out_type = 'nii'
                cvt.cmdline
                cvt.run()


def _pad_to_cube(arr: np.ndarray, time_axis=3):
    """
    makes spatial dims have even size.
    :param arr:
    :param time_axis:
    :return:
    """
    if time_axis < np.ndim(arr):
        size = max(arr.shape[:time_axis] + arr.shape[time_axis+1:])
        print(size)
    else:
        size = max(arr.shape)
        # print('ruhroh')

    ax_pad = [0] * np.ndim(arr)
    for i in range(len(ax_pad)):
        if i != time_axis:
            ideal = (size - arr.shape[i]) / 2
            ax_pad[i] = (int(np.floor(ideal)), int(np.ceil(ideal)))
        else:
            ax_pad[i] = (0, 0)
    arr = np.pad(arr, ax_pad, mode='constant', constant_values=(0, 0))
    return arr


def _nifti_load_and_call_wrapper(path: str, fxn, output, to_center, *args):
    nii = nib.load(path)
    nii_data = nii.get_fdata()
    nii_data = fxn(nii_data, *args)
    aff = nii.affine
    if to_center:
        aff[:3, 3] = 0
    new_nii = nib.Nifti1Image(nii_data, affine=aff, header=nii.header)
    nib.save(new_nii, output)
    return


def center_nifti(source_dir:str, fname:str ='orig.mgz', output=None):
    if not output:
        output = os.path.join(source_dir, 'orig.nii')
    nifti = nib.load(os.path.join(source_dir, fname))
    nii_aff = nifti.affine
    nii_aff[:3, 3] = np.zeros(3)
    centered = nib.Nifti1Image(nifti.get_fdata(), affine=nii_aff, header=nifti.header)
    nib.save(centered, output)


def create_low_res_anatomical(source_dir:str, fname:str ='orig.mgz', output=None, factor=3):
    if not output:
        output = os.path.join(source_dir, 'low_res.nii')
    factor = str(factor)
    in_path = os.path.join(source_dir, fname)
    subprocess.run(['mri_convert', in_path, '-vs', factor, factor, factor, output])
    subprocess.run(['mri_convert', output, '-iis', '1', '-ijs', '1', '-iks', '1', output])


def functional_to_cube(input_dirs: List[str], output: str = None, fname: str = 'moco.nii'):
    args = []
    for source_dir in input_dirs:
        source = os.path.join(source_dir, fname)
        if not os.path.exists(source):
            print("could not find file")
            exit(1)
        if not output:
            local_out = os.path.join(source_dir, 'f_cubed.nii')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
            local_out = os.path.join(out_dir, 'f_cubed.nii')
        args.append((source, _pad_to_cube, local_out, False))
    with Pool() as p:
        p.starmap(_nifti_load_and_call_wrapper, args)


def smooth(input_dirs: List[str], output: str, fname: str = 'fixed.nii', bright_tresh=1000.0, fwhm=4.0):
    sus = fsl.SUSAN()
    outputs = []
    for source_dir in input_dirs:
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        for source in files:
            if fname in source:
                if not output:
                    local_out = os.path.join(source_dir, 'smooth.nii.gz')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    local_out = os.path.join(out_dir, 'smooth.nii.gz')
                sus.inputs.in_file = os.path.join(source_dir, source)
                sus.inputs.fwhm = fwhm
                sus.inputs.brightness_threshold = bright_tresh
                sus.cmdline
                out = sus.run()
                shutil.copy('./' + fname[:-4] + '_smooth.nii.gz', local_out)
                os.remove('./' + fname[:-4] + '_smooth.nii.gz')
                outputs.append(out)
        return outputs


def _bet_wrapper(in_file, out_file, functional, bet, frac):
    bet.inputs.in_file = in_file
    bet.inputs.out_file = out_file
    bet.inputs.mask = True
    bet.inputs.functional = functional
    bet.inputs.frac = frac
    bet.inputs.args = '-R'
    #bet.inputs.remove_eyes = True
    bet.cmdline
    out = bet.run()


def skull_strip(input_dirs: List[str], output: Union[str, None] = None, fname: str = 'moco.nii.gz', is_time_series=True,
                fractional_thresh=.5, center=None):
    bet = fsl.BET()
    args = []
    for source_dir in input_dirs:
        if fname not in os.listdir(source_dir):
            print("could not find file")
            exit(1)
        if not output:
            local_out = os.path.join(source_dir, 'stripped.nii.gz')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
            local_out = os.path.join(out_dir, 'stripped.nii.gz')
        in_file = os.path.join(source_dir, fname)
        args.append((in_file, local_out, is_time_series, bet, fractional_thresh))
    with Pool() as p:
        p.starmap(_bet_wrapper, args)


def normalize(input_dirs: List[str], output: str = None, fname='smooth.nii.gz', drift_correction=True):
    """
    Centers and normalizes the intensity data using (X - mu) / std. Creates a normalized nifti
    :param input_dirs:
    :param output:
    :param fname:
    :param drift_correction: Only use drift correction if the average stimulus across the whole run is the same. Works
                             best if stimuli are presented the same number of times at regular intervals.
    :return:
    """
    for source_dir in input_dirs:
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        for source in files:
            if fname in source:
                if not output:
                    local_out = os.path.join(source_dir, 'normalized.nii')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    local_out = os.path.join(out_dir, 'normalized.nii')
                nifti = nib.load(os.path.join(source_dir, source))
                n_data = np.array(nifti.get_fdata())
                n_data = _normalize_array(n_data)
                if drift_correction:
                    n_data = _drift_correction(n_data)
                new_nifti = nib.Nifti1Image(n_data, affine=nifti.affine, header=nifti.header)
                nib.save(new_nifti, local_out)
                print(nifti.header)


def create_functional_mask(input_dirs: List[str], output: str, fname='normalized.nii.gz', thresh: Union[float, None] = None):
    """
    creates a binary mask of a input nifti. By default sets threshold at 3 std devs above the mean.
    :param input_dirs:
    :param output:
    :param fname:
    :param thresh:
    :return:
    """
    for source_dir in input_dirs:
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        for source in files:
            if fname in source:
                if not output:
                    mask_out = os.path.join(source_dir, 'bin_mask.nii')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    mask_out = os.path.join(out_dir, 'bin_mask.nii')
                nifti = nib.load(os.path.join(source_dir, source))
                n_data = np.array(nifti.get_fdata())
                if not thresh:
                    u = np.mean(n_data.flatten())
                    s = np.std(n_data.flatten())
                    thresh = u + 3*s
                n_data[n_data > thresh] = 9999999
                n_data[n_data <= thresh] = 0
                n_data /= 9999999
                new_nifti = nib.Nifti1Image(n_data, affine=nifti.affine, header=nifti.header)
                nib.save(new_nifti, mask_out)


