from __future__ import print_function
from __future__ import division

from pathlib import Path

import math
import ntpath

import scipy.ndimage
import shutil
import sys
from typing import List, Union, Tuple, Dict

import numpy as np
from nipype.interfaces import freesurfer
from builtins import str
from builtins import range

import os  # system functions
import subprocess

import glob

from scipy.ndimage import zoom, gaussian_filter
import filters

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as model  # model generation
import nipype.algorithms.rapidart as ra  # artifact detection
import nibabel as nib

# don't require nilearn
try:
    from nilearn import image as nimg
except (ModuleNotFoundError, ImportError):
    print("Nilearn not installed")

from matplotlib import pyplot as plt
from multiprocessing import Pool


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


def spatial_intensity_normalization(arr: np.ndarray):
    if np.ndim(arr) > 3:
        raise ValueError("must be 3 or fewer dimensions")
    flat = arr.flatten()
    mean = np.mean(flat)
    disp_arr = arr - mean
    disp_arr = scipy.ndimage.gaussian_filter(disp_arr, sigma=10)
    arr = arr - disp_arr
    return arr


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


def _apply_binary_mask_3D(in_img: nib.Nifti1Image, mask: nib.Nifti1Image) -> nib.Nifti1Image:
    n_data = np.array(in_img.get_fdata())
    mask = np.array(mask.get_fdata())
    print(n_data.shape)
    if len(n_data.shape) > 3:
        mask = mask[:, :, :, None]
        mask = np.tile(mask, (1, 1, 1, n_data.shape[-1]))
    print(mask.shape)
    n_data[mask <= .5] = 0
    print(n_data.shape)
    new_nifti = nib.Nifti1Image(n_data, affine=in_img.affine, header=in_img.header)
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
    st.inputs.slice_direction = slice_dir
    st.inputs.interleaved = True
    st.inputs.time_repetition = TR
    return st.run()


def _clean_img_wrapper(in_file, out_file, low_pass, high_pass, TR):
    clean_img = nimg.clean_img(in_file, confounds=None, detrend=True, standardize=False, low_pass = low_pass, high_pass = high_pass, t_r = TR)
    nib.save(clean_img, out_file)


def convert_to_sphinx(input_dirs: List[str], scan_pos='HFP', output: Union[None, str] = None, fname='f_nordic.nii'):
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
            local_out = os.path.join(scan_dir, fname.split('.')[0] + '_sphinx.nii')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(scan_dir))
            local_out = os.path.join(out_dir, fname.split('.')[0] + '_sphinx.nii')
        args_sphinx.append((in_file, local_out, cvt))
        args_flip.append((local_out, local_out, in_orientation, out_orientation, cvt))
    if scan_pos == 'HFS':
        with Pool() as p:
            res = p.starmap(_mri_convert_sphinx_wrapper, args_sphinx)
    elif scan_pos == 'HFP':
        with Pool() as p:
            res = p.starmap(_mri_convert_sphinx_wrapper, args_sphinx)
            res2 = p.starmap(_mri_convert_wrapper, args_flip)
    return input_dirs


def _mcflt_wrapper(in_file, out_file, ref_file, mcflt):
    mcflt.inputs.in_file = in_file
    mcflt.inputs.cost = 'normcorr'
    mcflt.inputs.ref_file = ref_file
    mcflt.inputs.out_file = out_file
    mcflt.inputs.dof = 12
    mcflt.inputs.save_plots = True
    mcflt.inputs.save_rms = True
    mcflt.cmdline
    return mcflt.run()


def NORDIC(input_dirs: List[str], noise_path=None, filename='f_nordic'):
    """
    Perform NORDIC denoising of images. Should be done before any other manipulation of images that might disrupt the
    noise pattern.
    :param input_dirs: A list of strings corresponding to the functional images you wish to denoise.
    :param noise_path: Path to the noise image. Generally recommended, but if not used, the NORDIC function will
    try and estimate the noise itself.
    :param filename: A string of the new filename
    :return: A list of strings corresponding to the outputs.
    """
    b_path = os.path.dirname(Path(__file__).absolute())
    print(b_path)
    os.chdir(b_path)
    if noise_path is None:
        noise_path = 'None' # Matlab needs a text/char input, not None
    cmd = '$MATLAB -nojvm -r '
    fun = ' "monk_nordic({},{},{}); exit;"'.format("{'"+"','".join(input_dirs)+"'}", "'" + noise_path + "'", "'"+filename+"'")
    cmd = cmd + fun
    print(cmd)
    subprocess.run(cmd, shell=True)
    os.chdir('..')
    out_dirs = [os.path.join(os.path.dirname(input_dir), filename+'.nii') for input_dir in input_dirs]
    return out_dirs


def motion_correction(input_dirs: List[str], ref_path: str, outname='moco.nii.gz', output: Union[None, str] = None, fname='f_sphinx.nii',
                      check_rms=True, abs_threshold=3, var_threshold=1) -> Union[List[str], None]:
    """
    preform fsl motion correction. If check rms is enabled will remove data where too much motion is detected.
    :param var_threshold:
    :param abs_threshold:
    :param output:
    :param ref_path: path to image to use as reference for ALL moco
    :param input_dirs:
    :param fname:
    :param check_rms:
    :return:
    saves rms correction displacement at moco.nii.gz_abs.rms
    """
    args = []
    out_dirs = []
    for source_dir in input_dirs:
        outputs = []
        if os.path.isfile(os.path.join(source_dir, fname)):
            mcflt = fsl.MCFLIRT()
            if not output:
                out_dir = source_dir
                local_out = os.path.join(source_dir, outname)
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(out_dir, outname)
            path = os.path.join(source_dir, fname)
            args.append((path, local_out, ref_path, mcflt))
            out_dirs.append(out_dir)
    with Pool() as p:
        res = p.starmap(_mcflt_wrapper, args)
    if check_rms:
        output = './'
        good_moco = plot_moco_rms_displacement(out_dirs, output, abs_threshold, var_threshold)
        return good_moco
    else:
        return input_dirs


def nonlinear_moco(moving_epi, reference, outfile):
    out_no_ext = outfile.split('.')[0]
    mc_cmd = "antsMotionCorr  -d 3 -o [{output},{output}.nii.gz,{avg}] " \
             " -m MI[ {avg} , {inputB} , 1 , 1 , Random, 0.05  ] -t Affine[ 0.01 ] -i 10 -u 1 -e 1 -s 0 -f 1 " \
             " -m CC[  {avg}, {inputB} , 1 , 3] -t SyN[0.1, 3, 0] -i 30 -u 1 -e 1 -f 2 -s 1" \
             " -m CC[  {avg}, {inputB} , 1 , 3] -t SyN[0.05, 3, 0] -i 10 -u 1 -e 1 -f 1 -s 0" \
             " -v\n".format(output=out_no_ext, inputB=moving_epi, avg=reference)
    subprocess.call(mc_cmd, shell=True)
    return outfile


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


def sample_frames(SOURCE: List[str], num_samples, output=None, fname='f_nordic.nii') -> str:
    """
    Returns a frame to in the middle of a session
    :param out: output path
    :param SOURCE: list of source run dirs
    :param fname: name of nifti in run dirs
    :return: path to output 3d image
    """
    run_dir_idxs = np.random.choice(len(SOURCE), num_samples)
    out_paths = []
    for i, run_dir_idx in enumerate(run_dir_idxs):
        run_dir = SOURCE[run_dir_idx]
        in_file = os.path.join(run_dir, fname)

        if os.path.exists(in_file) and '.nii' in fname:
            nifti = nib.load(in_file)
            data = nifti.get_fdata()
            dims = data.shape
            if len(dims) == 3:
                data = data[:, :, :, None]
                dims = data.shape
            elif len(dims) < 3:
                raise ValueError("must pass 4D timeseries or Volume inputs")
            target_idx = np.random.choice(dims[3])
            frame = data[:, :, :, target_idx]
            output = os.path.join(os.path.dirname(run_dir), '3d_epi_rep' + str(run_dir_idx) + '_' + str(target_idx) + '.nii')
            frame = np.squeeze(frame)
            rep_img = nib.Nifti1Image(frame, affine=nifti.affine, header=nifti.header)
            nib.save(rep_img, output)
            out_paths.append(output)
    return out_paths


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


def bandpass_filter_functional(input_file, out_file, low_period, plot=True, save_out=True):
    """
    frequencies in cycles / tr
    :param input_file:
    :param out_file:
    :param block_length_trs:
    :param plot:
    :param save_out:
    :return:
    """
    fnii = nib.load(input_file)
    fdata = np.array(fnii.get_fdata())

    filtered_nii = filters.butter_bandpass_filter(fdata,
                                                  low_freq_cutoff=1 / low_period,
                                                  high_freq_cutoff=.5,
                                                  fs=1,
                                                  order=5)
    if plot:
        mean_og = np.mean(fdata, axis=(0, 1, 2))
        mean_filt = np.mean(filtered_nii, axis=(0, 1, 2))
        fig, axs = plt.subplots()
        axs.plot(mean_og)
        axs.plot(mean_filt)
    if save_out:
        out_nii = nib.Nifti1Image(filtered_nii, affine=fnii.affine, header=fnii.header)
        nib.save(out_nii, out_file)
    return filtered_nii


def batch_bandpass_functional(functional_input_dirs, block_length_trs, fname='epi_masked.nii', out_name='epi_filtered.nii.gz'):
    args = []
    if block_length_trs == 1:
        period = int(input("enter max length in trs that we expect to see meaningful signal change "
                           "(e.g. probably about 2 blocklengths) "))
    else:
        period = 3 * block_length_trs

    for source_dir in functional_input_dirs:
        if not os.path.isdir(source_dir):
            print("Failure", sys.stderr)
        in_file = os.path.join(source_dir, fname)
        if not os.path.exists(in_file):
            print('Failed to find requested file')
            exit()
        out_file = os.path.join(source_dir, out_name)
        args.append((in_file, out_file, period, False, True))
    with Pool() as p:
        p.starmap(bandpass_filter_functional, args)
    return functional_input_dirs


def linear_affine_registration(functional_input_dirs: List[str], template_file: str, fname: str = 'stripped.nii.gz', output: str = None, dof=12):
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
        args.append((in_file, local_out, mat_out, template_file, flt, dof))
    with Pool() as p:
        return p.starmap(_flirt_wrapper, args)


def antsApplyTransforms(inP, refP, outP, lTrns, interp, img_type_code=3, dim=3, invertTrans: Union[List, bool] = False):
    ''' might want to try interp='BSpline'
    lTrans is a list of paths to transform files (.e.g, .h5)
    I think invert trans will just work...

    refP indicates the dimensions desired...

    from kurts code
    '''
    if type(invertTrans) is bool:
        invertTrans = [invertTrans] * len(lTrns)
    elif type(invertTrans) is not list or len(invertTrans) != len(lTrns):
        raise ValueError

    from nipype.interfaces.ants import ApplyTransforms
    at = ApplyTransforms()
    at.inputs.transforms = lTrns
    at.inputs.dimension = dim
    at.inputs.input_image = inP
    at.inputs.input_image_type = img_type_code
    #    if initialTrsnfrmP!=None:
    #        at.inputs.initial_moving_transform = initialTrsnfrmP
    at.inputs.reference_image = refP
    at.inputs.output_image = outP
    at.inputs.interpolation = interp
    at.inputs.invert_transform_flags = invertTrans
    #    at.inputs.verbose = 1
    print(at.cmdline)
    at.run()


def ResampleImageToTarget(in_vol, target_vol, out_path, interp='interpolate'):
    """
    takes a nifti and transforms the true data matrix such that the affine is scaled identity.
    :return:
    """
    antsApplyTransforms(in_vol, target_vol, out_path, lTrns=['identity'], interp=interp)
    return out_path


def antsCoreg(fixedP, movingP, outP, initialTrsnfrmP=None,
              across_modalities=False, outPref='antsReg',
              run=True, n_jobs=64, full=False):
    """
    From kurts code
    :param fixedP:
    :param movingP:
    :param outP:
    :param initialTrsnfrmP:
    :param across_modalities:
    :param outPref:
    :param transforms:
    :param run:
    :param n_jobs:
    :return:
    """
    cmd = "antsRegistration" \
          " --verbose 1 --dimensionality 3 --float 0 --collapse-output-transforms 1 --write-composite-transform 1" \
          " --output [ ./,./Warped.nii.gz,./InverseWarped.nii.gz ] " \
          " --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ]" \
          " --initial-moving-transform [" + fixedP + "," + movingP + ",1 ]" \
          " --transform Rigid[ 0.1 ] --metric MI[" + fixedP + "," + movingP + ",1,32,Regular,0.25 ]" \
          " --convergence [1000x500x250,1e-8,10 ] --shrink-factors 4x2x1 --smoothing-sigmas 2x1x0vox" \
          " --transform Affine[ 0.1 ] --metric MI[ " + fixedP + "," + movingP + ",1,32,Regular,0.25 ]" \
          " --convergence [1000x500,1e-9,10 ] --shrink-factors 2x1 --smoothing-sigmas 1x0vox" \
          " --transform Affine[ 0.01 ] --metric MI[ " + fixedP + "," + movingP + ",1,32,Regular,0.25 ]" \
          " --convergence [500,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox" \
          " --transform SyN[ .1,4,0 ] --metric CC[ " + fixedP + "," + movingP + ",1,6 ]" \
          " --convergence [200,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox" \
          " --transform SyN[ .01,3,0 ] --metric CC[ " + fixedP + "," + movingP + ",1,5 ]" \
          " --convergence [200,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox" \
          " --transform SyN[ .01,3,0 ] --metric CC[ " + fixedP + "," + movingP + ",1,3 ]" \
          " --convergence [200,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox" \
          " --transform SyN[ .001,3,0 ] --metric CC[ " + fixedP + "," + movingP + ",1,3 ]" \
          " --convergence [100,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox"
    print(cmd)
    subprocess.call(cmd, shell=True)


def antsCoReg(fixedP, movingP, outP, ltrns=['Affine', 'SyN'], n_jobs=2, full=False):
    """
    From kurks code
    :param fixedP:
    :param movingP:
    :param outP:
    :param ltrns:
    :param n_jobs:
    :return:
    """
    outF = os.path.dirname(outP)
    os.chdir(outF)
    antsCoreg(fixedP,
        movingP,
        outP,
        initialTrsnfrmP=None, # we are working on resampled img
        across_modalities=True,
        outPref='antsReg',
        run=True,n_jobs=n_jobs, full=full)
    frwdTrnsP = os.path.join(outF, 'Composite.h5')
    invTrnsP = os.path.join(outF, 'InverseComposite.h5')
    return frwdTrnsP, invTrnsP


def _itkSnapManual(anatP, funcP, outF):
    '''manually register to anat
       author: kurt'''
    print('-'*30)
    s = 'do manual reg with ITK SNAP: \n\nmain image: \n%s'%(os.path.abspath(anatP))
    s+='\n\nsecondaryimage: \n%s'%(os.path.abspath(funcP))
    step1NiiP = outF+'/itkManual.nii.gz'
    step1TxtP = outF+'/itkManual.txt'
    s+='\n\nsave nii as: \n%s'%(os.path.abspath(step1NiiP))
    s+='\n\nand transform as: \n%s'%(os.path.abspath(step1TxtP))
    print(s)
    done = 'y' in input('done? (y/n): ')
    return step1TxtP, step1NiiP


def manual_itksnap_registration(functional_input_dirs: List[str], template_file: str, fname='moco.nii.gz'):
    """
    From kurt
    :param functional_input_dirs:
    :param fname:
    :param template_file:
    :return:
    """
    for source_dir in functional_input_dirs:
        if not os.path.isdir(source_dir):
            print("Failure", sys.stderr)
        in_file = os.path.join(source_dir, fname)
        transform, reg_nii = _itkSnapManual(os.path.abspath(template_file),
                                            os.path.abspath(in_file),
                                            os.path.abspath(source_dir))
        return transform, reg_nii


def _nirt_wrapper(in_file, out_file, temp_file, affine_mat_file, fnt):
    fnt.inputs.in_file = in_file
    fnt.inputs.ref_file = temp_file
    if affine_mat_file:
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
                local_out = os.path.join(source_dir, output)
            in_file = os.path.join(source_dir, source_fname)
            affine_mat_file = None
            if affine_fname:
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
    if type(transform_input_dir) is str:
        transform_input_dir = [transform_input_dir] * len(functional_input_dirs)
    try:
        sources = zip(functional_input_dirs, transform_input_dir)
    except Exception:
        print("function_input_dirs and transform_input_dirs must be lists of the same length.", sys.stderr)
        exit(-2)
    for source_dir, transform_dir in sources:
        in_file = os.path.join(source_dir, source_fname)
        field_file = os.path.join(transform_dir, transform_fname)
        if os.path.exists(in_file) and os.path.exists(field_file):
            if not output:
                local_out = os.path.join(source_dir, 'registered.nii.gz')
            else:
                local_out = os.path.join(source_dir, output)

            args.append((in_file, local_out, template_file, field_file, apw))
        else:
            raise FileNotFoundError("The specified source nifti or nonlinear transform tensor cannot be found. \n "
                                    "In file: " + in_file,
                                    "\nwarp field file: " + field_file)
    with Pool() as p:
        return p.starmap(_apply_warp_wrapper, args)


def estimate_warp_field(hf_source=(), fh_source=(), lr_source=(), rl_source=(), fname='f.nii', warpfield_out_dir=None, output=None):
    """
    Estmates the wrpfield f=using epis with different encoding directions.
    :param hf_source: Source files for hf encoded epis
    :param fh_source:  Source files for fh encoded epis
    :param lr_source:  Source files for rl encoded epis
    :param rl_source:  Source files for lr encoded epis
    :param fname: fname to look for
    :param warpfield_out_dir: full path for output warpfield files directory.
    :param output: name for corrected output nii files
    :return:
    """
    out_buf = ""
    phase_estimation_nifti = None
    affine = None
    f_nii = None
    for i, encode_dir in enumerate([hf_source, fh_source, lr_source, rl_source]):
        for dir in encode_dir:
            if i == 0:
                aq_dat = '0 1 0 1 \n'
            elif i == 1:
                aq_dat = '0 -1 0 1 \n'
            elif i == 2:
                aq_dat = '1 0 0 1 \n'
            else:
                # i == 3
                aq_dat = '-1 0 0 1 \n'
            out_buf = out_buf + aq_dat
            file = os.path.join(dir, fname)
            f_nii = nib.load(file)
            f_data = f_nii.get_fdata()
            if phase_estimation_nifti is None:
                phase_estimation_nifti = f_data[:, :, :, 0].reshape(list(f_data.shape[0:3]) + [1])
                affine = f_nii.affine.reshape([-1] + list(f_nii.affine.shape))
            else:
                phase_estimation_nifti = np.concatenate([phase_estimation_nifti,
                                                         f_data[:, :, :, 0].reshape(list(f_data.shape[0:3]) + [1])],
                                                        axis=3)
                affine = np.concatenate([affine, f_nii.affine.reshape([-1] + list(f_nii.affine.shape))], axis=0)
    print('loaded data and created warp est inputs.')
    warp_est_nifti = nib.Nifti1Image(phase_estimation_nifti, affine=np.mean(affine, axis=0), header=f_nii.header)
    warp_est_path = os.path.join(warpfield_out_dir, 'warp_est_src.nii')
    nib.save(warp_est_nifti, warp_est_path)
    scan_param_path = os.path.join(warpfield_out_dir, 'scan_encode_params.txt')
    with open(scan_param_path, 'w') as f:
        f.write(out_buf)
    if output:
        subprocess.run(['topup',
                        '--imain=' + warp_est_path,
                        '--datain=' + scan_param_path,
                        '--config=b02b0.cnf',
                        '--out=' + warpfield_out_dir,
                        '--iout=' + os.path.join(warpfield_out_dir, output),
                        '--fout=' + os.path.join(warpfield_out_dir, 'warpfield_hertz.nii.gz')])
    else:
        subprocess.run(['topup',
                        '--imain=' + warp_est_path,
                        '--datain=' + scan_param_path,
                        '--config=b02b0.cnf',
                        '--out=' + warpfield_out_dir,
                        '--fout=' + os.path.join(warpfield_out_dir, 'warpfield_hertz.nii.gz')])
    output_locs = {'warp_field_coef': os.path.join(warpfield_out_dir, 'my_out_fieldcoef.nii.gz'),
                   'fixed_warp_nii': os.path.join(warpfield_out_dir, output)}
    print('warp field estimate created.')
    return output_locs


def _epi_reg_wrapper(epi, t1_whole, t1_stripped, fmap, out):
    if fmap:
        subprocess.run(['epi_reg', '--epi=' + epi, '--t1=' + t1_whole, '--t1brain=' + t1_stripped, '--fmap=' + fmap, '--out=' + out])
    else:
        subprocess.run(['epi_reg', '--epi=' + epi, '--t1=' + t1_whole, '--t1brain=' + t1_stripped,
                        '--out=' + out])


def auto_epi_reg(functional_input_dirs: List[str], t1_whole_head: str, t1_stripped: str, field_map_estimate: str = None,
                 fname: str = 'stripped.nii.gz', output: str = None):
    args = []
    for source_dir in functional_input_dirs:
        if not os.path.isdir(source_dir):
            print("Failure", sys.stderr)
        chosen_name = fname.split('.')[0]
        if not output:
            local_out = os.path.join(source_dir, chosen_name + '_reg.nii.gz')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
            local_out = os.path.join(out_dir, chosen_name + '_reg.nii.gz')
        in_file = os.path.join(source_dir, fname)
        args.append((in_file, t1_whole_head, t1_stripped, field_map_estimate, local_out))
    with Pool() as p:
        return p.starmap(_epi_reg_wrapper, args)


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


def create_low_res_anatomical(source_dir:str, fname:str ='orig.mgz', output=None, factor=[2, 2, 2], affine_scale=None, resample='interpolate'):
    if not output:
        output = os.path.join(source_dir, 'low_res.nii')
    in_path = os.path.join(source_dir, fname)
    cmd_resample = 'mri_convert %s -vs %s %s %s -rt %s %s --out_type nii'%(in_path, factor[0], factor[1], factor[2], resample, output)
    subprocess.run(cmd_resample,shell=True)
    if affine_scale is not None:
        if len(affine_scale) < 3: # we expect a length of three
            affine_scale = [affine_scale[0], affine_scale[0], affine_scale[0]]
            print('Warning: length of affine_scale argument is less than three, using first element for all arguments')
        cmd_affine_set = 'mri_convert %s -iis %s -ijs %s -iks %s -rt %s %s --out_type nii'%(output, affine_scale[0], affine_scale[1], affine_scale[2], resample, output)
        subprocess.run(cmd_affine_set,shell=True)
    return output


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


def _bet_wrapper(in_file, out_file, functional, bet, frac, remove_eye):
    bet.inputs.in_file = in_file
    bet.inputs.out_file = out_file
    bet.inputs.mask = True
    bet.inputs.functional = functional
    bet.inputs.frac = frac
    bet.inputs.args = '-R'
    bet.inputs.remove_eyes = remove_eye
    bet.cmdline
    out = bet.run()


def skull_strip(input_dirs: List[str], output: Union[str, None] = None, fname: str = 'moco.nii.gz', is_time_series=True,
                fractional_thresh=.5, center=None, remove_eyes=False):
    bet = fsl.BET()
    args = []
    for source_dir in input_dirs:
        if fname not in os.listdir(source_dir):
            print("could not find file")
            exit(1)
        if not output:
            local_out = os.path.join(source_dir, 'stripped.nii.gz')
        else:
            local_out = os.path.join(source_dir, output)
        in_file = os.path.join(source_dir, fname)
        args.append((in_file, local_out, is_time_series, bet, fractional_thresh, remove_eyes))
    with Pool() as p:
        p.starmap(_bet_wrapper, args)


def normalize(input_dirs: List[str], output: str = None, fname='smooth.nii.gz', drift_correction=True, spatial_intesity=False):
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
        if not output:
            local_out = os.path.join(source_dir, 'normalized.nii')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
            local_out = os.path.join(out_dir, 'normalized.nii')
        nifti = nib.load(os.path.join(source_dir, fname))
        n_data = np.array(nifti.get_fdata())
        if spatial_intesity:
            n_data = spatial_intensity_normalization(n_data)
        else:
            n_data = _normalize_array(n_data)
        if drift_correction and np.ndim(n_data) == 4:
            n_data = _drift_correction(n_data)
        else:
            print("Drift Correction off or no a time course")
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
