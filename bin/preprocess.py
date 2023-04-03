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
import datetime
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
    n_data[mask <= 0.5] = 0.
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


def convert_to_sphinx(input_dirs: List[str], scan_pos='HFP', output: Union[None, str] = None, fname='f_topup.nii.gz'):
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
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in input_dirs]
    assert scan_pos == 'HFP' or scan_pos == 'HFS', "Parameter scan_pos must be either 'HFP' or 'HFS'"

    args_sphinx = []
    args_flip = []
    in_orientation = 'RAS' # Only applicable for flipping
    out_orientation = 'RAI' # Only applicable for flipping
    out = []

    for in_file in sources:
        scan_dir = os.path.dirname(in_file)
        fname = os.path.basename(in_file)
        os.environ.setdefault("SUBJECTS_DIR", scan_dir)
        cvt = freesurfer.MRIConvert()
        if not output:
            local_out = os.path.join(scan_dir, fname.split('.')[0] + '_sphinx.nii')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(scan_dir))
            local_out = os.path.join(out_dir, fname.split('.')[0] + '_sphinx.nii')
        args_sphinx.append((in_file, local_out, cvt))
        args_flip.append((local_out, local_out, in_orientation, out_orientation, cvt))
        out.append(local_out)
    if scan_pos == 'HFS':
        with Pool() as p:
            res = p.starmap(_mri_convert_sphinx_wrapper, args_sphinx)
    elif scan_pos == 'HFP':
        with Pool() as p:
            res = p.starmap(_mri_convert_sphinx_wrapper, args_sphinx)
            res2 = p.starmap(_mri_convert_wrapper, args_flip)
    return out


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


def _NORDIC_NOISE_MERGE(input_dirs: List[str], noise_path, filename='f_noise'):
    """concatenates the noise image to the end of the timeseries"""
    out_dirs = [os.path.join(os.path.dirname(input_dir), filename+'.nii.gz') for input_dir in input_dirs]

    for input_dir,out_dir in zip(input_dirs,out_dirs):
        cmd = 'fslmerge -t {} {} {}'.format(out_dir,input_dir,noise_path)
        print(cmd)
        subprocess.run(cmd, shell=True)
    return out_dirs


def create_warp_field(image_1,image_2,image_1_enc='HF',image_2_enc='FH',number_images=None):
    dirpath = os.path.dirname(image_1)
    image_1_img = nib.load(image_1)
    image_2_img = nib.load(image_2)
    image_1_len = image_1_img.header.get_data_shape()[3]
    image_2_len = image_2_img.header.get_data_shape()[3]
    max_image_length = np.maximum(image_1_len,image_2_len)
    if number_images==None:
        number_images = max_image_length
    elif number_images > max_image_length:
        number_images = max_image_length

    if number_images == max_image_length and image_1_len == image_2_len:
        merged_image = nib.Nifti1Image(np.concatenate((image_1_img.get_fdata(),image_2_img.get_fdata()),axis=3),image_1_img.affine)
    else:
        merged_image = nib.Nifti1Image(np.concatenate((image_1_img.get_fdata()[:,:,:,0:number_images],image_2_img.get_fdata()[:,:,:,0:number_images]),axis=3),image_1_img.affine)

    merged_filepath = os.path.join(dirpath,'merged_SE_%s_images.nii.gz'%(number_images))
    nib.save(merged_image,merged_filepath)

    scan_param_filepath = os.path.join(dirpath,'scan_encode_params_%s_images.txt'%(number_images))

    out_buf = ""

    if 'HF' in image_1_enc:
        out_buf = out_buf + '0 1 0 0.0371245 \n'*number_images
    elif 'FH' in image_1_enc:
        out_buf = out_buf + '0 -1 0 0.0371245 \n'*number_images
    elif 'LR' in image_1_enc:
        out_buf = out_buf + '1 0 0 0.0371245 \n'*number_images
    elif 'RL' in image_1_enc:
        out_buf = out_buf + '-1 0 0 0.0371245 \n'*number_images

    if 'HF' in image_2_enc:
        out_buf = out_buf + '0 1 0 0.0371245 \n'*number_images
    elif 'FH' in image_2_enc:
        out_buf = out_buf + '0 -1 0 0.0371245 \n'*number_images
    elif 'LR' in image_2_enc:
        out_buf = out_buf + '1 0 0 0.0371245 \n'*number_images
    elif 'RL' in image_2_enc:
        out_buf = out_buf + '-1 0 0 0.0371245 \n'*number_images

    with open(scan_param_filepath, 'w') as f:
        f.write(out_buf)



    topup_prefix_out = os.path.join(dirpath,'topup_%s_images'%(number_images))

    topup_field_out = os.path.join(dirpath,'topup_field_%s_images'%(number_images))

    topup_unwarped_out = os.path.join(dirpath,'topup_unwarped_%s_images'%(number_images))

    cmd = 'topup --imain={} --datain={} --config=b02b0_macaque.txt --out={} --fout={} --iout={}'.format(merged_filepath,scan_param_filepath,topup_prefix_out,topup_field_out,topup_unwarped_out)

    t1 = datetime.datetime.now()

    print(cmd)
    subprocess.call(cmd,shell=True)
    t2 = datetime.datetime.now()
    delta = t2-t1
    print("topup with %s images finished in %s.%s seconds" %(number_images,delta.seconds,delta.microseconds))

    return topup_prefix_out


def applytopup(func_in,topup_basename,func_dir = 'HF'):

    out_buf = ""

    if 'HF' in func_dir:
        out_buf = out_buf + '0 1 0 0.0371245 \n'
    elif 'FH' in func_dir:
        out_buf = out_buf + '0 -1 0 0.0371245 \n'
    elif 'LR' in func_dir:
        out_buf = out_buf + '1 0 0 0.0371245 \n'
    elif 'RL' in func_dir:
        out_buf = out_buf + '-1 0 0 0.0371245 \n'
    source_dir = os.path.dirname(func_in)
    scan_param_filepath = os.path.join(source_dir, 'scan_encode_params.txt')
    with open(scan_param_filepath, 'w') as f:
        f.write(out_buf)

    func_out = os.path.join(source_dir, 'f_topup.nii.gz')

    cmd = 'applytopup --imain={} --inindex={} --datain={} --topup={} --out={} --method=jac'.format(func_in, 1,scan_param_filepath,topup_basename,func_out)
    
    print(cmd)
    subprocess.call(cmd,shell=True)

    return func_out 


def topup(input_dirs: List[str], image_1_path, image_2_path, image_1_enc, image_2_enc, number_images, func_enc, filename='f_topup'):
    """
    Perform topup of the images.
    """

    topup_prefix_out = create_warp_field(image_1_path,image_2_path,image_1_enc,image_2_enc,number_images)

    args = []

    for funcP in input_dirs:
        args.append((funcP,topup_prefix_out,func_enc))

    with Pool(32) as p:
        res = p.starmap(applytopup, args)
    return res


def NORDIC(input_dirs: List[str], filename='f_nordic'):
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
    noise_option = 0 # no longer used, vestigial

    cmd = '$MATLAB -nojvm -r '
    fun = ' "monk_nordic({},{},{}); exit;"'.format("{'"+"','".join(input_dirs)+"'}",noise_option, "'"+filename+"'")
    cmd = cmd + fun
    print(cmd)
    subprocess.run(cmd, shell=True)
    os.chdir('..')
    out_dirs = [os.path.join(os.path.dirname(input_dir), filename+'.nii') for input_dir in input_dirs]
    
    return out_dirs


def motion_correction(sources: List[str], ref_path: str, outname='moco.nii.gz', output: Union[None, str] = None, fname='f_topup_sphinx.nii',
                      check_rms=True, abs_threshold=3, var_threshold=1) -> Union[List[str], None]:
    """
    preform fsl motion correction. If check rms is enabled will remove data where too much motion is detected.
    :param var_threshold:
    :param abs_threshold:
    :param output:
    :param ref_path: path to image to use as reference for ALL moco
    :param sources:
    :param fname:
    :param check_rms:
    :return:
    saves rms correction displacement at moco.nii.gz_abs.rms
    """
    args = []
    out = []
    for source in sources:
        if os.path.isfile(source):
            source_dir = os.path.dirname(source)
        else:
            source_dir = source
            source = os.path.join(source, fname)
        if os.path.isfile(source):
            mcflt = fsl.MCFLIRT()
            if not output:
                local_out = os.path.join(source_dir, outname)
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(out_dir, outname)
            args.append((source, local_out, ref_path, mcflt))
            out.append(local_out)
        else:
            raise FileNotFoundError("Requested MOCO source file : " + str(source) + " does not exist.")
    with Pool() as p:
        res = p.starmap(_mcflt_wrapper, args)
    return out


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
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in input_dirs]
    outputs = []
    for source in sources:
        if os.path.isfile(source):
            source_dir = os.path.dirname(source)
            st = fsl.SliceTimer()
            if not output:
                out_dir = source_dir
                local_out = os.path.join(source_dir, 'slc.nii.gz')
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(source_dir, 'slc.nii.gz')
            path = os.path.join(source_dir, fname)
            args.append((path, local_out, slice_dir, TR, st))
            outputs.append(local_out)
        else:
            raise FileNotFoundError(source + " does not exist")
    with Pool() as p:
        res = p.starmap(_slice_time_wrapper, args)
    return outputs


def check_time_series_length(input_dirs: List[str], fname='f.nii.gz', expected_length: int=279):
    """
    Make sure each loaded nifti has length equal to expected number of TRs
    Parameters
    ----------
    input_dirs
    fname
    expected_length

    Returns
    -------

    """
    good = []
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in input_dirs]
    for path in sources:
        file = nib.load(path)
        data = file.get_fdata()
        if data.shape[-1] == expected_length:
            good.append(path)
    return good


def sample_frames(SOURCE: List[str], num_samples, output=None, fname='f_nordic.nii') -> List[str]:
    """
    Returns a frame to in the middle of a session
    :param out: output path
    :param SOURCE: list of source run dirs
    :param fname: name of nifti in run dirs
    :return: path to output 3d image
    """
    run_dir_idxs = np.random.choice(len(SOURCE), num_samples)
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in SOURCE]
    out_paths = []
    for i, run_dir_idx in enumerate(run_dir_idxs):
        in_file = sources[run_dir_idx]
        run_dir = os.path.dirname(in_file)
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

    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in functional_input_dirs]
    out = []
    for in_file in sources:
        if not os.path.exists(in_file):
            print('Failed to find requested file')
            exit()
        source_dir = os.path.dirname(in_file)
        out_file = os.path.join(source_dir, out_name)
        args.append((in_file, out_file, period, False, True))
        out.append(out_file)
    with Pool() as p:
        p.starmap(bandpass_filter_functional, args)
    return out


def linear_affine_registration(functional_input_dirs: List[str], template_file: str, fname: str = 'stripped.nii.gz', output: str = None, dof=12):
    flt = fsl.FLIRT()
    args = []
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in functional_input_dirs]
    out = []
    for in_file in sources:
        fname = os.path.basename(in_file)
        source_dir = os.path.dirname(in_file)
        chosen_name = fname.split('.')[0]
        if not output:
            local_out = os.path.join(source_dir, chosen_name + '_flirt.nii.gz')
            mat_out = os.path.join(source_dir, chosen_name + '_flirt.mat')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
            local_out = os.path.join(out_dir, chosen_name + '_flirt.nii.gz')
            mat_out = os.path.join(out_dir, chosen_name + '_flirt.mat')
        out.append(local_out)
        args.append((in_file, local_out, mat_out, template_file, flt, dof))
    with Pool() as p:
        p.starmap(_flirt_wrapper, args)
    return out


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
              across_modalities=False, outPref='antsReg', nonlinear=True,
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
          " --convergence [500,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox"
    if nonlinear:
        cmd = cmd + " --transform SyN[ .1,4,0 ] --metric CC[ " + fixedP + "," + movingP + ",1,6 ]" \
          " --convergence [200,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox" \
          " --transform SyN[ .01,3,0 ] --metric CC[ " + fixedP + "," + movingP + ",1,5 ]" \
          " --convergence [200,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox" \
          " --transform SyN[ .01,3,0 ] --metric CC[ " + fixedP + "," + movingP + ",1,3 ]" \
          " --convergence [200,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox" \
          " --transform SyN[ .001,3,0 ] --metric CC[ " + fixedP + "," + movingP + ",1,3 ]" \
          " --convergence [100,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox"
    print(cmd)
    subprocess.call(cmd, shell=True)


def antsCoReg(fixedP, movingP, outP, ltrns=('Affine', 'SyN'), n_jobs=2, full=False):
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
    if 'Syn' in ltrns:
        nonlinear = True
    else:
        nonlinear = False
    antsCoreg(fixedP,
        movingP,
        outP,
        initialTrsnfrmP=None, # we are working on resampled img
        across_modalities=True,
        outPref='antsReg',
        nonlinear=nonlinear,
        run=True, n_jobs=n_jobs, full=full)
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
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in functional_input_dirs]
    for in_file in sources:
        source_dir = os.path.dirname(in_file)
        fname = os.path.basename(in_file)
        if not os.path.isdir(source_dir):
            print("Failure", sys.stderr)
        transform, reg_nii = _itkSnapManual(os.path.abspath(template_file),
                                            os.path.abspath(in_file),
                                            os.path.abspath(source_dir))
        return transform, reg_nii


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


def create_low_res_anatomical(source_dir: str, fname:str ='orig.mgz', output=None, factor=[2, 2, 2], affine_scale=None, resample='interpolate'):
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
        subprocess.run(cmd_affine_set, shell=True)
    return output


def functional_to_cube(input_dirs: List[str], output: str = None, fname: str = 'moco.nii'):
    args = []
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in input_dirs]
    out = []
    for source in sources:
        source_dir = os.path.dirname(source)
        if not os.path.exists(source):
            print("could not find file")
            exit(1)
        if not output:
            local_out = os.path.join(source_dir, 'f_cubed.nii')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
            local_out = os.path.join(out_dir, 'f_cubed.nii')
        args.append((source, _pad_to_cube, local_out, False))
        out.append(local_out)
    with Pool() as p:
        p.starmap(_nifti_load_and_call_wrapper, args)
    return out


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
    out = []
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in input_dirs]
    for in_file in sources:
        source_dir = os.path.dirname(in_file)
        if not output:
            local_out = os.path.join(source_dir, 'stripped.nii.gz')
        else:
            local_out = os.path.join(source_dir, output)
        out.append(local_out)
        args.append((in_file, local_out, is_time_series, bet, fractional_thresh, remove_eyes))
    with Pool() as p:
        p.starmap(_bet_wrapper, args)
    return out
