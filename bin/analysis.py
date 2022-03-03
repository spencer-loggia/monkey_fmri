import math

import os
import scipy.ndimage
import shutil
from typing import List, Tuple
import subprocess

import numpy as np
import nibabel as nib

from scipy.stats import gamma
from scipy.stats import norm

from scipy.ndimage import gaussian_filter, binary_fill_holes, label

from multiprocessing import Pool

import preprocess
from preprocess import _pad_to_cube

from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.measure import regionprops

try:
    from sklearnex import patch_sklearn
    from sklearnex.linear_model import LinearRegression

    patch_sklearn()
except ModuleNotFoundError:
    print("Intel(R) Hardware Acceleration is not enabled. ")
    from sklearn.linear_model import LinearRegression

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm

from typing import List, Union, Tuple, Dict

import torch


def _norm_4d(arr: np.ndarray):
    """
    Simple normalization function, centers on mean, 1 unit equal one std dev of original data.
    :param arr: input array
    :return: normalized array
    """
    # mean = np.mean(arr.flatten())
    # arr -= mean
    norm_factor = np.std(arr.flatten())
    arr /= norm_factor
    return arr


def _set_data_range(arr, min_range, max_range):
    """
    expands a data set to encompass the given range from min to max.
    :param arr:
    :param min_range:
    :param max_range:
    :return:
    """
    arr += min(arr.flatten())  # min 0
    arr /= max(arr.flatten())  # 0 - 1
    range_size = max_range - min_range
    arr *= range_size
    arr += min_range
    return arr


def _create_dir_if_needed(base: str, name: str):
    out_dir = os.path.join(base, name)
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    return out_dir


def ts_smooth(input_ts: np.ndarray, kernel_std=1., temporal_smoothing=True):
    """
    Smooth the functional data
    :param input_ts: original time series data
    :param kernel_std: std dev of the gaussian kernel
    :param temporal_smoothing: Bool, defualt: True. Keep in mind using temporal may effect hemodynamic estimate negatively
    :return: timeseries of the same shape as input
    """

    def _smooth(input_arr, std):
        output = gaussian_filter(input_arr, sigma=std)
        return output

    if temporal_smoothing:
        smoothed = _smooth(input_ts, kernel_std)
    else:
        smoothed = np.apply_along_axis(lambda m: _smooth(m, kernel_std), axis=3, arr=input_ts)
    return smoothed


def rsa():
    raise NotImplementedError


def hyperalign():
    raise NotImplementedError


def average_functional_data(run_dirs, output, fname='normalized.nii', through_time=False):
    """
    Averages a set of epis producing a single 4D time series. If through_time is True, also averages through the time
    dimmensiong, producing a single 3D volume.
    :param run_dirs:
    :param output:
    :param fname:
    :param through_time:
    :return:
    """
    avg_func = None
    count = 0
    for source_dir in run_dirs:
        req_path = os.path.join(source_dir, fname)
        if os.path.isfile(req_path):
            brain = nib.load(req_path)
            brain_tensor = np.array(brain.get_fdata())
            if through_time:
                brain_tensor = np.mean(brain_tensor, axis=-1)
            if avg_func is None:
                avg_func = brain_tensor
            else:
                avg_func += brain_tensor
            count += 1
    avg_func /= count
    avg_nii = nib.Nifti1Image(avg_func, affine=brain.affine, header=brain.header)
    nib.save(avg_nii, output)
    return avg_func


def _mion_base(tr, time_length, oversampling=16, onset=0.0):
    """Implementation of the MION response function model.
    Parameters
    ----------
    tr: float
        scan repeat time, in seconds
    oversampling: int, optional
        temporal oversampling factor
    onset: float, optional
        hrf onset time, in seconds

    Returns
    -------
    response_function: array of shape(length / tr * oversampling, dtype=float)
        response_function sampling on the oversampled time grid
    """
    dt = tr / oversampling
    time_stamps = np.linspace(
        0, time_length, np.rint(float(time_length) / dt).astype(int)
    )
    time_stamps -= onset

    # parameters of the gamma function
    delay = 1.55
    dispersion = 5.5

    response_function = gamma.pdf(time_stamps, delay, loc=0, scale=dispersion)
    response_function /= response_function.sum()

    return response_function


def mion_response_function(tr, time_length, oversampling=16.0):
    """Implementation of the MION time derivative response function model.
    Parameters
    ----------
    tr: float
        scan repeat time, in seconds
    oversampling: int, optional
        temporal oversampling factor, optional

    Returns
    -------
    drf: array of shape(time_length / tr * oversampling, dtype=float)
        derived_response_function sampling on the provided grid
    """
    do = 0.1
    drf = (
                  _mion_base(tr, time_length, oversampling)
                  - _mion_base(tr, time_length, oversampling, do)
          ) / do
    target_size = time_length / tr
    factor = target_size / drf.shape[0]
    drf = scipy.ndimage.zoom(drf, factor, order=3)
    drf = drf / np.abs(np.sum(drf))  # integral 0x = 1
    drf = torch.from_numpy(drf).reshape([1, 1, -1]).float()
    return drf


def bold_response_function(tr=3, time_length=30):
    """
    Just gives an inverted mion function, this is fine because response function optimization will discover the true
    bold function. Should put a real bold function here though in case someone runs max_dynamic with autoconv off
    :param tr:
    :param time_length:
    :return:
    """
    return -1 * mion_response_function(tr, time_length, oversampling=16.0)


def maximal_dynamic_convolution_regressor(design_matrix: np.ndarray, gt: np.ndarray, conv_size=11, conv_pad=5,
                                          epochs=30, pid=0, mion=True, auto_conv=True, tr=3):
    """
    Hi so this is the meat of the algorithm for finding beta coeficients, e.g. first level MRI analysis.
    The beta coefficients can be thought of as our best guess at the magnitude of the response of a given voxel
    to a specific stimulus condition.
    esentially, we use a standard glm model to solve the equation :math:`[D \ast H] \Beta = V` where D is the
    design matrix (i.e. a onehot encoding of blocks presented over time, and k (conditions) x time (t) matrix)
    and V is the epi image (time x voxels) and H is a discrete convolution over Ds time axis.
    The purpose of H is to adjust for the temporal delay in mri signal by adjusting the design matirx D, allowing
    for a simple linear mapping to V via :math: `\Beta`.
    The MLE solution for beta can be found trivially via :math: `(D' D)^{-1} D' V`
    We start with an empirical definition of H, and then refine it via gradient decent. (if auto_conv is True)
    The overall optimal solution is found by alternating between fixing Beta and optimizing over H, and fixing H and
    optimizing Beta for a set number of iterations. (a case of the EM algorithm)
    :param design_matrix:
    :param gt:
    :param conv_size:
    :param conv_pad:
    :param epochs:
    :param pid:
    :param mion:
    :param auto_conv:
    :param tr:
    :return:
    """
    og_shape = gt.shape
    conv1d = torch.nn.Conv1d(kernel_size=conv_size, in_channels=1, out_channels=1, padding=conv_pad, bias=False)
    deconv1d = torch.nn.Conv1d(kernel_size=conv_size, in_channels=1, out_channels=1, padding=conv_pad, bias=False)
    deconv_weight = torch.ones((1, 1, conv_size)) / conv_size
    if mion:
        def_hrf = mion_response_function(tr, conv_size * tr)
        deconv_weight = -1 * deconv_weight
    else:
        def_hrf = bold_response_function(tr, time_length=conv_size * tr)
    if auto_conv:
        weight = def_hrf.clone()
    else:
        weight = def_hrf.clone()
        epochs = 1

    conv1d.weight = torch.nn.Parameter(weight, requires_grad=True)
    deconv1d.weight = torch.nn.Parameter(deconv_weight, requires_grad=True)
    design_matrix = torch.from_numpy(design_matrix)[:, None, :].float()  # design matrix should be (k, t) so that k conditions is treated as batch by torch convention
    if np.ndim(gt) > 1:
        gt = gt.reshape(-1, design_matrix.shape[0]).T  # reshape to all time, LL VOXELS
    gt = torch.from_numpy(gt).float()
    # linear = torch.nn.Linear(in_features=design_matrix.shape[-1], out_features=gt.shape[1], bias=False)
    beta = torch.normal(mean=.5, std=.2, size=[design_matrix.shape[-1], gt.shape[1]], requires_grad=False).float()
    optim = torch.optim.SGD(lr=.0001, params=list(conv1d.parameters()))
    optim_deconv = torch.optim.SGD(lr=.01, params=list(deconv1d.parameters()))
    sched = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=15, gamma=.1)
    deconv_sched = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=7, gamma=.1)
    mseloss = torch.nn.MSELoss()
    loss_deconv = torch.nn.MSELoss()
    # mseloss = _ssqrt_lfxn
    print('Starting optimization routine... (pid=', pid, ')')

    def pred_seq(dmat):
        # optim_deconv.zero_grad()
        xexp = conv1d(dmat.T).T
        if xexp.shape != dmat.shape:
            raise ValueError("Convolution result must have same dimension as design matrix (" +
                             str(dmat.shape) + ") , not " + str(xexp.shape) +
                             ". Adjust conv size and padding size.")
        # deconv_loss = mse_loss_deconv(dmat_exp, dmat)
        # deconv_loss.backward()
        # optim_deconv.step()
        # deconv_sched.step()
        xexp = xexp.squeeze()  # remove channel dim for conv
        y_hat = xexp @ beta  # time is now treated as the batch, we do (t x k) * (k x v) -> (t x v) where v is all voxels
        return xexp, y_hat

    loss_hist = []
    for i in range(epochs):
        # expectation
        optim.zero_grad()
        optim_deconv.zero_grad()
        x_expect, y_hat = pred_seq(design_matrix)
        # analytical maximization
        with torch.no_grad():
            beta = torch.inverse(x_expect.T @ x_expect) @ x_expect.T @ gt
        y_hat = x_expect @ beta
        flex_loss = mseloss(gt.T, y_hat.T)
        loss = flex_loss
        loss_hist.append(torch.log2(loss).detach().clone())
        if auto_conv:
            loss.backward()
            optim.step()
            sched.step()
        optim.zero_grad()
        optim_deconv.zero_grad()
        # deconvolution estimate update
        with torch.no_grad():
            xexp = conv1d(design_matrix.T).T
        dmat_exp = deconv1d(xexp.T).T
        d_loss = loss_deconv(dmat_exp, design_matrix)
        d_loss.backward()
        optim_deconv.step()
        deconv_sched.step()
        print("optim epoch #", i, "(pid=", pid, ")", ' loss=', flex_loss.detach().item(), 'deconvolve loss=', d_loss.detach().item())

    beta = np.array(beta).T
    beta = beta.reshape(list(og_shape[:-1]) + [design_matrix.shape[-1]])
    hemodynamic_est = np.array(conv1d.weight.clone().detach()).reshape(np.prod(conv1d.weight.shape))
    deconv = np.array(deconv1d.weight.clone().detach()).reshape(np.prod(deconv1d.weight.shape))
    return beta, hemodynamic_est, loss_hist, deconv


def contrast(beta_coef: np.ndarray, contrast_matrix: np.ndarray, pid=0) -> np.ndarray:
    """
     just a wrapper for matrix multiplication so I can put some expected dimensions :)
    :param beta_coef: a (w, h, d, k) matrix, where k is conditions
    :param contrast_matrix: a (k, m) matrix, where m is the number of contrasts to preform
    :return: resulting contrast, a (w, h, d, m) matrix
    """
    print(beta_coef.shape, 'x', contrast_matrix.shape, "(pid=", pid, ")")
    contrasts = beta_coef @ contrast_matrix
    print('=', contrasts.shape, "(pid=", pid, ")")
    return contrasts


def _pipe(d_mat, run, c_mat, mode, mion, auto_conv, tr, pid):
    """
    Single thread worker dispatch function.
    :param d_mat: design matrix from fsfast or local
    :param run: preprocessed functional data
    :param c_mat: contrast test definition matrix
    :return:
    """
    beta, res, loss_hist, deconv = maximal_dynamic_convolution_regressor(d_mat, run, pid=pid, mion=mion, auto_conv=auto_conv,
                                                                 tr=tr)
    contrasts = contrast(beta, c_mat, pid=pid)
    return contrasts, res, loss_hist, deconv


def _create_SNR_volume_wrapper(input_dir, noise_path, out_path, type='tSNR'):
    func = nib.load(input_dir)
    func_data = func.get_fdata()
    func_affine = func.affine
    noise = nib.load(noise_path)
    noise_data = noise.get_fdata()
    if type == 'tSNR':
        sd = np.std(noise_data, axis=3)
        sd[np.where(sd == 0)] = 1
        mean = np.mean(func_data, axis=3)
        res = mean / sd
    elif type == 'CNR':
        sdn = np.std(noise_data, axis=3)
        sdn[np.where(sdn == 0)] = 1
        sdf = np.std(func_data, axis=3)
        sdf[np.where(sdf == 0)] = 1
        res = sdf / sdn
    res_nii = nib.Nifti1Image(res, func_affine)
    nib.save(res_nii, os.path.join(out_path, '{}.nii.gz'.format(type)))


def create_SNR_map(input_dirs: List[str], noise_dir, output: Union[None, str] = None, fname='clean.nii.gz',
                   type='tSNR'):
    '''
    create_SNR_map: Creates a tSND or a CNR map.
    :param input_dirs:
    :param output:
    :param fname:
    :param type:
    :return:
    '''
    args = []
    noise_path = os.path.join(noise_dir, 'noise.nii.gz')
    for source_dir in input_dirs:
        if os.path.isfile(os.path.join(source_dir, fname)):
            input_dir = os.path.join(source_dir, fname)
            if not output:
                out_dir = source_dir
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))

        args.append((input_dir, noise_path, out_dir, type))
    with Pool() as p:
        res = p.starmap(_create_SNR_volume_wrapper, args)


def design_matrix_from_order_def(block_length: int, num_blocks: int, num_conditions: int, order: List[int]):
    """
    Creates as num_conditions x time onehot encoded matrix from an order number definition
    :param block_length:
    :param num_blocks:
    :param num_conditions:
    :param order:
    :param convolve:
    :param aq_mode:
    :return:
    """
    k = len(order)
    design_matrix = np.zeros((0, num_conditions))
    for i in range(num_blocks):
        block = np.zeros((block_length, num_conditions), dtype=float)
        active_condition = order[i % k]
        block[:, active_condition] = 1
        design_matrix = np.concatenate([design_matrix, block], axis=0)
    return design_matrix


def intra_subject_contrast(run_dirs: List[str], design_matrices: List[np.ndarray], contrast_matrix: np.ndarray,
                           contrast_descriptors: List[str], output_dir: str, fname: str = 'registered.nii.gz',
                           mode='standard', mion=False, use_python_mp=False, auto_conv=False, tr=3):
    """
    Compute each desired contrast on the functional data in parallel.
    :param mion:
    :param design_matrix: list of array of size conditions x num trs. length 1 if stim order same for all imas,
                          len equal to number of runs otherwise. order corresponds to ima order in run dirs.
    :param mode:
    :param run_dirs: locations of the functional runs
    :param contrast_matrix: The contrast test definitions.
    :param contrast_descriptors: Description of the contrast test specified by each row of the matrix.
    :param output_dir: where to save
    :param fname: name of the preprocessed functional file to use (must exist in each of the run dirs)
    :return:
    """
    run_stack = []
    affines = []
    header = None
    for design_matrix in design_matrices:
        if contrast_matrix.shape[0] != design_matrix.shape[1]:
            raise ValueError("Contrast matrix must have number of rows equal to the number of stimulus conditions.")
        if len(contrast_descriptors) != contrast_matrix.shape[1]:
            raise ValueError("Number of Contrast Descriptors must match number of cols in contrast matrix")

    for i, source_dir in enumerate(run_dirs):
        req_path = os.path.join(source_dir, fname)
        if os.path.isfile(req_path):
            brain = nib.load(req_path)
            brain_tensor = np.array(brain.get_fdata())
            exp_num_frames = design_matrices[i].shape[0]
            if brain_tensor.shape[-1] != exp_num_frames:
                print(
                    "WARNIING: Loaded functional data (" + source_dir + ") must have number of frames equal to length of "
                                                                        "active_condition map, not " + str(
                        brain_tensor.shape[-1]) + ' and ' + str(exp_num_frames))
                continue
            run_stack.append(brain_tensor[None, :, :, :, :])
            affines.append(np.array(brain.affine)[None, :, :])
            header = brain.header
        else:
            raise FileNotFoundError("Couldn't find " + fname + " in specified source dir " + source_dir)
    full_run_params = list(zip(design_matrices,
                               run_stack,
                               [contrast_matrix] * len(run_stack),
                               [mode] * len(run_stack),
                               [mion] * len(run_stack),
                               [auto_conv] * len(run_stack),
                               [tr] * len(run_stack),
                               list(range(len(run_stack)))))
    if use_python_mp:
        with Pool() as p:
            # dispatch workers, by default will claim all but one cpu core.
            res = p.starmap(_pipe, full_run_params)
    else:
        res = []
        for i in range(len(full_run_params)):
            res.append(_pipe(*full_run_params[i]))
    contrasts, hemo, loss_hists, deconvs = list(zip(*res))
    avg_contrasts = np.sum(np.stack(contrasts, axis=0), axis=0)[0]

    print(avg_contrasts.shape)
    affine = np.concatenate(affines, axis=0).mean(axis=0)
    for i, cond in enumerate(np.transpose(avg_contrasts, (3, 0, 1, 2))):
        contrast_nii = nib.Nifti1Image(cond, affine=affine, header=header)
        nib.save(contrast_nii, os.path.join(output_dir, contrast_descriptors[i] + '_contrast.nii'))

    fig, axs = plt.subplots(3)
    for i, h in enumerate(hemo):
        axs[0].plot(h, label=i)
    axs[0].legend()
    avg_hemo = np.stack(hemo, axis=0).mean(axis=0)
    avg_deconv = np.stack(deconvs, axis=0).mean(axis=0)
    axs[1].plot(avg_deconv)
    hemo_path = os.path.join(output_dir, 'hrf_est.npz')
    deconv_path = os.path.join(output_dir, 'deconvolution_est.npz')
    np.save(hemo_path, avg_hemo)
    np.save(deconv_path, avg_deconv)
    if None not in loss_hists:
        avg_loss_hist = np.sum(np.array(loss_hists), axis=0)
        axs[2].plot(avg_loss_hist)
    fig.show()
    plt.show()
    return avg_contrasts, hemo_path, deconv_path


def _find_scale_factor(high_res: np.ndarray, low_res: np.ndarray):
    """
    Computes the amount to scale one matrix to match the dims of another
    :param high_res:
    :param low_res:
    :return:
    """
    h_s = high_res.shape
    l_s = low_res.shape
    prod = np.array(h_s) / np.array(l_s)
    if all(prod == prod[0]):
        return prod[0]
    else:
        raise ValueError('dimension should have been scaled evenly here')


def _scale_and_standardize(scale, path):
    subdivide_arg = str(1 / scale)
    subprocess.run(['mri_convert', path, '-vs', subdivide_arg, subdivide_arg, subdivide_arg, path, '--out_type', 'nii'])
    subprocess.run(['mri_convert', path, '-iis', '1', '-ijs', '1', '-iks', '1', path, '--out_type', 'nii'])
    return path


def create_contrast_surface(anatomical_white_surface: str,
                            contrast_vol_path: str,
                            orig_low_res_anatomical: str,
                            orig_high_res_anatomical: str,
                            hemi: str,
                            subject_id='castor_test',
                            output=None):
    """
    Takes a contrast volume, resizes it to the appropriate dimensions, corrects the affine,
    and projects it onto an anatomical surface

    :param output: default None. If None places files in the same directory as contrast volume input. Otherwise
                   interpreted as path to directory to place output files.
    :param anatomical_white_surface: path to the white matter surface created from the original hig res anatomical
    :param contrast_vol_path: path to the contrast volume file
    :param orig_low_res_anatomical: The original donwnsampled (low res) anatomical that functional data
                                    (and thus contrasts) were registered to. Should be able to use the skull stripped
                                    version or the full version.
    :param orig_high_res_anatomical: The original high resolution anatomical scan image
    :param hemi: The hemisphere to process. (make sure surface file matches)
    :param subject_id: name of subject (project folder)
    """
    if hemi not in ['rh', 'lh']:
        raise ValueError('Hemi must be one of rh or lh.')
    high_res = nib.load(orig_high_res_anatomical)
    high_res_data = high_res.get_fdata()
    high_res_data = _pad_to_cube(high_res_data)
    low_res = nib.load(orig_low_res_anatomical).get_fdata()
    low_res = _pad_to_cube(low_res)
    scale = _find_scale_factor(high_res_data, low_res)
    contrast_nii = nib.load(contrast_vol_path)
    contrast_data = _norm_4d(np.array(contrast_nii.get_fdata()))
    aff = contrast_nii.affine
    new_nii = nib.Nifti1Image(contrast_data, affine=aff, header=contrast_nii.header)

    if not output:
        output = os.path.dirname(contrast_vol_path)
    out_desc = os.path.basename(contrast_vol_path).split('.')[0]
    path = os.path.join(output, 'tmp.nii')
    nib.save(new_nii, path)

    os.environ.setdefault("SUBJECTS_DIR", os.path.abspath(os.environ.get('FMRI_WORK_DIR')))

    # subprocess.run(['fslroi', path, path_gz, '0', '256', '0', '256', '0,', '256'])
    path = _scale_and_standardize(scale, path)
    overlay_out_path = os.path.join(output, 'sigsurface_' + hemi + '_' + out_desc + '.mgh')
    subprocess.run(['mri_vol2surf', '--src', path,
                    '--out', overlay_out_path,
                    '--hemi', hemi,
                    '--regheader', subject_id])


def labels_to_roi_mask(label_dir, hemi, out_dir, t1, subject_id) -> Tuple[str, list]:
    """
    creates a binary mask in volume space for each label.
    :param label_dir:
    :param hemi:
    :param out_dir:
    :param subject_id:
    :return:
    """
    os.environ.setdefault("SUBJECTS_DIR", os.path.abspath(os.environ.get('FMRI_WORK_DIR')))
    if hemi not in ['rh', 'lh']:
        raise ValueError('Hemi must be one of rh or lh.')
    for f in os.listdir(label_dir):
        output = os.path.join(out_dir, f.split('.')[0] + '.nii')
        if '.label' in f and hemi in f:
            subprocess.run(['mri_label2vol',
                            '--label', os.path.join(label_dir, f),
                            '--regheader', t1,
                            '--temp', t1,
                            '--proj', 'frac', '0', '1', '.1',
                            '--hemi', hemi,
                            '--subject', subject_id,
                            '--o', output])
    return out_dir


def create_contrast_overlay_image(contrast_data, sig_thresh, saturation):
    """
    Turns a single contrast volume into volume with color channels on a 4th dimmension.
    Positive contrsat in red (channel dim 0) negative in blue (channel_dim 1)
    :param contrast_data:
    :param sig_thresh:
    :param saturation:
    :return:
    """
    contrast_data = _norm_4d(contrast_data)
    contrast_data[np.abs(contrast_data) < sig_thresh] = 0
    pos_contrast = np.zeros_like(contrast_data)
    neg_contrast = np.zeros_like(contrast_data)
    pos_contrast[contrast_data > 0] = contrast_data[contrast_data > 0]
    neg_contrast[contrast_data < 0] = contrast_data[contrast_data < 0]
    pos_contrast[pos_contrast > saturation] = saturation
    neg_contrast[neg_contrast < -1 * saturation] = -1 * saturation
    pos_contrast = np.abs(pos_contrast)
    pos_contrast[pos_contrast > 0] += sig_thresh
    neg_contrast = np.abs(neg_contrast)
    neg_contrast[neg_contrast > 0] += sig_thresh
    mid_contrast = np.copy(neg_contrast) / 1.5
    contrast_img = np.stack([pos_contrast, mid_contrast, neg_contrast], axis=3)
    return contrast_img


def create_slice_maps(function_reg_vol, anatomical, reg_contrast, sig_thresh=10, saturation=15):
    """
    :param saturation:
    :param sig_thresh:
    :param function_reg_vol: (w, h, d)
    :param anatomical:  (w, h d)
    :param reg_contrast: (w, h, d)
    :return: len d List[w, h, c]  negative contrast on chan 2 (B) positive on chan 0 (R)
    """
    f_arr = nib.load(function_reg_vol).get_fdata()
    a_arr = nib.load(anatomical).get_fdata()
    contrast_data = nib.load(reg_contrast).get_fdata()
    contrast_img = create_contrast_overlay_image(contrast_data, sig_thresh, saturation)
    contrast_img = _set_data_range(contrast_img, 0, 255)
    f_arr = np.stack([f_arr, f_arr, f_arr], axis=3)
    std_f = np.std(f_arr.flatten())
    a_arr = np.stack([a_arr, a_arr, a_arr], axis=3)
    f_arr[f_arr > (4 * std_f)] = 4 * std_f
    f_arr = _set_data_range(f_arr, 0, 255)
    a_arr = _set_data_range(a_arr, 0, 255)
    con_idxs = contrast_img != 0
    f_arr[con_idxs] = contrast_img[con_idxs]
    a_arr[con_idxs] = contrast_img[con_idxs]
    f_arr = np.transpose(f_arr, axes=[1, 0, 2, 3])
    a_arr = np.transpose(a_arr, axes=[1, 0, 2, 3])

    # slice
    slices = [(f_arr[:, :, i, :].squeeze().astype(int), a_arr[:, :, i, :].squeeze().astype(int))
              for i in range(18, f_arr.shape[2] - 10)]
    fig, axs = plt.subplots(len(slices), 2)
    fig.set_size_inches(8, (2 * len(slices)))
    name = os.path.basename(reg_contrast).split('.')[0]
    fig.suptitle(name + " Overlayed On Registered Epi (Left) and DS T1 (right) \n "
                 "Coronal Slices (Posterior to Anterior) \n"
                 "Threshold: " + str(sig_thresh) + " std dev, Saturation: " + str(saturation) + " std dev", y=.999)
    fig.tight_layout(pad=2)
    for i, slice_tup in enumerate(slices):
        f_slice = slice_tup[0]
        a_slice = slice_tup[1]
        axs[i, 0].set_title("slice " + str(i + 18) + ' / ' + str(len(slices) + 28))
        axs[i, 0].imshow(f_slice)
        axs[i, 1].imshow(a_slice)
    out_path = os.path.join(os.path.dirname(function_reg_vol), name + '_slice_comparison.jpg')
    fig.savefig(out_path)
    return out_path


def _fit_model(model: GaussianMixture, data):
    assignments = model.fit_predict(data)
    bic = model.bic(data)
    return assignments, bic


def _segment_contrast_image(contrast_data, threshold=5., negative=False, size_thresh=7):
    # smooth over
    smoothed = ts_smooth(contrast_data, temporal_smoothing=True, kernel_std=.6)
    segs = np.zeros_like(smoothed)
    if negative:
        segs[smoothed < threshold] = 1
        desc = 'negative'
    else:
        segs[smoothed > threshold] = 1
        desc = 'positive'
    labels, num_rois = label(segs)
    sample_rp = regionprops(labels)

    # remove too small rois
    invalid_labels = {label_id + 1 for label_id in range(0, num_rois) if sample_rp[label_id].area <= size_thresh}
    valid_labels = set(range(1, num_rois)) - invalid_labels
    for label_id in invalid_labels:
        labels[labels == label_id] = 0

    # reindex labels
    id = 1
    for label_id in sorted(list(valid_labels)):
        if label_id != id:
            labels[labels == label_id] = id
        id += 1

    print(num_rois, "potential", desc, "rois detected")
    return labels, len(valid_labels) + 1


def _plot_3d_scatter(rois, label_id, brain_box, ax, color, size_tresh=0):
    """
    plot a brain in 3d with colored rois in pyplot
    """
    clust_idx_tuples = np.array(np.nonzero(rois == label_id))
    if clust_idx_tuples.shape[1] <= size_tresh:
        return ax
    clust_idx_tuples = clust_idx_tuples[:,
                       np.random.choice(clust_idx_tuples.shape[1], int(clust_idx_tuples.shape[1] / 10) + 1,
                                        replace=False)]
    ax.scatter(clust_idx_tuples[0], clust_idx_tuples[1], clust_idx_tuples[2], c=color, alpha=.15)
    ax.scatter(brain_box[0], brain_box[1], brain_box[2], c='gray', alpha=.5, s=.05)
    label_point = clust_idx_tuples.mean(axis=1).flatten()
    ax.text(label_point[0], label_point[1], label_point[2], '%s' % (str(label_id)), size=20, zorder=0,
            color=color)
    return ax


def _get_roi_time_course(rois, label_id, fdata, ax, block_length, block_order, colors, size_thresh=0, roi_name=None, ts_name=''):
    """

    :param num_rois:
    :param rois:
    :param fdata: a created functional average
    :param threshold: significance threshold to consider voxels
    :param cluster_min: minimum number of voxels
    :param cluster_max: maximum number of voxels
    :return:
    """

    clust_idxs = np.nonzero(rois == label_id)
    if len(clust_idxs[0]) <= size_thresh:
        return ax
    clust_ts_vox = fdata[clust_idxs]
    spatial_flattened = clust_ts_vox.reshape(-1, clust_ts_vox.shape[-1])
    mean_ts = np.mean(spatial_flattened, axis=0)
    trs = mean_ts.shape[0]
    ax.plot(mean_ts, label=ts_name)
    if roi_name is None:
        name = str(label_id)
    ax.set_title("ROI " + roi_name)
    ax.xaxis.set_ticks(np.arange(0, trs, block_length))
    float_map = 1 / len(np.unique(block_order))
    color_map = [0] * len(np.unique(block_order))
    for i, tr in enumerate(range(0, trs, block_length)):
        condition = int(block_order[(i % len(block_order))])
        c_pos = float(condition) * float_map
        c = colors(c_pos)
        color_map[condition] = c_pos
        ax.axvspan(tr, tr + block_length, color=c, alpha=0.3)
    return mean_ts, ax, color_map


def get_condition_time_series_comparision(functional_dirs, block_length, ima_order_num_map: Dict[str, int],
                                          order_num_defs: Dict[str, List[int]], target_condition, output,
                                          fname='epi_masked.nii', pre_onset_trs=6, post_offset_trs=18):
    """
    Goal is to take a set of epis that were recorded with different stimuli presentation order (needs to be a block
    cesign) and create a new functional file that compares the waveform of the functional blog with an average over
    other activity, while still capturing some of the temporal dynamics.
    :param functional_dirs: list of paths to ima dirs, each containing the corresponding epi
    :param block_length: length of target condition block
    :param ima_order_num_map: A dictionary mapping from ima numbers (as strings) to order numbers (as int)
    :param order_num_defs: A dictionary mapping from order numbers (as strings) to block orders (lists of condition integers)
    :param target_condition: The condition integer identifier who's blocks will be aligned and stacked
    :param output: path to save new time series nifti
    :param fname: the name of epi files to read in the ima directories
    :param pre_onset_trs: number of trs to include before block onset
    :param post_offset_trs: number of trs to include after block offset
    :return:
    """
    total_length = pre_onset_trs + block_length + post_offset_trs
    w, h, d, _ = nib.load(os.path.join(functional_dirs[0], fname)).get_fdata().shape
    corrected_arr = np.zeros((w, h, d, total_length))
    avg_count = 0
    for func_dir in functional_dirs:
        ima = os.path.basename(func_dir)
        order_num = ima_order_num_map[ima]
        cond_seq = order_num_defs[str(order_num)]
        cond_idxs = np.array([i for i in range(len(cond_seq)) if cond_seq[i] == target_condition])
        onset_trs = cond_idxs * block_length
        data_nii = nib.load(os.path.join(func_dir, fname))
        data = data_nii.get_fdata()
        padded_data = np.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (pre_onset_trs, post_offset_trs)), mode='mean')
        # start ad stop below are adjusted to account for padding
        start_trs = onset_trs
        stop_trs = pre_onset_trs + onset_trs + block_length + post_offset_trs
        for i in range(start_trs.shape[0]):
            avg_count += 1
            start = start_trs[i]
            stop = stop_trs[i]
            local_data = padded_data[:, :, :, start:stop]
            corrected_arr += local_data
    corrected_arr /= avg_count
    order_compare_nifti = nib.Nifti1Image(corrected_arr, affine=data_nii.affine, header=data_nii.header)
    nib.save(order_compare_nifti, output)
    return output


def segment_get_time_course(contrast_file: str, functional_file: str, block_length, block_order):
    c_nii = nib.load(contrast_file)
    c_data = np.array(c_nii.get_fdata())
    c_data = _norm_4d(c_data)
    assert np.isclose(np.std(c_data), 1, atol=1e-8)
    f_data = np.array(nib.load(functional_file).get_fdata())
    pos_seg, num_pos_rois = _segment_contrast_image(c_data, threshold=4, negative=False)
    neg_seg, num_neg_rois = _segment_contrast_image(c_data, threshold=-8, negative=True)
    mask = np.zeros_like(pos_seg)
    mask[pos_seg > 0] = 1
    mask[neg_seg > 0] += 1
    pos_seg[mask == 2] = -1
    neg_seg[mask == 2] = -1
    brain_area = np.array(np.nonzero(f_data[:, :, :, 0] > np.mean(f_data[:, :, :, 0].flatten())))
    brain_area = brain_area[:, np.random.choice(brain_area.shape[1], 2000, replace=False)]
    fig_3d = plt.figure()
    ax3d = Axes3D(fig_3d)
    colors = plt.get_cmap('hsv')
    for is_negative in [True, False]:
        if is_negative:
            seg = neg_seg
            num_rois = num_neg_rois
            color = 'blue'
            desc = 'negative'
        else:
            seg = pos_seg
            num_rois = num_pos_rois
            color = 'red'
            desc = 'positive'
        fig_ts, axs_ts = plt.subplots(num_rois)
        fig_ts.suptitle(desc + ' condition time series')
        fig_ts.set_size_inches(8, 1.5 * num_rois)
        fig_ts.tight_layout()
        color_map = None
        for label_id in range(1, num_rois):
            _, _, color_map = _get_roi_time_course(seg, label_id, f_data, axs_ts[label_id - 1], block_length, block_order, colors)
            _plot_3d_scatter(seg, label_id, brain_area, ax3d, color)
        _plot_3d_scatter(seg, -1, brain_area, ax3d, 'black')
    if color_map:
        fig_ts.legend([plt.Line2D([0], [0], color=colors(i), lw=2, alpha=.6) for i in color_map],
                      ['condition ' + str(i) for i in range(len(color_map))])
    seg = pos_seg + neg_seg
    all_seg = np.nonzero(seg != 0)
    cleaned_contrast = np.zeros_like(c_data)
    cleaned_contrast[all_seg] = c_data[all_seg]
    con_nii = nib.Nifti1Image(cleaned_contrast, affine=c_nii.affine, header=c_nii.header)
    return con_nii


if __name__ == '__main__':
    # # main method used for quick and dirty testing, do not expect to functional properly depending on local directory
    # # structure / filenames
    root = '/Users/loggiasr/Projects/fmri/monkey_fmri/WoosterMerridianMGH4CHAN'
    functional_dir = os.path.join(root, 'functional/Wooster_20211112/')
    SOURCE = [os.path.join(functional_dir, f) for f in os.listdir(functional_dir) if f[0] != '.']
    # desired_contrast_mat = np.array([[0, -1, 1],
    #                                  [-1, .5, .5]]).T
    # contrast_desc = ['vertical_vs_horizontal', 'null_vs_avg_vert_horiz']
    # res = intra_subject_contrast(run_dirs=SOURCE[:4],
    #                              paradigm_file=os.path.join(root, 'stimuli/meridian_mapper_order1.para'),
    #                              contrast_matrix=desired_contrast_mat,
    #                              contrast_descriptors=contrast_desc,
    #                              output_dir=os.path.join(root, 'analysis_out'),
    #                              fname='registered.nii.gz',
    #                              mode='maximal_dynamic',
    #                              use_python_mp=True)
    create_contrast_surface('WoosterMerridianMGH4CHAN/surf/lh.white',
                            '../WoosterMerridianMGH4CHAN/analysis/reg_horizontal_minus_vertical_merridians_mk2_contrast.nii',
                            'WoosterMerridianMGH4CHAN/mri/ds_t1_masked.nii',
                            'WoosterMerridianMGH4CHAN/mri/orig.mgz',
                            hemi='lh', subject_id='WoosterMerridianMGH4CHAN')

    # average_functional_data(
    #     [os.path.join('/Users/loggiasr/Projects/fmri/monkey_fmri/castor_test/functional/20100131', f)
    #      for f in os.listdir('/Users/loggiasr/Projects/fmri/monkey_fmri/castor_test/functional/20100131')],
    #     '/Users/loggiasr/Projects/fmri/monkey_fmri/castor_test/functional/avg_func_og.nii', fname='registered.nii.gz')
    #
    # cleaned_contrast = segment_get_time_course('/Users/loggiasr/Projects/fmri/monkey_fmri/castor_test/analysis_out/condition_vertical_vs_horizontal_contrast.nii',
    #                                   '/Users/loggiasr/Projects/fmri/monkey_fmri/castor_test/functional/avg_func.nii',
    #                                   block_length=16,
    #                                   block_order=[1, 0, 2, 0])
    # nib.save(cleaned_contrast, '/Users/loggiasr/Projects/fmri/monkey_fmri/WoosterMerridianMGH4CHAN/analysis_out/condition_vertical_vs_horizontal_newmgh_CLEAN_contrast.nii')
