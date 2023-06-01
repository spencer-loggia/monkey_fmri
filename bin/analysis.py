import copy
import itertools
import math
import multiprocessing

if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method("spawn", force=True)

import random
import traceback

import os
import pickle

import scipy.ndimage
import shutil
from typing import List, Tuple
import subprocess

import numpy as np
import nibabel as nib

from scipy.stats import gamma, spearmanr
from scipy.spatial.distance import pdist
from scipy.stats import norm
from scipy.stats import norm

from scipy.ndimage import gaussian_filter, binary_fill_holes, label

from multiprocessing import Pool

from skimage.measure import regionprops
from sklearn.mixture import GaussianMixture

try:
    from sklearnex import patch_sklearn
    from sklearnex.linear_model import LinearRegression
    patch_sklearn()
except ModuleNotFoundError:
    print("Intel Hardware Acceleration is not enabled. ")
    from sklearn.linear_model import LinearRegression

import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm

from typing import List, Union, Tuple, Dict

import torch

from nilearn.glm import first_level
from nilearn.image import high_variance_confounds
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.glm.first_level.hemodynamic_models import mion_hrf, spm_time_derivative


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


def _confound(epi):
    return pd.DataFrame(high_variance_confounds(epi, percentile=2, n_confounds=2))


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
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in run_dirs]
    for req_path in sources:
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
    sources = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in input_dirs]
    for source in sources:
        if os.path.isfile(source):
            if not output:
                out_dir = os.path.dirname(source)
            else:
                out_dir = output
        else:
            raise FileNotFoundError(source + " Does Not Exist.")
        args.append((source, noise_path, out_dir, type))
    with Pool() as p:
        res = p.starmap(_create_SNR_volume_wrapper, args)


def _make_nl_dm(frames, time_df, hrf, stim_min_onset, delay_periods):
    svd_converge = False
    trys = 0
    while not svd_converge and trys < 4:
        try:
            dm = first_level.make_first_level_design_matrix(frames, time_df, hrf,
                                                            drift_model='cosine', high_pass=.01,
                                                            min_onset=stim_min_onset, fir_delays=delay_periods)
        except (ValueError, np.linalg.LinAlgError):
            trys += 1
            continue
        svd_converge = True
    if not svd_converge:
        print("DM creation failed")
        exit(2)
    return dm


def design_matrix_from_order_def(block_length: int, num_blocks: int, num_conditions: int, order: List[int],
                                 base_conditions_idxs: List[int], condition_names: dict, tr_length=3, mion=True,
                                 fir=False):
    """
    Creates as num_conditions x time onehot encoded matrix from an order number definition
    :param block_length:
    :param num_blocks:
    :param num_conditions:
    :param order:
    :param base_conditions_idxs: the integers corresponding to base case (usually gray) conditions
    :return: design matrix with constant one on base conditions and linear drift nuisance regressors on cols -1 and -2
    """
    fir=False
    k = len(order)
    timing = {"trial_type": [],
              "onset": [],
              "duration": []}
    for i in range(num_blocks):
        active_condition = order[i % k]
        if active_condition in base_conditions_idxs:
            # skip conditions that are baseline
            continue
        cond_name = condition_names[str(active_condition)]
        timing["trial_type"].append(cond_name)
        timing["onset"].append(i * block_length * tr_length)
        timing["duration"].append(block_length * tr_length)
    frames = np.arange(num_blocks * block_length) * tr_length
    time_df = pd.DataFrame.from_dict(timing)

    if fir:
        hrf = 'fir'
    else:
        if mion:
            hrf = "mion"
        else:
            hrf = "spm + derivative"
    dm = _make_nl_dm(frames, time_df, hrf, delay_periods=[0], stim_min_onset=-32)
    if fir and mion:
        dm *= -1
    # reorder
    cond_names = [condition_names[cn] for cn in condition_names.keys() if int(cn) not in base_conditions_idxs]
    cols = list(dm.columns)
    for i, c in enumerate(cond_names):
        cols[i] = c
    dm = dm[cols]
    return dm


def design_matrix_from_run_list(run_list: np.array, num_conditions: int, base_condition_idxs: List[int],
                                condition_names: dict, condition_groups: dict, tr_length=3.0, mion=True,
                                reorder=True, use_cond_groups=True):
    """
    Creates a design matrix from a list of lengths number of trs, holding the stimulus condition at each tr.
    Basically just onehot encodes the run_list, except base case conditions are given a constant value and a
    Linear Drift regresssor is added.
    Warning: different conditions in the same group MUST have the same block length and FIR delay
    :param run_list:
    :param num_conditions:
    :param base_condition_idxs:
    :param reorder: if true will make all features in paradigm present in dm. Otherwise, only present features will be
                    in dm.
    :return:
    """
    # convert runlist to python list if it's some numerical datatype
    if type(run_list) is np.ndarray or type(run_list) is torch.Tensor:
        run_list = run_list.tolist()
    # set up nilearn default timing specification
    # construct separate timing dicts for hemodynamic
    # and fir mode conditions.
    hrf_timing = {"trial_type": [],
                  "onset": [],
                  "duration": []}

    fir_timing = {"trial_type": [],
                  "onset": [],
                  "duration": []}
    global_fir_delay = 0

    num_trs = len(run_list)
    frames = np.arange(num_trs) * tr_length
    runlength_encode = []
    # separate each time runlist changes so we can get the length of each continuous block.
    for _, group in itertools.groupby(run_list):
        runlength_encode.append(list(group))
    # iterate through the blocks
    t = 0
    block_lengths = {}
    cond2group = {}
    for key in condition_groups.keys():
        for cond_idx in condition_groups[key]["conds"]:
            cond2group[cond_idx] = key
    name_integerizer = {v: k for k, v in condition_names.items()}

    for block in runlength_encode:
        cond = block[0]  # the integer representation for the condition of this block
        if cond not in base_condition_idxs:
            # if this condition doesn't belong to a group, it's assigned its own regression using standard hrf (not fir)
            if condition_groups is None:
                hrf_timing['trial_type'].append(condition_names[str(cond)])
                hrf_timing['onset'].append(t * tr_length)
                hrf_timing['duration'].append(len(block) * tr_length)
            else:
                # standard case, models each condition group with its own regressor of set of FIR regressors.
                group_name = cond2group[cond]
                fir_delay = condition_groups[group_name]['fir']
                if use_cond_groups:
                    # regressor for each group
                    regressor_name = group_name
                else:
                    # regressor for each leaf stimulus type
                    regressor_name = condition_names[str(cond)]
                block_lengths[group_name] = len(block)
                if fir_delay is not None:
                    if fir_delay < 0:
                        # ok, code says we want to use single TR fir regressors, with 0 delay
                        for j in range(len(block)):
                            fir_timing['trial_type'].append(regressor_name + "_r" + str(j))
                            fir_timing['onset'].append((t + j) * tr_length)
                            fir_timing['duration'].append(tr_length)
                    else:
                        fir_timing['trial_type'].append(regressor_name)
                        fir_timing['onset'].append(t * tr_length)
                        fir_timing['duration'].append(tr_length * len(block))
                else:
                    hrf_timing['trial_type'].append(regressor_name)
                    hrf_timing['onset'].append(t * tr_length)
                    hrf_timing['duration'].append(len(block) * tr_length)
        t += len(block)
    hrf_time_df = pd.DataFrame.from_dict(hrf_timing)
    fir_time_df = pd.DataFrame.from_dict(fir_timing)
    if mion:
        hrf = "mion"
    else:
        hrf = "spm + derivative"
    
    # make seperate design matrix for FIR and HRF
    fir_dm = _make_nl_dm(frames, fir_time_df, "fir", delay_periods=[1], stim_min_onset=-32)
    hrf_dm = _make_nl_dm(frames, hrf_time_df, hrf, delay_periods=[0], stim_min_onset=-32)

    # fix naming
    cols = []
    for i, cname in enumerate(fir_dm.columns):
        if "_r" in cname:
            trunc = cname[:-8]
            cols.append(trunc.replace("_r", "_delay_"))
        else:
            cols.append(cname)
    fir_dm.columns = cols

    # if we use mion, invert the fir dm
    if mion:
        fir_dm *= -1
    # get our condition names
    if condition_groups is None or not use_cond_groups or reorder is False:
        cond_names_set = [condition_names[cn] for cn in condition_names.keys() if int(cn) not in base_condition_idxs]
    else:
        # for fir we have to generate additional conditions for each delay regressor matching nilearn nomenclature
        cond_names_set = list(condition_groups.keys())
    cond_names_list = []

    for cond_name in cond_names_set:
        if use_cond_groups:
            cond_group = cond_name
            fir_delay = condition_groups[cond_name]['fir']
        else:
            cond_group = cond2group[int(name_integerizer[cond_name])]
            fir_delay = condition_groups[cond_group]['fir']
        if fir_delay is None:
            cond_names_list.append(cond_name)
        else:
            cond_names_list += [cond_name + "_delay_" + str(delay)
                                for delay in range(block_lengths[cond_group])]

    # merge the two dms
    dm = pd.concat([fir_dm, hrf_dm], axis=1)
    dupes = dm.columns.duplicated()
    dm = dm.loc[:, ~dupes]
    # reorder design matrix
    if reorder:
        for cond in cond_names_list:
            if cond not in dm.columns:
                dm.insert(0, cond, 0)
        cols = list(dm.columns)
        for cn in cols:
            if cn not in cond_names_list:
                cond_names_list.append(cn)
        dm = dm[cond_names_list]
    return dm


def nilearn_glm(source: List[str], design_matrices: List[pd.DataFrame], base_condition_idxs: List[int], output_dir: str, fname: str, mion=True, fir=True, tr_length=3., smooth=1.):
    if fir:
        if mion:
            hrf = "mion"
        else:
            hrf = "spm + derivative"
    else:
        hrf = 'fir'
    tmp_dir = os.path.join(os.path.dirname(source[0]), 'tmp_glm_cache')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    fmri_glm = first_level.FirstLevelModel(
                minimize_memory=True,
                hrf_model=hrf,
                t_r=tr_length,
                standardize=False,
                slice_time_ref=0,
                smoothing_fwhm=smooth,
                signal_scaling=False,
                verbose=True,
                n_jobs=multiprocessing.cpu_count() - 1,
                noise_model='ols',
                memory=tmp_dir
        )
    epis = [nib.load(run) if os.path.isfile(run) else nib.load(os.path.join(run, fname)) for run in source]

    # estimate high variance confounds
    print("TRY: HIGH VAR CONFOUND ESTIMATION")
    with Pool() as p:
        confounds = p.map(_confound, epis)
    index = design_matrices[0].index
    design_matrices = [pd.concat([dm, confounds[i].set_index(index)], axis=1) for i, dm in enumerate(design_matrices)]
    fmri_glm = fmri_glm.fit(epis, design_matrices=design_matrices)
    fig, axs = plt.subplots(2)
    dm = fmri_glm.design_matrices_[0]
    dms = np.stack([dm.to_numpy() for dm in fmri_glm.design_matrices_], axis=0).mean(axis=0)
    dms = pd.DataFrame(data=dms, index=dm.index, columns=dm.columns)
    plot_design_matrix(dms, ax=axs[0])
    plot_design_matrix(dm, ax=axs[1])
    plt.show()
    code = random.randint(0, 999999)
    glm_path = os.path.join(output_dir, "glm_model_" + str(code) + ".pkl")
    with open(glm_path, "wb") as f:
        pickle.dump(fmri_glm, f)
    # if os.path.exists(tmp_dir):
    #     shutil.rmtree(tmp_dir)
    return glm_path


def create_averaged_beta(beta_paths, out_dir=None):
    betas = []
    affines = []
    for beta_path in beta_paths:
        beta_nii = nib.load(beta_path)
        beta = np.array(beta_nii.get_fdata())
        local_center = np.mean(beta[:, :, :, 1:].flatten())
        betas.append(beta - local_center)
        affines.append(np.array(beta_nii.affine))
    all_betas = np.stack(betas, axis=0)
    std = np.std(all_betas[:, :, :, :, 1:].flatten())
    avg_betas = all_betas.mean(axis=0) / std
    affine = np.mean(np.stack(affines, axis=0), axis=0)
    beta_nii = nib.Nifti1Image(avg_betas, affine=affine, header=beta_nii.header)
    if out_dir is None:
        out_dir = os.path.dirname(os.path.dirname(beta_paths[0]))
    out_path = os.path.join(out_dir, 'reg_beta_coef.nii.gz')
    nib.save(beta_nii, out_path)
    return out_path


def create_contrasts(beta_matrix: str, contrast_matrix: np.ndarray, contrast_descriptors: List[str], output_dir: str):
    """
    We want voxels in A that are greater than
    """
    beta_nii = nib.load(beta_matrix)

    betas = np.array(beta_nii.get_fdata())
    avg_contrasts = contrast(betas, contrast_matrix)

    out_paths = []
    for i, cond in enumerate(np.transpose(avg_contrasts, (3, 0, 1, 2))):
        contrast_nii = nib.Nifti1Image(cond, affine=beta_nii.affine, header=beta_nii.header)
        out = os.path.join(output_dir, contrast_descriptors[i] + '_contrast.nii.gz')
        nib.save(contrast_nii, out)
        out_paths.append(out)

    return avg_contrasts, out_paths


def nilearn_contrasts(glm_model_path, contrast_matrix, contrast_descriptors, output_dir, mode='z_score'):
    glm = pickle.load(open(glm_model_path, "rb"))
    contrast_matrices = [[] for _ in range(contrast_matrix.shape[1] + 1)] # room for auto background contrast
    dm_cols = list(glm.design_matrices_[0].columns)
    n_reg_dex = dm_cols.index("drift_1")
    constant_dex = dm_cols.index("constant")
    contrast_descriptors.append("background")
    is_fir = True in ["delay" in cname for cname in dm_cols[:n_reg_dex]]

    if is_fir:
        cm = contrast_matrix.tolist()
        expanded_cm = []
        cur = None
        count = 0
        c_idx = 0
        for i, cname in enumerate(dm_cols[:n_reg_dex]):
            if "delay" in cname:
                cond = cname[:-8]
            else:
                cond = cname
            if cur is None:
                cur = cond
            elif cur != cond:
                # count is the number of times this condition repeats (supposed to combined across seperate fir regressors)
                expanded_cm += [cm[c_idx]] * (count + 1)
                count = 0
                c_idx += 1
                cur = cond
            else:
                count += 1
        contrast_matrix = np.array(expanded_cm, dtype=float)

    for dm in glm.design_matrices_:
        num_cond = dm.shape[1]
        drift_regressors = num_cond - n_reg_dex
        if is_fir:
            drift_regressors += 1

        cm = np.pad(contrast_matrix, ((0, drift_regressors), (0, 0)), constant_values=((0, 0), (0, 0))).T

        # always compute background contrast
        back_contrast = np.zeros((1, cm.shape[1]))
        back_contrast[0, constant_dex] = 1
        cm = np.concatenate([cm, back_contrast], axis=0)

        for i, contrast in enumerate(cm):
            contrast_matrices[i].append(contrast)

    out_paths = []
    for i, contrast in enumerate(contrast_matrices):
        for dm in glm.design_matrices_[:4]:
            plot_contrast_matrix(contrast, design_matrix=dm)
            plt.show()
        contrast_nii = glm.compute_contrast(contrast, stat_type='t', output_type=mode)
        out_path = os.path.join(output_dir, contrast_descriptors[i] + '.nii.gz')
        nib.save(contrast_nii, out_path)
        out_paths.append(out_path)
    return out_paths


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
    :param subject_id: name of subject (project exp)
    """
    if hemi not in ['rh', 'lh']:
        raise ValueError('Hemi must be one of rh or lh.')
    high_res = nib.load(orig_high_res_anatomical)
    high_res_data = high_res.get_fdata()
    # high_res_data = _pad_to_cube(high_res_data)
    low_res = nib.load(orig_low_res_anatomical).get_fdata()
    # low_res = _pad_to_cube(low_res)
    # scale = _find_scale_factor(high_res_data, low_res)
    contrast_nii = nib.load(contrast_vol_path)
    contrast_data = np.array(contrast_nii.get_fdata())
    aff = contrast_nii.affine
    new_nii = nib.Nifti1Image(contrast_data, affine=aff, header=contrast_nii.header)

    if not output:
        output = os.path.dirname(contrast_vol_path)
    out_desc = os.path.basename(contrast_vol_path).split('.')[0]
    path = os.path.join(output, 'tmp.nii')
    nib.save(new_nii, path)

    os.environ.setdefault("SUBJECTS_DIR", os.path.abspath(os.environ.get('FMRI_WORK_DIR')))

    # subprocess.run(['fslroi', path, path_gz, '0', '256', '0', '256', '0,', '256'])
    # path = _scale_and_standardize(scale, path)
    overlay_out_path = os.path.join(output, 'sigsurface_' + hemi + '_' + out_desc + '.mgh')
    subprocess.run(['mri_vol2surf', '--src', path,
                    '--out', overlay_out_path,
                    '--hemi', hemi,
                    '--regheader', subject_id])
    return overlay_out_path


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
        output = os.path.join(out_dir, f.split('.')[0] + '.nii.gz')
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


def _segment_contrast_image(contrast_data, kernel=.6, threshold=5., negative=False, size_thresh=4):
    """
    :param contrast_data: sig_val: <contrast_types, w, h, d>
    :param threshold:
    :param negative:
    :param size_thresh:
    :return:
    """
    # smooth over
    smoothed = ts_smooth(contrast_data, temporal_smoothing=True, kernel_std=kernel)
    segs = np.zeros_like(smoothed)
    if negative:
        segs[smoothed < (-1 * threshold)] = 1
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
    num_rois = len(valid_labels) + 1
    # reindex labels
    id = 1
    for label_id in sorted(list(valid_labels)):
        if label_id != id:
            labels[labels == label_id] = id
        id += 1

    print(num_rois, "potential", desc, "rois detected")
    return labels, len(valid_labels) + 1


def get_auto_roi_masks(contrast_paths: List[str], out_dir='./', max_rois=10, min_rois=1, sig_threshold=6):
    all_c = None
    for contrast_path in contrast_paths:
        cnii = nib.load(contrast_path)
        c = cnii.get_fdata()
        if all_c is None:
            all_c = c
        else:
            all_c += c
    all_c /= len(contrast_paths)
    std = np.std(all_c.flatten())
    all_c /= std
    h_num_labels = 99999999
    l_num_labels = 0
    count = 0
    max_iter = 150
    kernel = .1
    while (min_rois > l_num_labels or h_num_labels > max_rois) and count < max_iter:
        pos_labels, num_pos_rois = _segment_contrast_image(all_c, threshold=sig_threshold, kernel=kernel)
        neg_labels, num_neg_rois = _segment_contrast_image(all_c, threshold=sig_threshold, kernel=kernel, negative=True)
        h_num_labels = max(num_pos_rois, num_neg_rois)
        l_num_labels = min(num_pos_rois, num_neg_rois)
        if h_num_labels > max_rois:
            kernel += .01
            sig_threshold += .5
        elif l_num_labels < min_rois:
            kernel -= .02
            sig_threshold -= .25
        count += 1
    out_dir = _create_dir_if_needed(os.path.dirname(out_dir), os.path.basename(out_dir))
    for pos in (True, False):
        if pos:
            labels = pos_labels
            n = "pos"
        else:
            labels = neg_labels
            n = "neg"
        for roi_idx in np.unique(labels):
            mask = (labels == roi_idx).astype(float)
            out_path = os.path.join(out_dir, n + "_autoroi_" + str(int(roi_idx)))
            mask_nii = nib.Nifti1Image(mask, header=cnii.header, affine=cnii.affine)
            nib.save(mask_nii, out_path)
        clean_data = (labels > 0).astype(float)
        cleaned_nii = nib.Nifti1Image(clean_data, header=cnii.header, affine=cnii.affine)
        nib.save(cleaned_nii, os.path.join(out_dir, "all_" + n + "_rois.nii.gz"))
    return out_dir


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


def get_roi_betas(roi_mask_paths, betas_path):
    """
    takes a list of roi mask nifti paths and beta path, and returns corresponding  list of beta coeeficients np arryas
    :param roi_mask_paths: a list of (w, h, d) niftis
    :param betas_path: a (w, h, d, k) nifti
    :return:
    """
    betas_nii = nib.load(betas_path)
    betas = np.array(betas_nii.get_fdata())
    if len(betas.shape) == 3:
        betas = betas[:, :, :, None]
    _, _, _, k = betas.shape
    print('loaded shape: ', betas.shape, 'beta matrix')
    roi_betas = []
    roi_names = []
    for roi_mask_path in roi_mask_paths:
        roi_mask_nii = nib.load(roi_mask_path)
        roi_names.append(os.path.basename(roi_mask_path).split('.')[0])
        roi_mask = np.array(roi_mask_nii.get_fdata())
        clust_idxs = np.nonzero(roi_mask >= .5)
        roi_beta = betas[clust_idxs[0], clust_idxs[1], clust_idxs[2]].reshape(-1, k)
        roi_betas.append(roi_beta)
    return roi_betas, roi_names


def plot_roi_activation_histogram(roi_mask_paths: List[str], betas_path, num_conditions, condition_desc):
    assert len(condition_desc) == num_conditions
    plot_size = int(np.ceil(np.sqrt(num_conditions)))
    fig, axs = plt.subplots(plot_size, plot_size)
    roi_betas, roi_names = get_roi_betas(roi_mask_paths, betas_path)
    colors = plt.get_cmap('hsv')
    num_rois = len(roi_betas)
    fig.set_size_inches(h=plot_size*1.5, w=plot_size*2)
    fig.tight_layout()
    for j, roi in enumerate(roi_betas):
        _, k = roi.shape
        assert k == num_conditions
        for i in range(num_conditions):
            condition = roi[:, i]
            print('roi', j, 'cond', i, 'mean', np.mean(condition), 'dev', np.std(condition), 'max mag', np.max(np.abs(condition)))
            axs[int(np.floor(i / plot_size)), i % plot_size].hist(condition, bins='auto', color=colors(j / num_rois), alpha=.3, label=roi_names[j])
            axs[int(np.floor(i / plot_size)), i % plot_size].set_title(condition_desc[i])
            axs[int(np.floor(i / plot_size)), i % plot_size].set_xlim(left=-10, right=10)
    fig.show()
    plt.legend(loc='lower left')
    plt.show()


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
                                          fname='epi_masked.nii.gz', pre_onset_trs=6, post_offset_trs=18):
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
    functional_dirs = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in functional_dirs]
    w, h, d, _ = nib.load(functional_dirs[0]).get_fdata().shape
    corrected_arr = np.zeros((w, h, d, total_length))
    avg_count = 0
    for func in functional_dirs:
        func_dir = os.path.dirname(func)
        ima = os.path.basename(func_dir)
        order_num = ima_order_num_map[ima]
        cond_seq = order_num_defs[str(order_num)]
        cond_idxs = np.array([i for i in range(len(cond_seq)) if cond_seq[i] == target_condition])
        onset_trs = cond_idxs * block_length
        data_nii = nib.load(func)
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
