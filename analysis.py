import os
from typing import List, Tuple

import numpy as np
import nipype
import nibabel as nib
import nipy
from scipy.special import gamma
from scipy.stats import norm
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn.preprocessing import OneHotEncoder
try:
    from sklearnex import patch_sklearn
    from sklearnex.linear_model import LinearRegression
    patch_sklearn()
except ModuleNotFoundError:
    print("Intel(R) Hardware Acceleration is not enabled. ")
    from sklearn.linear_model import LinearRegression


def _pos_norm_4d(arr: np.ndarray):
    time_base = np.min(arr, axis=3)
    arr -= time_base
    norm_factor = np.std(arr.flatten())
    arr /= norm_factor
    return arr


def _create_dir_if_needed(base: str, name: str):
    out_dir = os.path.join(base, name)
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    return out_dir


def ts_smooth(input_ts: np.ndarray, kernel_std=2, temporal_smoothing=True):
    # assumes last axis is time
    def _smooth(input_arr, std):
        output = gaussian_filter(input_arr, sigma=std)
        return output
    if temporal_smoothing:
        smoothed = _smooth(input_ts, kernel_std)
    else:
        smoothed = np.apply_along_axis(lambda m: _smooth(m, kernel_std), axis=3, arr=input_ts)
    return smoothed


def hemodynamic_convolution(x_arr: np.ndarray, kernel: str, temporal_window: int = 10, **kwargs) -> np.ndarray:
    """

    :param temporal_window: number of trs to use in discrete convolution
    :param x_arr: Input stimuli sequence. (k, t) where k is the number of stimuli classes
    :param kernel: function name to tranform activations to hemodynamic activation space. Convolves over time dime (ax -1)
    :param mean: free parameter argument if kernel is gauss based, the lead of the peak on the current position, meausured in trs
    :param var:
=    :return: a matrix with same shape as input.
    """
    if 'mean' in kwargs:
        mean = kwargs['mean']
    else:
        mean = 2
    if 'var' in kwargs:
        var = kwargs['var']
    else:
        var = 1.5

    if kernel == 'normal':
        mean = int(temporal_window / 2) + mean
        n_arr = norm.pdf(np.arange(temporal_window), loc=mean, scale=var)
        hemo_resp = np.apply_along_axis(lambda m: np.convolve(m, n_arr, mode='same'), axis=0, arr=x_arr)
        return hemo_resp

    def _gamma_hrf_conv_arr(lmean, lvar, tw):
        beta = lmean + np.sqrt(lmean + 4 * lvar) * (1 / (2 * lvar))
        alpha = lvar * np.power(beta, 2)
        temp_arr = np.arange(tw)
        conv_arr = (np.power(beta, alpha) * np.power(temp_arr, (alpha - 1)) * np.power(np.e, (-temp_arr * beta))) / gamma(alpha)
        conv_arr = np.concatenate([np.zeros(tw), conv_arr])
        return conv_arr

    if kernel == 'gamma_hrf':
        conv_arr = _gamma_hrf_conv_arr(mean, var, temporal_window)
        hemo_resp = np.apply_along_axis(lambda m: np.convolve(m, conv_arr, mode='same'), axis=0, arr=x_arr)
        return hemo_resp

    if kernel == 'weighted_gamma_hrf':
        if 'weight' in kwargs:
            weight = kwargs['weight']
        else:
            weight = .7
        if 'mean2' in kwargs:
            mean2 = kwargs['mean2']
        else:
            mean2 = 4
        if 'var2' in kwargs:
            var2 = kwargs['var2']
        else:
            var2 = 1
        conv_arr1 = _gamma_hrf_conv_arr(mean, var, temporal_window)
        conv_arr2 = _gamma_hrf_conv_arr(mean2, var2, temporal_window)
        conv_arr = weight * conv_arr1 - (1 - weight) * conv_arr2
        hemo_resp = np.apply_along_axis(lambda m: np.convolve(m, conv_arr, mode='same'), axis=0, arr=x_arr)
        return hemo_resp

    raise NotImplementedError


def mv_glm_regressor(x_arr: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preforms multiclass regression to predict gt (activation data) from x_arr (stimuli presentation data.)
    :param x_arr: Transformed design matrix (convolved onehot encoded stimuli presentation)
    :param gt: Activation signals (w, h, d, t) if whole brain or (t) if one voxel
    :return: a tuple of Model fit parameters (Beta) and Error (difference between predicted and observed)
    """
    og_shape = gt.shape
    if np.ndim(gt) > 1:
        gt = gt.reshape(-1, x_arr.shape[0]).T # reshape to all time, LL VOXELS
    glm = LinearRegression()
    glm.fit(x_arr, gt)
    beta = glm.coef_
    beta = beta.reshape(list(og_shape[:-1]) + [x_arr.shape[1]])
    residuals = (gt - glm.predict(x_arr)).T.reshape(og_shape)
    return beta, residuals


def contrast(beta_coef: np.ndarray, contrast_matrix: np.ndarray) -> np.ndarray:
    """
     just a wrapper for matrix multiplication so I can put some expected dimensions :)
    :param beta_coef: a (w, h, d, k) matrix, where k is conditions
    :param contrast_matrix: a (k, m) matrix, where m is the number of contrasts to preform
    :return: resulting contrast, a (w, h, d, m) matrix
    """
    return beta_coef @ contrast_matrix


def _pipe(d_mat, run, c_mat):
    beta, res = mv_glm_regressor(d_mat, ts_smooth(run))
    contrasts = contrast(beta, c_mat)
    return contrasts


def intra_subject_contrast(run_dirs: List[str], active_condition: List[int], contrast_matrix: np.ndarray,
                           contrast_descriptors: List[str], output_dir: str, fname: str = 'registered.nii.gz'):
    run_stack = []
    affines = []
    exp_num_frames = len(active_condition)

    encoder = OneHotEncoder(sparse=False)
    stimuli_feature_matrix = encoder.fit_transform(np.array([active_condition]).reshape(-1, 1))
    design_matrix = hemodynamic_convolution(stimuli_feature_matrix, kernel='weighted_gamma_hrf', temporal_window=15)
    if contrast_matrix.shape[0] != design_matrix.shape[1]:
        raise ValueError("Contrast matrix must have number of rows equal to the number of stimulus conditions.")
    if len(contrast_descriptors) != contrast_matrix.shape[1]:
        raise ValueError("Number of Contrast Descriptors must match number of cols in contrast matrix")

    for source_dir in run_dirs:
        req_path = os.path.join(source_dir, fname)
        if os.path.isfile(req_path):
            brain = nib.load(req_path)
            brain_tensor = np.array(brain.get_fdata())
            if brain_tensor.shape[-1] != exp_num_frames:
                raise ValueError("Loaded functional data must have number of frames equal to length of "
                                 "active_condition map")
            run_stack.append(brain_tensor[None, :, :, :, :])
            affines.append(np.array(brain.affine)[None, :, :])
        else:
            raise FileNotFoundError("Couldn't find " + fname + " in specified source dir " + source_dir)

    with Pool() as p:
        res = p.starmap(_pipe, zip([design_matrix]*len(run_stack), run_stack, [contrast_matrix]*len(run_stack)))
    avg_contrasts = np.mean(np.stack(res, axis=0), axis=0)[0]
    affine = np.concatenate(affines, axis=0).mean(axis=0)
    for i, cond in enumerate(np.transpose(avg_contrasts, (3, 0, 1, 2))):
        contrast_nii = nib.Nifti1Image(cond, affine=affine)
        nib.save(contrast_nii, os.path.join(output_dir, 'condition_' + contrast_descriptors[i] + '_contrast.nii'))

    return avg_contrasts


if __name__ == '__main__':
    # x = np.zeros((90, 3))
    # x[0:30, 0] = 1
    # x[30:60, 1] = 1
    # x[60:90, 2] = 1
    # gt = np.random.normal(0, 1, (10, 10, 10, 90))
    # gt[:, :, 2:8, 30:60] = np.random.normal(1, 1, (6, 30))
    # h = hemodynamic_convolution(x_arr=x, kernel='weighted_gamma_hrf', temporal_window=15)
    # gt = ts_smooth(gt)
    # beta, residuals = mv_glm_regressor(h, gt)
    # cont_mat = np.array([-.5, 1, -.5])
    # contrast_matrix = contrast(beta, cont_mat)
    # plt.show()
    import pandas as pd
    stimuli = pd.read_csv('full_session_test/meridian_stimuli/meridian_mapper_order1.para', delimiter=' ')
    stimuli.head()
    stimuli_list = list(stimuli['ID'])
    root = 'full_session_test/20100131Castor/MION'
    sources = [os.path.join(root, f) for f in os.listdir(root) if f.isnumeric()]
    res = intra_subject_contrast(sources, stimuli_list,
                                 output_dir='full_session_test/analysis_out',
                                 contrast_matrix=np.array([[-1, 1, 0], [-1, 0, 1]]).T,
                                 contrast_descriptors=['null_vs_horizontal', 'null_vs_vertical'])
