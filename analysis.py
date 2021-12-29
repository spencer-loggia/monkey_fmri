import os
import scipy.ndimage
import shutil
from typing import List, Tuple
import subprocess

import numpy as np
import nibabel as nib

from scipy.special import gamma
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
    mean = np.mean(arr.flatten())
    arr -= mean
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


def hemodynamic_convolution(x_arr: np.ndarray, kernel: str, temporal_window: int = 10, tr_len=3, **kwargs) -> np.ndarray:
    """

    :param temporal_window: number of trs to use in discrete convolution
    :param x_arr: Input stimuli sequence. (k, t) where k is the number of stimuli classes
    :param kernel: function name to tranform activations to hemodynamic activation space. Convolves over time dime (ax -1)
    :param mean: free parameter argument if kernel is gauss based, the lead of the peak on the current position, meausured in trs
    :param var:
    :return: a matrix with same shape as input.
    """
    if 'mean' in kwargs:
        mean = kwargs['mean']
    else:
        mean = 4
    if 'var' in kwargs:
        var = kwargs['var']
    else:
        var = 3

    if kernel == 'normal':
        mean = int(temporal_window / 2) + mean
        n_arr = norm.pdf(np.arange(temporal_window), loc=mean, scale=var)
        hemo_resp = np.apply_along_axis(lambda m: np.convolve(m, n_arr, mode='same'), axis=0, arr=x_arr)
        return hemo_resp

    def _gamma_hrf_conv_arr(lmean, lvar, tw):
        beta = lmean + np.sqrt(lmean + 4 * lvar) * (1 / (2 * lvar))
        alpha = lvar * np.power(beta, 2)
        temp_arr = np.arange(tw)
        conv_arr = (np.power(beta, alpha) * np.power(temp_arr, (alpha - 1)) * np.power(np.e,
                                                                                       (-temp_arr * beta))) / gamma(
            alpha)
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
            weight = .85
        if 'mean2' in kwargs:
            mean2 = kwargs['mean2']
        else:
            mean2 = 7
        if 'var2' in kwargs:
            var2 = kwargs['var2']
        else:
            var2 = 1
        conv_arr1 = _gamma_hrf_conv_arr(mean, var, temporal_window)
        conv_arr2 = _gamma_hrf_conv_arr(mean2, var2, temporal_window)
        conv_arr = weight * conv_arr1 - (1 - weight) * conv_arr2
        plt.plot(conv_arr)
        plt.show()
        hemo_resp = np.apply_along_axis(lambda m: np.convolve(m, conv_arr, mode='same'), axis=0, arr=x_arr)
        return hemo_resp

    if kernel == 'manual_bold':
        conv_arr = [.01, .02, .05, .09, .14, .2, .15, .11, .07, .03, .00, -.01, -.02, -.03, -.02, -.015, -.01, -.005, 0, 0, 0]
        conv_arr = [0]*len(conv_arr) + conv_arr
        plt.plot(conv_arr)
        plt.show()
        x_arr = scipy.ndimage.zoom(x_arr, (1, tr_len))
        hemo_resp = np.apply_along_axis(lambda m: np.convolve(m, conv_arr, mode='same'), axis=0, arr=x_arr)
        hemo_resp = scipy.ndimage.zoom(hemo_resp, (1, (1 / tr_len)))
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
        gt = gt.reshape(-1, x_arr.shape[0]).T  # reshape to all time, LL VOXELS
    glm = LinearRegression()
    glm.fit(x_arr, gt)
    beta = glm.coef_
    beta = beta.reshape(list(og_shape[:-1]) + [x_arr.shape[1]])
    residuals = (gt - glm.predict(x_arr)).T.reshape(og_shape)
    return beta, residuals


def _ssqrt_lfxn(x: torch.Tensor, y: torch.Tensor, chunk_size=10000):
    euc_dists = y - x
    loss = torch.sum(torch.sqrt(euc_dists))
    return loss


def _rmse_lfxn(x: torch.Tensor, y:torch.Tensor, chunk_size=10000):
    x = x.contiguous()
    y = y.contiguous()
    loss = torch.zeros(1)
    x_blocks = torch.split(x, split_size_or_sections=chunk_size, dim=0)
    y_blocks = torch.split(y, split_size_or_sections=chunk_size, dim=0)
    for i in range(len(x_blocks)):
        euc_dists = torch.cdist(x_blocks[i], y_blocks[i])
        loss = loss + torch.sum(torch.pow(euc_dists, 2))
    return loss


def maximal_dynamic_convolution_regressor(design_matrix: np.ndarray, gt: np.ndarray, conv_size=21, conv_pad=10, epochs=100, pid=0):
    og_shape = gt.shape
    conv1d = torch.nn.Conv1d(kernel_size=conv_size, in_channels=1, out_channels=1, padding=conv_pad)
    conv1d.weight = torch.nn.Parameter(torch.ones_like(conv1d.weight).float(), requires_grad=True)
    design_matrix = torch.from_numpy(design_matrix)[:, None, :].float()  # design matrix should be (k, t) so that k conditions is treated as batch by torch convention
    if np.ndim(gt) > 1:
        gt = gt.reshape(-1, design_matrix.shape[0]).T  # reshape to all time, LL VOXELS
    gt = torch.from_numpy(gt).float()
    # linear = torch.nn.Linear(in_features=design_matrix.shape[-1], out_features=gt.shape[1], bias=False)
    beta = torch.normal(mean=.5, std=.2, size=[design_matrix.shape[-1], gt.shape[1]], requires_grad=False).float()
    optim = torch.optim.SGD(lr=.001, params=list(conv1d.parameters()))
    mseloss = torch.nn.MSELoss()
    # mseloss = _ssqrt_lfxn
    print('Starting optimization routine... (pid=', pid, ')')

    def pred_seq(dmat):
        xexp = conv1d(dmat.T).T
        if xexp.shape != dmat.shape:
            raise ValueError("Convolution result must have same dimension as design matrix (" +
                             str(dmat.shape) + ") , not " + str(xexp.shape) +
                             ". Adjust conv size and padding size.")
        xexp = xexp.squeeze()  # remove channel dim for conv
        y_hat = xexp @ beta  # time is now treated as the batch, we do (t x k) * (k x v) -> (t x v) where v is all voxels
        return xexp, y_hat

    for i in range(epochs):
        # expectation
        optim.zero_grad()
        print("optim epoch #", i, "(pid=", pid, ")")
        x_expect, y_hat = pred_seq(design_matrix)
        # analytical maximization
        with torch.no_grad():
            beta = torch.inverse(x_expect.T @ x_expect) @ x_expect.T @ gt
        y_hat = x_expect @ beta
        loss = mseloss(gt.T, y_hat.T)
        loss.backward()
        optim.step()

    beta = np.array(beta).T
    beta = beta.reshape(list(og_shape[:-1]) + [design_matrix.shape[-1]])
    hemodynamic_est = np.array(conv1d.weight.clone().detach()).reshape(np.prod(conv1d.weight.shape))
    return beta, hemodynamic_est


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


def _pipe(d_mat, run, c_mat, mode, pid):
    """
    Single thread worker dispatch function.
    :param d_mat: design matrix from fsfast or local
    :param run: preprocessed functional data
    :param c_mat: contrast test definition matrix
    :return:
    """
    if mode == 'standard':
        beta, res = mv_glm_regressor(d_mat, ts_smooth(run, kernel_std=.5))
    elif mode == 'maximal_dynamic':
        beta, res = maximal_dynamic_convolution_regressor(d_mat, ts_smooth(run, kernel_std=.5), pid=pid)
    else:
        raise ValueError("intra subject contrast mode must be either standard or maximal_dynamic")
    contrasts = contrast(beta, c_mat, pid=pid)
    return contrasts, res


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


def create_design_matrix(param_file, subject_ids: List[str], subject_dir_paths: List[str], anal_name=None,
                         mode='manual',
                         tr=2.0, paradigm_type='blocked', funcstem='registered', convolve=False, aq_mode='bold'):
    """
    Creates an experiment design matrix. Either Uses fsfast functions or uses hemodynamic convolution functions defined
    here (~10x faster)
    :param convolve: whether to convolve the deign matrix with a built in hemodynamic function
    :param param_file:
    :param subject_ids:
    :param subject_dir_paths:
    :param anal_name:
    :param mode:
    :param tr:
    :param paradigm_type:
    :param funcstem:
    :return:
    """
    if mode not in ['fsfast', 'manual']:
        raise ValueError('mode must be either fsfast or manual')
    if mode == 'fsfast':
        subprocess.run(['mkanalysis-sess',
                        '-analysis', anal_name,
                        '-TR', str(tr),
                        '-paradigm', param_file,
                        '-event-related',
                        '-native',
                        '-nc', '3',
                        '-gamma',
                        '-fwhm', '1',
                        '-funcstem', funcstem,
                        '-refeventdur', '2.0',
                        '-fsd', 'functional'], )
        sid = ', '.join(subject_ids)
        with open('./tmp_subject_ids.info', 'w') as f:
            f.write(sid)
        sloc = ', '.join(subject_dir_paths)
        with open('./tmp_subject_loc.info', 'w') as f:
            f.write(sloc)
        subprocess.run(['selxavg-sess',
                        '-analysis', anal_name,
                        '-sf', './tmp_subject_ids.info',
                        '-df', './tmp_subject_loc.info'])

    elif mode == 'manual':
        stimuli = pd.read_csv(param_file, delimiter=' ')
        stimuli_list = list(stimuli['ID'])
        encoder = OneHotEncoder(sparse=False)
        design_matrix = encoder.fit_transform(np.array([stimuli_list]).reshape(-1, 1))
        if convolve:
            if aq_mode == 'bold':
                design_matrix = hemodynamic_convolution(design_matrix, kernel='manual_bold', temporal_window=10)
            elif aq_mode == 'mion':
                design_matrix = hemodynamic_convolution(design_matrix, kernel='manual_mion', temporal_window=10)
    return design_matrix


def design_matrix_from_order_def(block_length: int, num_blocks: int, num_conditions: int, order: List[int], convolve=False, aq_mode='bold'):
    k = num_conditions
    design_matrix = np.zeros((0, k))
    for i in range(num_blocks):
        block = np.zeros((block_length, k), dtype=float)
        active_condition = order[i % k]
        block[:, active_condition] = 1
        design_matrix = np.concatenate([design_matrix, block], axis=0)
    if convolve:
        if aq_mode == 'bold':
            design_matrix = hemodynamic_convolution(design_matrix, kernel='manual_bold', temporal_window=10)
        elif aq_mode == 'mion':
            design_matrix = hemodynamic_convolution(design_matrix, kernel='manual_mion', temporal_window=15)
    return design_matrix


def intra_subject_contrast(run_dirs: List[str], design_matrices: List[np.ndarray], contrast_matrix: np.ndarray,
                           contrast_descriptors: List[str], output_dir: str, fname: str = 'registered.nii.gz',
                           mode='standard', use_python_mp=False):
    """
    Compute each desired contrast on the functional data in parallel.
    :param design_matrix: list of array of size conditions x num trs. length 1 if stim order same for all imas,
                          len equal to number of runs otherwise. order corresonds to ima order in run dirs.
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
                print("WARNIING: Loaded functional data (" + source_dir + ") must have number of frames equal to length of "
                                 "active_condition map, not " + str(brain_tensor.shape[-1]) + ' and ' + str(exp_num_frames))
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
                               list(range(len(run_stack)))))
    if use_python_mp:
        with Pool() as p:
            # dispatch workers, by default will claim all but one cpu core.
            res = p.starmap(_pipe, full_run_params)
    else:
        res = []
        for i in range(len(full_run_params)):
            res.append(_pipe(*full_run_params[i]))
    contrasts, hemo = list(zip(*res))
    avg_contrasts = np.mean(np.stack(contrasts, axis=0), axis=0)[0]

    print(avg_contrasts.shape)
    affine = np.concatenate(affines, axis=0).mean(axis=0)
    for i, cond in enumerate(np.transpose(avg_contrasts, (3, 0, 1, 2))):
        cond = _norm_4d(cond)
        contrast_nii = nib.Nifti1Image(cond, affine=affine, header=header)
        nib.save(contrast_nii, os.path.join(output_dir, contrast_descriptors[i] + '_contrast.nii'))
    fig, axs = plt.subplots(1)
    for i, h in enumerate(hemo):
        axs.plot(h, label=i)
    axs.legend()
    fig.show()
    plt.show()
    return avg_contrasts


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

    os.environ.setdefault("SUBJECTS_DIR", os.path.abspath(os.getcwd()))

    if not os.path.exists('./mri/orig.mgz'):
        _create_dir_if_needed('./', 'mri')
        shutil.copy(orig_high_res_anatomical, './mri/orig.mgz')
    if not os.path.exists('./surf/lh.white'):
        _create_dir_if_needed('./', 'surf')
        shutil.copy(anatomical_white_surface, './surf/lh.white')
    subdivide_arg = str(1 / scale)
    subprocess.run(['mri_convert', path, '-vs', subdivide_arg, subdivide_arg, subdivide_arg, path])
    subprocess.run(['mri_convert', path, '-iis', '1', '-ijs', '1', '-iks', '1', path])
    path_gz = os.path.join(output, 'tmp.nii.gz')
    subprocess.run(['fslroi', path, path_gz, '0', '256', '0', '256', '0,', '256'])
    overlay_out_path = os.path.join(output, 'sigsurface_' + hemi + '_' + out_desc + '.mgh')
    subprocess.run(['mri_vol2surf', '--src', path_gz,
                    '--out', overlay_out_path,
                    '--hemi', hemi,
                    '--regheader', subject_id])


def _fit_model(model: GaussianMixture, data):
    assignments = model.fit_predict(data)
    bic = model.bic(data)
    return assignments, bic


def _segment_contrast_image(contrast_data, threshold=5., negative=False, size_thresh=5):
    # smooth over
    smoothed = ts_smooth(contrast_data, temporal_smoothing=True, kernel_std=1)
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


def _get_roi_time_course(rois, label_id, fdata, ax, block_length, block_order, size_thresh=0):
    """

    :param num_rois:
    :param rois:
    :param fdata: a created functional average
    :param threshold: significance threshold to consider voxels
    :param cluster_min: minimum number of voxels
    :param cluster_max: maximum number of voxels
    :return:
    """
    colors = ['grey', 'red', 'blue', 'cyan', 'purple', 'yellow', 'green', 'orange', 'pink', 'bronze']
    clust_idxs = np.nonzero(rois == label_id)
    if len(clust_idxs[0]) <= size_thresh:
        return ax
    clust_ts_vox = fdata[clust_idxs]
    spatial_flattened = clust_ts_vox.reshape(-1, clust_ts_vox.shape[-1])
    mean_ts = np.mean(spatial_flattened, axis=0)
    trs = mean_ts.shape[0]
    std_ts = np.std(spatial_flattened, axis=0)
    ax.plot(mean_ts, c='blue')
    ax.plot(mean_ts - std_ts, c='gray')
    ax.plot(mean_ts + std_ts, c='gray')
    ax.set_title("ROI # " + str(label_id))
    ax.xaxis.set_ticks(np.arange(0, trs, block_length))
    for i, tr in enumerate(range(0, trs, block_length)):
        c = colors[block_order[i % len(block_order)]]
        ax.axvspan(tr, tr + block_length, color=c, alpha=0.5)
    return ax


def segment_get_time_course(contrast_file: str, functional_file: str, block_length, block_order):
    c_nii = nib.load(contrast_file)
    c_data = np.array(c_nii.get_fdata())
    c_data = _norm_4d(c_data)
    f_data = np.array(nib.load(functional_file).get_fdata())
    pos_seg, num_pos_rois = _segment_contrast_image(c_data, threshold=np.std(c_data) * 2, negative=False)
    neg_seg, num_neg_rois = _segment_contrast_image(c_data, threshold=np.std(c_data) * -2, negative=True)
    mask = np.zeros_like(pos_seg)
    mask[pos_seg > 0] = 1
    mask[neg_seg > 0] += 1
    pos_seg[mask == 2] = -1
    neg_seg[mask == 2] = -1
    brain_area = np.array(np.nonzero(f_data[:, :, :, 0] > np.mean(f_data[:, :, :, 0].flatten())))
    brain_area = brain_area[:, np.random.choice(brain_area.shape[1], 2000, replace=False)]
    fig_3d = plt.figure()
    ax3d = Axes3D(fig_3d)
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
        fig_ts, axs_ts = plt.subplots(num_rois - 1)
        fig_ts.suptitle(desc + ' condition time series')
        fig_ts.set_size_inches(8, 1.5 * num_rois)
        fig_ts.tight_layout()
        for label_id in range(1, num_rois):
            _get_roi_time_course(seg, label_id, f_data, axs_ts[label_id - 1], block_length, block_order)
            _plot_3d_scatter(seg, label_id, brain_area, ax3d, color)
        _plot_3d_scatter(seg, -1, brain_area, ax3d, 'black')
        fig_ts.show()
    fig_3d.show()
    plt.show()
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
                            'WoosterMerridianMGH4CHAN/analysis/reg_horizontal_minus_vertical_merridians_mk2_contrast.nii',
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
