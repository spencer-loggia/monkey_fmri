import os
import shutil
from typing import List, Tuple
import subprocess

import numpy as np
import nibabel as nib

from scipy.special import gamma
from scipy.stats import norm
from scipy.ndimage import gaussian_filter

from multiprocessing import Pool
from preprocess import _pad_to_cube

from sklearn.preprocessing import OneHotEncoder
try:
    from sklearnex import patch_sklearn
    from sklearnex.linear_model import LinearRegression
    patch_sklearn()
except ModuleNotFoundError:
    print("Intel(R) Hardware Acceleration is not enabled. ")
    from sklearn.linear_model import LinearRegression

import pandas as pd

from typing import List, Union, Tuple, Dict


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


def ts_smooth(input_ts: np.ndarray, kernel_std=2., temporal_smoothing=True):
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


def hemodynamic_convolution(x_arr: np.ndarray, kernel: str, temporal_window: int = 8, **kwargs) -> np.ndarray:
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
            mean2 = 5
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
        gt = gt.reshape(-1, x_arr.shape[0]).T  # reshape to all time, LL VOXELS
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
    """
    Single thread worker dispatch function.
    :param d_mat: design matrix from fsfast or local
    :param run: preprocessed functional data
    :param c_mat: contrast test definition matrix
    :return:
    """
    beta, res = mv_glm_regressor(d_mat, ts_smooth(run, kernel_std=.5))
    contrasts = contrast(beta, c_mat)
    return contrasts


def _create_SNR_volume_wrapper(input_dir, noise_path, out_path,type='tSNR'):
    func = nib.load(input_dir)
    func_data = func.get_fdata()
    func_affine = func.affine
    noise = nib.load(noise_path)
    noise_data = noise.get_fdata()
    if type == 'tSNR':
        sd = np.std(noise_data,axis=3)
        sd[np.where(sd == 0)] = 1
        mean = np.mean(func_data,axis=3)
        res = mean/sd
    elif type == 'CNR':
        sdn = np.std(noise_data, axis=3)
        sdn[np.where(sdn == 0)] = 1
        sdf = np.std(func_data, axis=3)
        sdf[np.where(sdf == 0)] = 1
        res = sdf/sdn
    res_nii = nib.Nifti1Image(res, func_affine)
    nib.save(res_nii, os.path.join(out_path, '{}.nii.gz'.format(type)))


def create_SNR_map(input_dirs: List[str], noise_dir, output: Union[None, str] = None, fname = 'clean.nii.gz',type='tSNR'):
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

        args.append((input_dir,noise_path,out_dir,type))
    with Pool() as p:
        res = p.starmap(_create_SNR_volume_wrapper,args)



def create_design_matrix(param_file, subject_ids: List[str], subject_dir_paths: List[str], anal_name=None, mode='manual',
                         tr=2.0, paradigm_type='blocked', funcstem='registered'):
    """
    Creates an experiment design matrix. Either Uses fsfast functions or uses hemodynamic convolution functions defined
    here (~10x faster)
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
                        '-fsd', 'functional'],)
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
        stimuli_feature_matrix = encoder.fit_transform(np.array([stimuli_list]).reshape(-1, 1))
        design_matrix = hemodynamic_convolution(stimuli_feature_matrix, kernel='weighted_gamma_hrf', temporal_window=15)

    return design_matrix


def intra_subject_contrast(run_dirs: List[str], paradigm_file: str, contrast_matrix: np.ndarray,
                           contrast_descriptors: List[str], output_dir: str, fname: str = 'registered.nii.gz'):
    """
    Compute each desired contrast on the functional data in parallel.
    :param run_dirs: locations of the functional runs
    :param paradigm_file: The events as they occur.
    :param contrast_matrix: The contrast test definitions.
    :param contrast_descriptors: Description of the contrast test specified by each row of the matrix.
    :param output_dir: where to save
    :param fname: name of the preprocessed functional file to use (must exist in each of the run dirs)
    :return:
    """
    run_stack = []
    affines = []
    header = None
    design_matrix = create_design_matrix(paradigm_file, ['castor_test'], ['./castor_test'])
    exp_num_frames = len(design_matrix)

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
            header = brain.header
        else:
            raise FileNotFoundError("Couldn't find " + fname + " in specified source dir " + source_dir)

    with Pool() as p:
        # dispatch workers, by default will claim all but one cpu core.
        res = p.starmap(_pipe, zip([design_matrix]*len(run_stack), run_stack, [contrast_matrix]*len(run_stack)))

    avg_contrasts = np.mean(np.stack(res, axis=0), axis=0)[0]

    affine = np.concatenate(affines, axis=0).mean(axis=0)
    for i, cond in enumerate(np.transpose(avg_contrasts, (3, 0, 1, 2))):
        contrast_nii = nib.Nifti1Image(cond, affine=affine, header=header)
        nib.save(contrast_nii, os.path.join(output_dir, 'condition_' + contrast_descriptors[i] + '_contrast.nii'))

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


if __name__ == '__main__':
    # main method used for quick and dirty testing, do not expect to functional properly depending on local directory
    # structure / filenames
    root = '/Users/loggiasr/Projects/fmri/monkey_fmri/castor_test'
    functional_dir = os.path.join(root, 'functional/20100131/')
    SOURCE = [os.path.join(functional_dir, f) for f in os.listdir(functional_dir) if f[0] != '.']
    desired_contrast_mat = np.array([[0, -1, 1],
                                     [-1, .5, .5]]).T
    contrast_desc = ['vertical_vs_horizontal', 'null_vs_avg_vert_horiz']
    # res = intra_subject_contrast(run_dirs=SOURCE,
    #                              paradigm_file=os.path.join(root, 'stimuli/meridian_mapper_order1.para'),
    #                              contrast_matrix=desired_contrast_mat,
    #                              contrast_descriptors=contrast_desc,
    #                              output_dir=os.path.join(root, 'analysis_out'),
    #                              fname='registered.nii.gz', )
    create_contrast_surface('castor_test/surf/rh.white',
                            'castor_test/analysis_out/condition_vertical_vs_horizontal_contrast.nii',
                            'castor_test/mri/stripped.nii.gz',
                            'castor_test/mri/orig.mgz',
                            hemi='rh', subject_id='castor_test')
