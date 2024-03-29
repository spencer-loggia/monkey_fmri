import copy
import itertools
import multiprocessing
import pickle
import typing

import pandas
import pandas as pd
import torch
from typing import List, Union, Tuple

import json
import nibabel
import shutil

import input_control
import os
import preprocess
import analysis
import unpack
import fnmatch
import numpy as np

import traceback
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

matplotlib.rcParams['axes.prop_cycle'] = cycler(color='brgcmyk')

from subprocess import call


# subject support function must be operating with working directory appropriately set

def _env_setup():
    subj_root = os.path.realpath(os.environ.get('FMRI_WORK_DIR'))
    project_root = os.path.realpath(os.path.join(subj_root, '..', '..'))
    os.chdir(project_root)
    return subj_root, project_root


def include_patterns(*patterns):
    """ Function that can be used as shutil.copytree() ignore parameter that
    determines which files *not* to ignore, the inverse of "normal" usage.

    This is a factory function that creates a function which can be used as a
    callable for copytree()'s ignore argument, *not* ignoring files that match
    any of the glob-style patterns provided.

    ‛patterns’ are a sequence of pattern strings used to identify the files to
    include when copying the directory tree.

    Example usage:

        copytree(src_directory, dst_directory,
                 ignore=include_patterns('*.sldasm', '*.sldprt'))

    Author: martineau
    """

    def _ignore_patterns(path, all_names):
        # Determine names which match one or more patterns (that shouldn't be
        # ignored).
        keep = (name for pattern in patterns
                for name in fnmatch.filter(all_names, pattern))
        # Ignore file names which *didn't* match any of the patterns given that
        # aren't directory names.
        dir_names = (name for name in all_names if os.path.isdir(os.path.join(path, name)))
        return set(all_names) - set(keep) - set(dir_names)

    return _ignore_patterns


def get_images(null, *argv):
    session_id = argv[0]
    subj_root, project_root = _env_setup()
    from_dicom = input_control.bool_input("load new images from dicoms?")
    f_dir = preprocess._create_dir_if_needed(subj_root, 'sessions')
    if from_dicom:
        dicom_dir = input_control.dir_input('Enter directory contraining DICOMS of images we want to look at')
        ima_numbers = input_control.int_list_input(
            "Enter whitespace separated list of valid image IMA numbers. (e.g. '2 4 5')"
        )
        SOURCE = unpack.unpack_other_list(dicom_dir, f_dir, ima_numbers, session_id)
        print("Paths of the new images: \n",
              SOURCE)
        return SOURCE
    else:
        return None


def get_epis(*argv):
    session_id = argv[0]
    subj_root, project_root = _env_setup()
    from_dicom = input_control.bool_input("load new session dicoms? (otherwise must import niftis from exp)")
    f_dir = preprocess._create_dir_if_needed(subj_root, 'sessions')
    if from_dicom:
        dicom_dir = input_control.dir_input('Enter directory containing DICOMS from scans we are trying to analyze')
        run_numbers = input_control.int_list_input(
            "Enter whitespace separated list of valid epi run IMA numbers. (e.g. '2 4 5')")
        SOURCE = unpack.unpack_run_list(dicom_dir, f_dir, run_numbers, session_id, 'f')
        SOURCE = [os.path.relpath(s, project_root) for s in SOURCE]
        print("Created the following functional run directories: \n ",
              SOURCE,
              " \n each containing a raw functional file 'f.nii.gz'")
    else:
        nifti_source = input_control.dir_input(
            "Enter path to session directory containing run subdirectories containing niftis")
        nifti_name = input("Enter name of nifti files to transfer (should be unprocessed versions")
        sess_target_dir = os.path.join(f_dir, str(session_id))
        if not os.path.samefile(nifti_source, sess_target_dir):
            shutil.copytree(nifti_source, sess_target_dir, ignore=include_patterns(nifti_name))
        else:
            preprocess._create_dir_if_needed(f_dir, str(session_id))
        SOURCE = [os.path.relpath(os.path.join(sess_target_dir, f), project_root) for f in os.listdir(sess_target_dir)
                  if f.isnumeric()]
        if nifti_name != 'f.nii.gz':
            for run_dir in SOURCE:
                raw_nifti = os.path.join(run_dir, nifti_name)
                target_nifti = os.path.join(run_dir, 'f.nii.gz')
                if not os.path.samefile(raw_nifti, target_nifti):
                    shutil.move(raw_nifti, target_nifti)
    check = input_control.bool_input("Check functional time series lengths? ")
    if check:
        expected = int(input("What is the expected number of trs? "))
        SOURCE = preprocess.check_time_series_length(SOURCE, fname='f.nii.gz', expected_length=expected)
    SOURCE = [os.path.join(s, "f.nii.gz") for s in SOURCE]
    return SOURCE


def get_functional_target():
    subj_root = os.environ.get('FMRI_WORK_DIR')
    path = input_control.dir_input("Enter path to functional data representative image (should be masked): ")
    if ".nii.gz" in path:
        out = 'functional_target.nii.gz'
    elif ".nii" in path:
        out = 'functional_target.nii'
    else:
        raise ValueError("did not enter a valid nifti.")
    out_path = os.path.join(subj_root, 'mri', out)
    try:
        shutil.copy(path, out_path)
    except shutil.SameFileError:
        pass
    return out_path


def get_fixation_csv(source):
    subj_root, project_root = _env_setup()
    path = input_control.dir_input("Path to fixation csv file (length trs, header with IMAs)")
    out_paths = []
    fix_data = pd.read_csv(path)
    for run_dir in source:
        if os.path.isfile(run_dir):
            ima_dir = os.path.dirname(run_dir)
        else:
            ima_dir = run_dir
        ima = os.path.basename(ima_dir)
        try:
            ima_fix_data = fix_data[ima]
            out_paths.append(os.path.join(ima_dir, "fixation.csv"))
            ima_fix_data.to_csv(out_paths[-1])
        except KeyError:
            print("WARN: no fixation data found for IMA " + str(ima))
            pass
    return out_paths


def dilate_mask(inpath, outpath=None):
    """
    Simple function to call fslmath's dilM
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    if outpath is None:
        filename = os.path.basename(inpath)
        outname = 'dil_' + filename
        outpath = os.path.join(os.path.dirname(inpath), outname)
    cmd = 'fslmaths %s -bin -mul -1 -add 1 -kernel boxv 7 -ero -mul -1 -add 1 %s' % (inpath, outpath)
    call(cmd, shell=True)
    print(cmd)
    if outpath[-3:] != '.gz':
        outpath = outpath + '.gz'
    return outpath


def downsample_anatomical(inpath, factor=[2, 2, 2], out_dir=None, affine_scale=None, resample='interpolate'):
    """
    Resample should be either 'interpolate' or 'nearest'
    :param inpath:
    :param factor:
    :param out_dir:
    :param resample:
    :return:
    """
    subj_root, project_root = _env_setup()
    source_dir = os.path.dirname(inpath)
    name = os.path.basename(inpath)
    out_name = 'ds_' + name.split('.')[0] + '.nii.gz'
    factor = input_control.int_list_input("Enter the scale factor (x y z):")
    if out_dir is None:
        output_dir = preprocess._create_dir_if_needed(subj_root, 'mri')
    else:
        output_dir = out_dir
    output = os.path.relpath(os.path.join(output_dir, out_name), project_root)
    preprocess.create_low_res_anatomical(source_dir, name, output,
                                         factor=factor, affine_scale=affine_scale, resample=resample)
    return output


def downsample_vol_rois(roi_dict, ds_roi_dict, ds_t1, affine_scale=None, resample='nearest',
                        output_dir=None):
    options = list(roi_dict)
    subj_root, project_root = _env_setup()
    choice = input_control.select_option_input(options)
    roi_set_name = options[choice]
    roi_set_path = roi_dict[roi_set_name]
    ds_t1 = nibabel.load(ds_t1)
    zooms = ds_t1.header.get_zooms()
    factor = [zooms[0], zooms[1], zooms[2]]
    # factor = input_control.int_list_input("Enter the scale factor (x y z):")
    print("Scaling voxels to:", factor)
    if output_dir is None:
        output_dir = os.path.relpath(
            preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set_name), 'ds_roi_vols'),
            project_root)
    for roi_file in os.listdir(roi_set_path):
        if '.nii' in roi_file:
            out = os.path.join(output_dir, roi_file)
            preprocess.create_low_res_anatomical(roi_set_path, roi_file, out, factor=factor, affine_scale=affine_scale,
                                                 resample=resample)
    ds_roi_dict[roi_set_name] = output_dir
    return ds_roi_dict


def downsample_vol_rois_cmdline_wrap(dir, factor=[2, 2, 2], affine_scale=None, resample='nearest', output_dir=None):
    """
    So we can use outside of project structure
    """
    factor = input_control.int_list_input("Enter the scale factor (x y z):")
    roi_dict = {os.path.basename(dir): dir}
    out_dict = {}
    return downsample_vol_rois(roi_dict, out_dict, factor=factor, affine_scale=affine_scale, resample=resample,
                               output_dir=output_dir)

def bandpass_wrapper(functional_dirs, paradigm_path):
    subj_root, project_root = _env_setup()
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    blocklength = paradigm_data['block_length_trs']
    return preprocess.batch_bandpass_functional(functional_dirs, blocklength)


def coreg_wrapper(source_space_vol_path, target_space_vol_path, nonlinear=True):
    """

    :param source_space_vol_path:
    :param target_space_vol_path:
    :return: forward_transform_path, inverse_transform_path
    """
    subj_root, project_root = _env_setup()
    out = os.path.join(os.path.dirname(source_space_vol_path), 'coreg_3df.nii.gz')
    if nonlinear:
        ltrns = ['Affine', 'SyN']
    else:
        ltrns = ['Affine']
    return tuple([os.path.relpath(f, project_root) for f in preprocess.antsCoReg(os.path.realpath(target_space_vol_path),
                                                                                 os.path.realpath(source_space_vol_path),
                                                                                 outP=out,
                                                                                 ltrns=ltrns,
                                                                                 n_jobs=2,
                                                                                 full=nonlinear)])


def _apply_warp_wrapper(s, vol, out, transforms, interp, type_code, dim, invert, project_root):
    preprocess.antsApplyTransforms(s, vol, out, transforms, interp,
                                   img_type_code=type_code, dim=dim, invertTrans=invert)
    return os.path.relpath(out, project_root)


def apply_warp(source, vol_in_target_space, forward_gross_transform_path=None, fine_transform_path=None, type_code=0,
               dim=3, interp='Linear'):
    subj_root, project_root = _env_setup()
    if type(source) is not list:
        source = [source]
    transforms = [fine_transform_path, forward_gross_transform_path]
    transforms = [t for t in transforms if t is not None]
    to_invert = [False] * len(transforms)
    args = [source, [vol_in_target_space] * len(source),
            [os.path.join(os.path.dirname(s), 'reg_' + os.path.basename(s)) for s in source],
            [transforms] * len(source),
            [interp] * len(source),
            [type_code] * len(source),
            [dim] * len(source),
            [to_invert] * len(source),
            [project_root] * len(source)
            ]
    args = list(zip(*args))
    with multiprocessing.Pool(8) as p:
        out_paths = p.starmap(_apply_warp_wrapper, args)
    if len(out_paths) == 1:
        out_paths = out_paths[0]
    return out_paths


def apply_warp_4d(source, vol_in_target_space, forward_gross_transform_path=None, fine_transform_path=None):
    subj_root, project_root = _env_setup()
    return apply_warp(source, vol_in_target_space, forward_gross_transform_path=forward_gross_transform_path,
                      fine_transform_path=fine_transform_path, type_code=3, dim=3)


def apply_warp_4d_refactor(source, vol_in_target_space, forward_gross_transform_path=None, fine_transform_path=None,
                           fname='moco.nii.gz'):
    gg_source = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in source]
    warped_source = apply_warp_4d(gg_source, vol_in_target_space, forward_gross_transform_path, fine_transform_path)
    return warped_source


def apply_warp_inverse(source, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path,
                       out=None, type_code=0, dim=3, interp='NearestNeighbor'):
    """
    :param source:
    :param vol_in_target_space:
    :param forward_gross_transform_path:
    :param reverse_fine_transform_path:
    :param out:
    :return:
    """
    subj_root, project_root = _env_setup()
    if out is None:
        out = os.path.join(os.path.dirname(vol_in_target_space), 'inverse_trans_' + os.path.basename(source))
    inverse_transforms = [os.path.abspath(forward_gross_transform_path), os.path.abspath(reverse_fine_transform_path)]
    to_invert = [True, False]
    preprocess.antsApplyTransforms(os.path.abspath(source), os.path.abspath(vol_in_target_space), os.path.abspath(out),
                                   inverse_transforms, interp,
                                   img_type_code=type_code, dim=dim, invertTrans=to_invert)
    return os.path.relpath(out, project_root)


def apply_warp_inverse_vol_roi_dir(ds_vol_roi_dict, vol_in_target_space, forward_gross_transform_path,
                                   reverse_fine_transform_path, func_space_rois_dict):

    subj_root, project_root = _env_setup()
    options = list(ds_vol_roi_dict)
    choice = input_control.select_option_input(options)
    roi_set_name = options[choice]
    roi_set_path = ds_vol_roi_dict[roi_set_name]
    output_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set_name), 'auto_func_space_rois')
    for roi_file in os.listdir(roi_set_path):
        if '.nii' in roi_file:
            f = os.path.join(roi_set_path, roi_file)
            out = os.path.join(output_dir, roi_file)
            apply_warp_inverse(f, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path,
                               out=out)
    func_space_rois_dict[roi_set_name] = os.path.relpath(output_dir, project_root)
    return func_space_rois_dict


def apply_binary_mask_functional(source, mask, fname='reg_moco.nii.gz'):
    source = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in source]
    out = []
    mask_nii = nibabel.load(mask)
    for run_dir in source:
        src_nii = nibabel.load(run_dir)
        masked_nii = preprocess._apply_binary_mask_3D(src_nii, mask_nii)
        out.append(os.path.join(os.path.dirname(run_dir), 'epi_masked.nii.gz'))
        nibabel.save(masked_nii, out[-1])
    return out


def apply_binary_mask_vol(src_vol, mask):
    subj_root, project_root = _env_setup()
    out = os.path.join(os.path.dirname(src_vol), 'masked_' + os.path.basename(src_vol).split('.')[0] + '.nii.gz')
    src_nii = nibabel.load(src_vol)
    mask_nii = nibabel.load(mask)
    masked_nii = preprocess._apply_binary_mask_3D(src_nii, mask_nii)
    nibabel.save(masked_nii, out)
    return os.path.relpath(out, project_root)


def create_slice_overlays(function_reg_vol, anatomical, reg_contrasts):
    subj_root, project_root = _env_setup()
    out_paths = []
    sig_thresh = float(input("enter significance threshold (std. dev.): "))
    sig_sat = float(input("enter significance saturation point (std. dev.): "))
    if sig_sat < sig_thresh or sig_thresh < 0:
        raise ValueError("Saturation point must be greater than threshold, threshold must be positive")
    if type(reg_contrasts) is not list:
        reg_contrasts = [reg_contrasts]
    for contrast in reg_contrasts:
        out_paths.append(os.path.relpath(analysis.create_slice_maps(function_reg_vol, anatomical, contrast,
                                                                    sig_thresh=sig_thresh, saturation=sig_sat),
                                                                    project_root))
    return out_paths


def get_3d_rep(src: Union[List[str], str], use_topup=True, out_name="3d_epi_rep.nii.gz"):
    """
    :param src:
    :return:
    """
    if use_topup:
        fname = 'f_topup.nii.gz'
    if not use_topup:
        fname = 'f_nordic.nii'
    subj_root, project_root = _env_setup()
    data_arr = []
    sess_dir = os.path.dirname(os.path.dirname(src[0]))
    affine = None
    header = None
    src = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in src]
    for f_path in src:
        nii = nibabel.load(f_path)

        if affine is None:
            header = nii.header
            affine = nii.affine
        data_arr.append(np.median(nii.get_fdata(), axis=3))
        print(data_arr[-1].shape)
    data_arr = np.stack(data_arr, axis=0)
    data_arr = np.median(data_arr, axis=0)
    target_nii = nibabel.Nifti1Image(data_arr, header=header, affine=affine)
    out = os.path.join(sess_dir, out_name)
    nibabel.save(target_nii, out)
    return os.path.relpath(out, project_root)


def convert_to_sphinx_vol_wrap(src, *argv):
    pos = str(argv[0])
    if type(src) is str:
        src = [src]
    paths = []
    for path in src:
        subj_root = os.environ.get('FMRI_WORK_DIR')
        project_root = os.path.join(subj_root, '..', '..')
        os.chdir(project_root)
        out = preprocess.convert_to_sphinx(input_dirs=[path], scan_pos=pos)[0]
        paths.append(os.path.relpath(out, project_root))
    if len(paths) == 1:
        paths = paths[0]
    return paths


def itk_manual(source_vol, template):
    """

    :param source_vol: source vol
    :param template: target vol
    :return: tuple(transform_path, transformed_nii_path)
    """
    subj_root, project_root = _env_setup()
    out_dir = os.path.dirname(source_vol)
    return tuple([os.path.relpath(p, project_root) for p in preprocess._itkSnapManual(template, source_vol, out_dir)])


def _define_contrasts(condition_integerizers, base_index):
    print(
        "Need to define contrasts to preform to continue with analysis. Enter indexes of positive and negative conditions"
        "for each contrast. If multidexed conditions are being used, may enter tuples of indexes. ")
    print("Recall: conditions are mapped as follows ",
          ['condition set ' + str(i) + ' : ' + condition_integerizers[cond_int] for i, cond_int in
           enumerate(condition_integerizers)])
    add_contrast = True
    contrasts = []
    contrast_descriptions = []
    if type(condition_integerizers['0']) is dict:
        contrast_matrix_shape = [len(cond_int) for cond_int in condition_integerizers]
    else:
        contrast_matrix_shape = [len(condition_integerizers)]
    while add_contrast:
        contrast_matrix = np.zeros(contrast_matrix_shape)
        pos_conds = input_control.int_list_input("Enter positive contrast indexes: ")
        neg_conds = input_control.int_list_input("Enter negative contrast indexes: ")
        for pos_cond in pos_conds:
            contrast_matrix[pos_cond] = 1
        for neg_cond in neg_conds:
            contrast_matrix[neg_cond] = -1
        contrast_matrix[base_index] = 0  # the base case should not be considered in contrasts generally
        contrast_matrix[contrast_matrix == 1] /= np.count_nonzero(contrast_matrix == 1)
        contrast_matrix[contrast_matrix == -1] /= np.count_nonzero(contrast_matrix == -1)
        contrast_matrix = contrast_matrix.tolist()
        contrast_descriptions.append(input("Name this contrast: "))
        contrasts.append(contrast_matrix)
        add_contrast = input_control.bool_input("Add another contrast? ")
    return contrasts, contrast_descriptions


def _order_def_from_nilearn_timing(condition_integerizer):
    files_dir = input_control.dir_input("Enter directory containing all nilearn timing files. They will be assigned an"
                                        " order number.")
    files = os.listdir(files_dir)

    cond_int = {v.lower().strip(): k for k, v in condition_integerizer.items()}
    order_def_map = {}
    for order_num, file in files:
        if '.txt' in file:
            order_def = []
            data = pd.read_csv(os.path.join(files_dir, file), sep='\t')
            conds = data['Condition'].tolist()
            durs = data['Duration'].tolist()
            for i, cond_name in enumerate(conds):
                duration = durs[i]
                cond_idx = cond_int[cond_name]
                order_def += [cond_idx] * int(duration)
            order_def_map[str(order_num)] = order_def
            print(file, "assigned order number", order_num)
    return order_def_map


def _create_paradigm():
    subj_root, project_root = _env_setup()
    proj_config = 'config.json'
    with open(proj_config, 'r') as f:
        proj_data = json.load(f)
    para_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, '../..'), 'paradigms')
    print('Constructing new paradigm definition json')
    para_def_dict = {}
    name = input('what is this paradigm called? ')
    para_def_dict['name'] = name
    f_path = os.path.join(para_dir, name + '_experiment_config.json')
    if name in proj_data['paradigms'] or os.path.exists(f_path):
        print("paradigm already exists.")
        with open(f_path, 'r') as f:
            para_def_dict = json.load(f)
            config_file_path = os.path.relpath(os.path.join(para_dir, name + '_experiment_config.json'), project_root)
        if name not in proj_data['data_map']:
            proj_data['data_map'][name] = {subj: {} for subj in proj_data['subjects']}
            proj_data['paradigms'][para_def_dict['name']] = config_file_path
        with open(proj_config, 'w') as f:
            json.dump(proj_data, f, indent=4)
        print("Successfully saved experiment / paradigm configuration json at", config_file_path)
        return para_def_dict, config_file_path

    num_trs = int(input('how many trs are in each run ? '))
    para_def_dict['trs_per_run'] = num_trs
    predefine_orders = input_control.bool_input("Predefine stimuli presentation orders? (highly recommended for your "
                                                "future convenience and reproducibility, unless your orders are in "
                                                "someway stochastic)")
    para_def_dict['is_runtime_defined'] = not predefine_orders
    tr_len = input_control.numeric_input("How many seconds per TR? (Enter -1 if can be variable.)")
    if tr_len > 0:
        para_def_dict['lr_length'] = tr_len

    if predefine_orders:
        block_design = input_control.bool_input('Define stimuli order using standard block definitions? Otherwise must '
                                                'provide a length number of trs list of condition indexes for each '
                                                'order number. ')
    else:
        block_design = False
    para_def_dict['is_block'] = block_design
    # BELOW IS LOCKED TO ONE UNTIL IMPLEMENTATION IS DONE
    # num_order_sets = int(input("How many stimuli condition sets are there: (i.e. how many indexes are needed to specify a specific condition?)"))
    num_order_sets = 1
    num_orders = []
    if predefine_orders:
        for order_set in range(num_order_sets):
            num_orders.append(int(input('how many unique orders are there for condition set ' + str(order_set) + '? ')))
    if block_design:
        para_def_dict['block_length_trs'] = int(input('TRs per block? '))
    else:
        para_def_dict['block_length_trs'] = 1
    if len(num_orders) == 1:
        para_def_dict['num_orders'] = num_orders[0]
    else:
        para_def_dict['num_orders'] = num_orders

    para_def_dict['condition_integerizer'] = []
    para_def_dict['num_conditions'] = []

    for order_set in range(num_order_sets):
        num_conditions = int(input('how many uniques conditions are there in condition set ' + str(order_set) + '? '))
        temp_cond_map = {}
        para_def_dict['num_conditions'].append(num_conditions)
        for i in range(num_conditions):
            temp_cond_map[str(i)] = input('description for condition set' + str(order_set) + ' condition #' + str(i))
        para_def_dict['condition_integerizer'].append(temp_cond_map)

    if num_order_sets == 1:
        para_def_dict['num_conditions'] = para_def_dict['num_conditions'][0]
        para_def_dict['condition_integerizer'] = para_def_dict['condition_integerizer'][0]
    else:
        print("WARNING: n-dimmensional condition sets are not fully stable yet.")
        print(
            "WARNING: When filling in order numbeer maps with multiple order sets - if order definition lengths are not equal, "
            "the two sets will be overlayed by repeating provided sequence, ignoring specified base case conditions. ")
    if len(num_orders) > 1:
        condition_map = para_def_dict['condition_integerizer'][0]
    else:
        condition_map = para_def_dict['condition_integerizer']

    print("Choose the base case condition from from primary condition set (condition set 0)")
    para_def_dict['base_case_condition'] = int(input_control.select_option_input(list(condition_map.keys())))
    para_def_dict['order_number_definitions'] = []

    if predefine_orders:
        load_orders_from_file = False
        if not block_design:
            print("Since you're not generating order defs automatically, recommend loading from file")
            load_orders_from_file = input_control.bool_input("Do you want to do that? ")
        if load_orders_from_file:
            print("Please choose a tsv / csv file with order numbers as columns and a row for each tr, filled with "
                  "condition index values, or tuples of condition index values. If multiple condition sets are being used, "
                  "load an order def file for each.")
            for order_set_idx, num_order in enumerate(num_orders):
                print("Defining order numbers for condition set " + str(order_set_idx))
                from_nilearn = input_control.bool_input("Load order defs from directory of nilearn timing files.")
                if from_nilearn:
                    para_def_dict['block_length_trs'] = 1
                    print('WARNING: set block length to 1 since nilearn orders are explicit')
                    order_def_map = _order_def_from_nilearn_timing(condition_map)
                else:
                    file = input_control.dir_input("Path to file: ")
                    df = pandas.read_csv(file)
                    df.columns = [int(c) for c in df.columns]
                    order_def_map = df.reset_index().to_dict(orient='list')
                para_def_dict['order_number_definitions'].append(order_def_map)
            if len(num_orders) == 1:
                para_def_dict['order_number_definitions'] = para_def_dict['order_number_definitions'][0]
        else:
            for order_set_idx, num_order in enumerate(num_orders):
                order_def_map = {}
                print("Defining order numbers for condition set " + str(order_set_idx))
                for i in range(num_order):
                    order_def_map[str(i)] = input_control.int_list_input(
                        'enter the block order if block design, or event sequence if event related for order number ' + str(
                            i))
                para_def_dict['order_number_definitions'].append(order_def_map)
            if len(num_orders) == 1:
                para_def_dict['order_number_definitions'] = para_def_dict['order_number_definitions'][0]
    else:
        print("Stimuli orders will be defined at runtime. A mapping from the session name to the stimuli order list "
              "will be recorded in this paradigm's \"order_number_definitions\" field")
        para_def_dict['order_number_definitions'] = {}
    good = False
    while not good:
        try:
            contrasts_id, contrast_desc = _define_contrasts(para_def_dict['condition_integerizer'],
                                                            para_def_dict['base_case_condition'])
        except Exception:
            print("Caught exception. Trying again.... ")
            continue
        good = True
    para_def_dict['desired_contrasts'] = contrasts_id
    para_def_dict['contrast_descriptions'] = contrast_desc

    para_def_dict['num_runs_included_in_contrast'] = 0
    config_file_path = os.path.relpath(os.path.join(para_dir, name + '_experiment_config.json'), project_root)
    with open(config_file_path, 'w') as f:
        json.dump(para_def_dict, f, indent=4)
    print("Successfully saved experiment / paradigm configuration json at", config_file_path)
    proj_data['data_map'][name] = {subj: {} for subj in proj_data['subjects']}
    proj_data['paradigms'][para_def_dict['name']] = config_file_path
    with open(proj_config, 'w') as f:
        json.dump(proj_data, f, indent=4)
    return para_def_dict, config_file_path


def create_load_paradigm(add_new_option=True):
    subj_root, project_root = _env_setup()
    proj_config = os.path.abspath(os.path.join(project_root, 'config.json'))
    with open(proj_config, 'r') as f:
        config = json.load(f)
    paradigms = config['paradigms']
    key_integerizer = list(paradigms.keys())
    if add_new_option:
        choices = key_integerizer + ['Define new paradigm...']
    else:
        choices = key_integerizer
    choice = input_control.select_option_input(choices)
    if choice == len(key_integerizer):
        paradigm_def, para_file_path = _create_paradigm()
        config['paradigms'][paradigm_def['name']] = para_file_path
        with open(proj_config, 'w') as f:
            json.dump(config, f, indent=4)
    else:
        para_file_path = paradigms[key_integerizer[choice]]
    return para_file_path


def create_load_ima_order_map(source):
    subj_root, project_root = _env_setup()
    sess_dir = os.path.dirname(source[0])
    if os.path.isfile(source[0]):
        sess_dir = os.path.dirname(sess_dir)
    omap_path = os.path.relpath(os.path.join(sess_dir, 'ima_order_map.json'), project_root)
    if os.path.exists(omap_path):
        if input_control.bool_input("IMA -> order number map already defined for this session. Use existing?"):
            return omap_path
    auto_fill = input_control.bool_input("fill with zeros? (use if runtime order def.)")
    ima_order_map = {}
    one_index_source = input_control.bool_input("Are order numbers 1 indexed (in experiment logs)? ")
    for s in source:
        ima = os.path.basename(os.path.dirname(s)).strip()
        if auto_fill:
            order_num = -1
        else:
            order_num = int(input("Enter order number (as in log) for ima " + ima))
            if one_index_source:
                order_num -= 1
        ima_order_map[ima] = order_num
    with open(omap_path, 'w') as f:
        json.dump(ima_order_map, f, indent=4)
    print("Successfully saved ima to order number mapping json at", omap_path)
    return omap_path


def _design_matrices_from_condition_lists(ima_order_map, condition_names, num_conditions,
                                          base_conditions, sess_dir, sess_name, paradigm_data,
                                          condition_groups, tr_length, run_wise=False, mion=True,
                                          use_cond_groups=True, fir=True):
    design_matrices = []

    print("Stimuli orders are defined manually at runtime for this paradigm. "
          "Please load order csv file for session", sess_dir,  "(rows are trs, cols are IMAs)")
    runlist = os.path.join(sess_dir, "runlist.txt")
    if os.path.exists(runlist):
        print("Reading existing ima runlist at", runlist)
    else:
        runlist = input_control.dir_input("Path to csv file: ")
    ima_order = pd.read_csv(runlist, sep='\t')
    # should be same order as sessions
    for ima in ima_order_map.keys():
        if ima in ima_order:
            c = ima_order[ima]
        else:
            print("IMA", ima, "not found in runlist")
            continue
        clist = c.tolist()
        # check if integers
        if False in [str(item).isnumeric() for item in clist]:
            cond_int = {v.lower().strip(): k for k, v in condition_names.items()}
            int_clist = []
            for cond_name in clist:
                try:
                    int_cond = int(cond_int[cond_name.lower().strip()])
                    int_clist.append(int_cond)
                except KeyError:
                    print("condition descriptor", cond_name,
                          "in the runlist did not match any known condition in the paradigm.")
                    exit(1)
            clist = int_clist
        else:
            clist = [int(c) for c in clist]

        dm = analysis.design_matrix_from_run_list(clist, num_conditions, base_conditions, condition_names,
                                                  condition_groups, tr_length=tr_length, mion=mion,
                                                  reorder=(not run_wise), use_cond_groups=use_cond_groups)
        design_matrices.append(dm)
        if sess_name in paradigm_data['order_number_definitions']:
            paradigm_data['order_number_definitions'][os.path.basename(sess_dir)][ima] = clist
        else:
            paradigm_data['order_number_definitions'][os.path.basename(sess_dir)] = {ima: clist}

    return design_matrices


def get_design_matrices(paradigm_path, ima_order_map_path, source, mion=True, fir=True,
                        run_wise=False, use_cond_groups=True):
    if not (type(source[0]) in (list, tuple)):
        source = [source]
    if not (type(ima_order_map_path) in (list, tuple)):
        ima_order_map_path = [ima_order_map_path]
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)

    base_conditions = [paradigm_data['base_case_condition']]
    condition_names = paradigm_data['condition_integerizer']
    block_length = int(paradigm_data['block_length_trs'])
    num_blocks = int(paradigm_data['trs_per_run'] / block_length)
    num_conditions = int(paradigm_data['num_conditions'])
    is_block_design = paradigm_data['is_block']
    runtime_order_defs = paradigm_data['is_runtime_defined']
    if "tr_length" in paradigm_data:
        tr_length = paradigm_data["tr_length"]
    else:
        tr_length = input_control.numeric_input(
            "No tr length defined in paradigm. What tr length (secs) should we use?")

    condition_groups = None
    if 'condition_groups' in paradigm_data:
        condition_groups = paradigm_data['condition_groups']
    design_matrices = []
    complete_source = []
    for i, session_imas in enumerate(ima_order_map_path):
        with open(session_imas, "r") as f:
            ima_order_map = json.load(f)
        # if this source specfies file path we need to go up two levels to find the session directory
        if os.path.isfile(source[i][0]):
            sess_dir = os.path.dirname(os.path.dirname(source[i][0]))
        else:
            sess_dir = os.path.dirname(source[i][0])
        sess_name = os.path.basename(sess_dir)
        # returns source with file if passed that way
        complete_source += source[i]
        if runtime_order_defs:
            sess_dms = _design_matrices_from_condition_lists(ima_order_map, condition_names, num_conditions,
                                                             base_conditions, sess_dir, sess_name, paradigm_data,
                                                             condition_groups, tr_length, run_wise=run_wise, fir=fir,
                                                             use_cond_groups=use_cond_groups)
            with open(paradigm_path, 'w') as f:
                json.dump(paradigm_data, f, indent=4)

        else:
            sess_dms = []
            for run_dir in source[i]:
                if os.path.isfile(run_dir):
                    ima = os.path.basename(os.path.dirname(run_dir))
                else:
                    ima = os.path.basename(run_dir)
                order_num = ima_order_map[ima]
                order = list(paradigm_data['order_number_definitions'][str(order_num)])
                if is_block_design:
                    sess_dm = analysis.design_matrix_from_order_def(block_length, num_blocks, num_conditions, order,
                                                                    base_conditions,
                                                                    condition_names=condition_names,
                                                                    tr_length=tr_length,
                                                                    mion=mion,
                                                                    fir=fir)
                else:
                    sess_dm = analysis.design_matrix_from_run_list(order, num_conditions, base_conditions,
                                                                   condition_names, condition_groups,
                                                                   tr_length=tr_length, mion=mion)
                sess_dms.append(sess_dm)
        # sess dms is ordered by runlist (chrono)
        for j, sess_dm in enumerate(sess_dms):
            ima = os.path.dirname(source[i][j])
            fix_path = os.path.join(ima, "fixation.csv")
            if os.path.exists(fix_path):
                fix_data = pd.read_csv(fix_path, index_col=0)
                fix_data.columns = ["fixation"]
                fix_data.set_index(sess_dms[j].index, inplace=True)
                sess_dms[j] = pd.concat([sess_dm, fix_data], axis=1)
            else:
                print("WARN: No fixation data provided for session", sess_name, "ima", ima)

        design_matrices += sess_dms
    return design_matrices, complete_source, tr_length


def get_beta_matrix(source, paradigm_path, ima_order_map_path, mion, fname='epi_masked.nii.gz', out_dir=None,
                    run_wise=False, smooth=2., use_cond_groups=True):
    """
    :param source: list of lists of runs for each session
    :param paradigm_path:
    :param ima_order_map_path: list of ima order maps for each session
    :param mion:
    :param fname:
    :return:

    """
    if not (type(source[0]) in (list, tuple)):
        source = [source]
    if not (type(ima_order_map_path) in (list, tuple)):
        ima_order_map_path = [ima_order_map_path]
    subj_root, project_root = _env_setup()
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)

    if mion is None:
        mion = False
    assert type(mion) is bool

    base_conditions = [paradigm_data['base_case_condition']]

    source = [[os.path.join(f, fname) if not os.path.isfile(f) else f for f in s] for s in source]
    design_matrices, complete_source, tr_length = get_design_matrices(paradigm_path, ima_order_map_path, source, mion,
                                                                      run_wise=run_wise,
                                                                      use_cond_groups=use_cond_groups)

    print("using mion: ", mion)
    print('source fname: ', os.path.basename(source[0][0]))
    if out_dir is None:
        out_dir = os.path.dirname(os.path.dirname(source[0][0]))
    print("Using Smoothing Kernel of", smooth)
    glm_path = analysis.nilearn_glm(complete_source, design_matrices, base_conditions,
                                    output_dir=out_dir, fname=fname, mion=mion, tr_length=tr_length, smooth=smooth)
    print("Run Level Beta Coefficient Matrices Created")
    return glm_path


def create_contrast(beta_path, paradigm_path, use_cond_groups=True):
    subj_root, project_root = _env_setup()
    sess_dir = os.path.relpath(os.path.dirname(beta_path), project_root)
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    contrast_descriptions = paradigm_data['contrast_descriptions']
    base_case = paradigm_data["base_case_condition"]
    if 'condition_groups' in paradigm_data and len(paradigm_data['condition_groups']) > 0 and use_cond_groups:
        base_case = None
    contrast_def = list(zip(*paradigm_data['desired_contrasts']))
    contrast_def = np.array([c for i, c in enumerate(contrast_def)], dtype=float)
    contrast_paths = analysis.nilearn_contrasts(beta_path, contrast_def, contrast_descriptions, output_dir=sess_dir)
    print("contrasts created at: " + str(contrast_paths))
    return contrast_paths


def construct_subject_glm(para, mion=True, run_wise=False, use_cond_groups=True, smooth=1.0):
    subj_root, project_root = _env_setup()
    proj_config_path = 'config.json'
    with open(para, 'r') as f:
        paradigm_data = json.load(f)
    subject = os.path.basename(subj_root)
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    para_name = paradigm_data['name']
    sessions_dict = proj_config['data_map'][paradigm_data['name']][subject]

    if len(sessions_dict) == 0:
        print("No Runs Have Been Promoted From Session to Subject Level.")
        return None, None
    sources = []
    ima_order_maps = []
    total_runs = 0
    for key in sessions_dict:
        session_betas = sessions_dict[key]['beta_file']
        imas = sessions_dict[key]['imas_used']
        sources.append([])
        ima_order_maps.append(os.path.join(os.path.dirname(session_betas), "ima_order_map.json"))
        for ima in imas:
            session_path = os.path.join(os.path.dirname(session_betas), str(ima))
            sources[-1].append(session_path)
            total_runs += 1
    total_path = os.path.relpath(os.path.join(subj_root, 'analysis'), project_root)
    glm_path = get_beta_matrix(sources, para, ima_order_maps, mion=mion, fname='epi_masked.nii.gz', out_dir=total_path,
                               run_wise=run_wise, use_cond_groups=use_cond_groups, smooth=smooth)
    print('Constructed subject beta matrix for paradigm', para_name, 'using', len(ima_order_maps), "sessions and",
          total_runs, ' individual runs')
    return glm_path


def get_run_betas(para, mion=True):
    import warnings
    warnings.filterwarnings("ignore")
    subj_root, project_root = _env_setup()
    proj_config_path = 'config.json'
    with open(para, 'r') as f:
        paradigm_data = json.load(f)
    subject = os.path.basename(subj_root)
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    para_name = paradigm_data['name']
    sessions_dict = proj_config['data_map'][paradigm_data['name']][subject]
    base_conditions = [paradigm_data['base_case_condition']]
    condition_groups = paradigm_data['condition_groups']
    condition_names = paradigm_data['condition_integerizer']

    if input_control.bool_input("Add behavioral data?"):
        bp = input_control.dir_input("Enter path to behavior csv file.")
        behavior_key = pd.read_csv(bp)
    else:
        behavior_key = None

    if len(sessions_dict) == 0:
        print("No Runs Have Been Promoted From Session to Subject Level.")
        return None, None
    data_log = {"condition_name": [],
                "condition_integer": [],
                "condition_group": [],
                "beta_path": [],
                "session": [],
                "correct": [],
                "ima": []}
    # creates condition for every stimuli type instead of every stimuli group
    # full_cond_glm_path = construct_subject_glm(para=para, mion=mion, run_wise=False, use_cond_groups=False, smooth=0.5)
    #full_cond_glm_path = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1/subjects/jeeves/analysis/glm_model_169033.pkl"
   # full_cond_glm = pickle.load(open(full_cond_glm_path, "rb"))

    cond2group = {}
    for key in condition_groups.keys():
        for cond_idx in condition_groups[key]["conds"]:
            cond2group[cond_idx] = key

    for key in sessions_dict:
        sources = []
        session_betas = sessions_dict[key]['beta_file']
        imas = sessions_dict[key]['imas_used']
        ima_order_map = os.path.join(os.path.dirname(session_betas), "ima_order_map.json")
        for ima in imas:
            session_path = os.path.join(os.path.dirname(session_betas), str(ima))
            sources.append(session_path)

        glm_path = get_beta_matrix(sources, para, ima_order_map, mion=mion, run_wise=False, use_cond_groups=False, smooth=0.)
        glm = pickle.load(open(glm_path, "rb"))
        imas = sessions_dict[key]['imas_used']
        sess_path = os.path.dirname(sessions_dict[key]['session_net'])
        sess = os.path.basename(sess_path)
        if behavior_key is not None:
            sess_tkns = sess.split("_")
            sess_tkns = sess_tkns[-1]
            syear = int(sess_tkns[:4])
            smonth = int(sess_tkns[4:6])
            sday = int(sess_tkns[6:8])
            sess_behave = behavior_key.loc[(behavior_key["year"] == syear) &
                                           (behavior_key["month"] == smonth) &
                                           (behavior_key["day"] == sday)]
        else:
            sess_behave = None
        for lindex, ima in enumerate(imas):
            if behavior_key is not None:
                ima_behave = sess_behave[sess_behave["ima"] == int(ima)]
            else:
                ima_behave = None
            dm_cols = list(glm.design_matrices_[lindex].columns)

            # get index in dm cols of constant and first drift regressor
            drift_dex = dm_cols.index("drift_1")
            constant_idx = dm_cols.index("constant")

            # get ima glm parameters
            ima_betas = np.array(glm.results_[lindex][0.0].theta)  # <c, v>
            ima_residuals = np.array(glm.results_[lindex][0.0].residuals)  # <t, v>
            np_dm = glm.design_matrices_[lindex].to_numpy().T  # <c, t>
            masker = glm.masker_  # a neat nilearn object that can intelligently mask and unmask your data!

            dm_base_conds = []
            for cond in dm_cols[:drift_dex]:
                cond = str(cond)
                if "delay" in cond:
                    dm_base_conds.append(cond[:-8])
                else:
                    dm_base_conds.append(cond)

            rl_encode = []
            # rl_encode has number of delays for each condition in the dm
            for _, group in itertools.groupby(dm_base_conds):
                rl_encode.append(len(list(group)))

            run_dir = os.path.join(sess_path, str(ima))

            if mion:
                all_conds = list(np_dm.T[:, :drift_dex].sum(axis=0) < 0)  # what conditions are used (nonzero) in this dm??
            else:
                all_conds = list(np_dm.T[:, :drift_dex].sum(axis=0) > 0)

            dm_index = 0
            # collapse dm to discard grey blocks

            # iterate through all conditions in any dm (including delay)
            for cond, delays in enumerate(rl_encode):
                # para index stores the index of the condition in the paradigm
                para_index = cond
                if base_conditions[0] <= cond < constant_idx:
                    para_index += 1
                cond_name = condition_names[str(para_index)]

                # all_conds give the conditions present in this specific dm,
                if all_conds[dm_index]:
                    # this cond exists in this dm
                    mask_dm = np.abs(np_dm) > .2
                    occ_beta_sets = []
                    for delay in range(delays):
                        condition_time_raw = np.nonzero(mask_dm[dm_index])
                        condition_time_indexes = []
                        # adjacent indexes should be grouped
                        prev = -10
                        for i, t_dex in enumerate(condition_time_raw[0].tolist()):
                            if t_dex != prev + 1:
                                condition_time_indexes.append([i])
                            else:
                                condition_time_indexes[-1].append(i)
                            prev = t_dex
                        # multiply coi betas by coi dm and add residuals to create cleaned real time series.
                        predicted_ts = np.outer(ima_betas[dm_index].T, np_dm[dm_index, condition_time_raw])  # <v, t> this is the time series we predict for this stimulus in isolation
                        cleaned_ts = predicted_ts + ima_residuals[condition_time_raw].T  # <v, t> this is what we'll chop up to get our trial data
                        for occ_num, indexes in enumerate(condition_time_indexes):
                            if delay == 0:
                                occ_beta_sets.append([])
                                # create a new entry in the data log if this the first delay.
                                parent_group = cond2group[para_index]
                                data_log["condition_name"].append(cond_name)
                                data_log["condition_group"].append(parent_group)
                                data_log["condition_integer"].append(int(para_index))
                                data_log["session"].append(sess)
                                data_log["ima"].append(ima)
                                if occ_num == 0:
                                    print("Adding", len(condition_time_indexes), "Instances of Class", cond_name, "from Session", sess, "IMA", ima)

                                if behavior_key is not None and "Choice" not in cond_name:
                                    try:
                                        cond_behave_data = ima_behave[ima_behave["condition_name"] == cond_name]["correct"]
                                        correct = int(cond_behave_data.iloc[occ_num])
                                        data_log["correct"].append(correct)
                                    except IndexError:
                                        print("Failed to get behavior information for sess", sess, "ima", ima, "cond",
                                              cond_name)
                                        data_log["correct"].append(0)
                                else:
                                    data_log["correct"].append(0)
                            # create the trial data
                            out_path = os.path.join(run_dir, dm_cols[dm_index] + "_" + str(occ_num) + "_instance_beta.nii.gz")
                            # transform 1d masked voxels into 3d func space
                            trial_data = masker.inverse_transform(cleaned_ts[:, indexes].squeeze().T)
                            # instance_beta = nibabel.Nifti1Image(trial_data, affine=affine, header=header)
                            nibabel.save(trial_data, out_path)
                            occ_beta_sets[occ_num].append(out_path)
                        dm_index += 1

                    data_log["beta_path"] += occ_beta_sets
                else:
                    dm_index += delays

        os.remove(glm_path)
    data_log_out = os.path.join(subj_root, "analysis", para_name + "_stimulus_response_data_key.csv")
    data_log = pandas.DataFrame.from_dict(data_log, orient='columns')
    data_log.to_csv(data_log_out)
    return data_log_out


def surface_run_betas(data_key, ds_t1, t1, manual, auto, paradigm, white_surfs):
    """
    creates and adds registered decode trials and their surface projections to the data key
    """
    reg_epis = []
    right_textures = []
    left_textures = []
    surfs = []
    data = pd.read_csv(data_key)
    data.drop("Unnamed: 0", axis=1, inplace=True)
    
    for row in data.iterrows():
        sources = eval(row[1]["beta_path"])
        try:
            reg_sources = apply_warp(sources, ds_t1, manual, auto)
            surfs, _ = generate_subject_overlays(paradigm, reg_sources, white_surfs, t1, ds_t1)
        except FileNotFoundError:
            print("could not finish", sources)
            right_textures.append(None)
            left_textures.append(None)
            reg_epis.append(None)
            continue
        right_surfs = [s for s in surfs if "rh" in s]
        left_surfs = [s for s in surfs if "lh" in s]
        reg_epis.append(reg_sources)
        right_textures.append(right_surfs)
        left_textures.append(left_surfs)

    data["reg_beta_paths"] = reg_epis
    data["right_surface_textures"] = right_textures
    data["left_surface_textures"] = left_textures
    data.to_csv(data_key)


def beta_from_glm():
    raise NotImplementedError


def promote_session(reg_beta: str, paradigm_path, functional, ima_order_path, *argv):
    """
    This session is marked as complete-ish and will be included by future subject level analysis, and for construction of
    the subject level beta matrix. Basically it's added to the subject config data index.
    :param reg_sontrasts:
    :param paradigm_path:
    :param functional:
    :param ima_order_path:
    :param argv:
    :return:
    """
    subj_root, project_root = _env_setup()
    session_id = str(argv[0])
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    para_name = paradigm_data['name']
    proj_config_path = 'config.json'
    subject = os.path.basename(subj_root)
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    session_net_path = os.path.relpath(os.path.join(subj_root, 'sessions', session_id, 'session_net.json'),
                                       project_root)

    session_info = {'imas_used': [os.path.basename(f) if os.path.isdir(f) else os.path.basename(os.path.dirname(f))
                                  for f in functional],
                    'session_net': session_net_path,
                    'ima_order_map': ima_order_path,
                    'beta_file': reg_beta}
    proj_config['data_map'][para_name][subject][session_id] = session_info
    with open(proj_config_path, 'w') as f:
        json.dump(proj_config, f, indent=4)


def delete_paradigm(paradigm_path):
    subj_root, project_root = _env_setup()
    proj_config_path = 'config.json'
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    para_name = paradigm_data['name']
    remove = input_control.bool_input("WARNING: Paradigm " + para_name
                                      + " and associated data will be permanently removed. Continue? ")
    if remove:
        proj_config['paradigms'].pop(para_name)
        proj_config['data_map'].pop(para_name)

        with open(proj_config_path, 'w') as f:
            json.dump(proj_config, f, indent=4)
    return para_name, remove


def demote_session(paradigm_path, *argv):
    """
    Removes this session from subject config data index. It will no longer be included in future subject level analysis.
    :param paradigm_path:
    :return:
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    subject = os.path.basename(subj_root)
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    session_id = str(argv[0])
    para_name = paradigm_data['name']
    proj_config_path = os.path.join(subj_root, '..', '..', 'config.json')
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    proj_config['data_map'][para_name][subject].pop(session_id)
    with open(proj_config_path, 'w') as f:
        json.dump(proj_config, f, indent=4)


def generate_subject_overlays(para_path, contrast_paths, white_surfs, t1_path, ds_t1_path):
    """
    :param source: The total list of sessions
    :param white_surfs:
    :param t1_path:
    :param ds_t1_path:
    :return:
    """
    subj_root, project_root = _env_setup()
    with open(para_path, 'r') as f:
        para_data = json.load(f)
    os.chdir(project_root)
    out_paths = []

    for contrast_rel_path in contrast_paths:
        contrast_path = os.path.join(project_root, contrast_rel_path)
        for i, hemi in enumerate(['lh', 'rh']):
            # freesurfer has the dumbest path lookup schema so we have to be careful here
            print(subj_root)
            out_path = analysis.create_contrast_surface(os.path.abspath(white_surfs[i]),
                                                        os.path.abspath(contrast_path),
                                                        os.path.abspath(ds_t1_path),
                                                        os.path.abspath(t1_path),
                                                        hemi=hemi, subject_id='.')
            out_paths.append(os.path.relpath(out_path, project_root))
    return out_paths, True


def define_surface_rois(surface_overlays: list, surf_labels: dict):
    """
    expects surface labels to be an roi_set -> surf_label_dir dict
    :param surface_overlays:
    :param surf_labels:
    :return:
    """
    subj_root, project_root = _env_setup()
    preprocess._create_dir_if_needed(subj_root, 'rois')
    roi_set = input("Name the roi set you are creating. e.g. face-areas: ")
    preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois'), roi_set)
    preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set), 'surf_labels')
    print("ROI labeling on surface must be done graphically in freeview. \n"
          "1. Open freeview and load this subjects rh.inflated surface and lh.infalated surface. \n"
          "2. Load surface overlays to guide roi creation"
          "3. create labels for each roi and give an informative name."
          "4. save the rois in \'" + subj_root + "/rois/" + roi_set + "/surf_labels\'")
    surf_labels[roi_set] = os.path.relpath(os.path.join(subj_root, 'rois', roi_set, 'surf_labels'), project_root)
    input('done? ')
    return surf_labels


def manual_volume_rois(ds_t1: str, paradigm_complete: bool, ds_vol_rois: dict):
    """
    Define a set of rois on in the downsampled volume space that functional data is registered to
    :param ds_t1:
    :param vol_rois:
    :return:
    """
    if paradigm_complete is False:
        print("Please create subject contrasts for at least one paradigm before defining functional ROIs.")
        return None
    subj_root, project_root = _env_setup()
    preprocess._create_dir_if_needed(subj_root, 'rois')
    roi_set = input("Name the roi set you are creating. e.g. face-areas: ")
    preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois'), roi_set)
    preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set), 'ds_vol_rois')
    print("Manual ROI labeling on volumes must be done graphically in freeview. \n"
          "1. Open freeview and load " + os.path.abspath(ds_t1) + "\n"
                                                                  "2. Load desired contrast volumes to guide roi creation"
                                                                  "3. create masks for each roi and give an informative name."
                                                                  "4. save the rois in \'" + os.path.join(subj_root,
                                                                                                          "rois",
                                                                                                          roi_set,
                                                                                                          "ds_vol_rois") + "\'")
    ds_vol_rois[roi_set] = os.path.relpath(os.path.join(subj_root, 'rois', roi_set, 'ds_vol_rois'), project_root)
    input('done? ')
    return ds_vol_rois


def automatic_volume_rois(paradigm_file, paradigm_contrasts: List[str], ds_vol_rois):
    subj_root, project_root = _env_setup()
    preprocess._create_dir_if_needed(subj_root, 'rois')
    with open(paradigm_file, 'r') as f:
        paradigm_data = json.load(f)
    contrast_names = paradigm_data['contrast_descriptions']
    roi_set_name = input("Enter name for roi set: ").strip()
    max_rois = int(input("Enter desired max number of rois"))
    min_rois = int(input("Enter desired min number of rois"))
    preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois'), roi_set_name)
    out_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set_name), 'ds_vol_rois')
    contrasts_touse = []
    print("Select contrast (s) to generate pos / neg ROIs from: ")
    add = True
    while add:
        choice = input_control.select_option_input(contrast_names)
        path = paradigm_contrasts[choice]
        contrasts_touse.append(path)
        add = input_control.bool_input("add another contrast?")
    out_dir = analysis.get_auto_roi_masks(contrasts_touse, out_dir, max_rois=max_rois, min_rois=min_rois)
    ds_vol_rois[roi_set_name] = os.path.relpath(out_dir, project_root)
    return ds_vol_rois


def surf_labels_to_vol_mask(surf_labels: dict, white_surface, t1, ds_t1, vol_rois: dict) -> dict:
    """
    Takes dictionary keyed on roi set names and returns the vol roi set dict amended to include new vol generated from surface labels.
    :param vol_rois:
    :param surf_labels:
    :param white_surface:
    :param t1:
    :param ds_t1:
    :return:
    """
    subj_root, project_root = _env_setup()
    roi_options = list(surf_labels.keys())
    choice = input_control.select_option_input(roi_options)
    roi_set_name = roi_options[choice]
    roi_surf_dir = surf_labels[roi_set_name]
    hemis = ['lh', 'rh']
    out_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set_name), 'vol_rois')
    for hemi in hemis:
        analysis.labels_to_roi_mask(roi_surf_dir, hemi, out_dir, t1, subject_id='.')
    vol_rois[roi_set_name] = os.path.relpath(out_dir, project_root)
    return vol_rois


def load_t1_data():
    subj_root, project_root = _env_setup()
    cur_t1_path = input_control.dir_input("enter path to t1: ")
    cur_t1_mask_path = input_control.dir_input("enter path to t1 mask: ")
    t1_name = os.path.basename(cur_t1_path)
    t1_mask_name = os.path.basename(cur_t1_mask_path)
    t1_proj_path = os.path.relpath(os.path.join(subj_root, 'mri', t1_name))
    t1_mask_proj_path = os.path.relpath(os.path.join(subj_root, 'mri', t1_mask_name))
    try:
        shutil.copy(cur_t1_path, t1_proj_path)
    except shutil.SameFileError:
        pass
    try:
        shutil.copy(cur_t1_mask_path, t1_mask_proj_path)
    except shutil.SameFileError:
        pass
    return t1_proj_path, t1_mask_proj_path


def load_white_surfs():
    subj_root, project_root = _env_setup()
    surfs = []
    for hemi in ['left', 'right']:
        cur_surf_path = input_control.dir_input("enter path to " + hemi + " surf: ")
        surf_name = os.path.basename(cur_surf_path)
        surf_proj_path = os.path.relpath(os.path.join(subj_root, 'surf', surf_name), project_root)
        if not os.path.exists(surf_proj_path):
            shutil.copy(cur_surf_path, surf_proj_path)
        surfs.append(surf_proj_path)
    return surfs


def motion_correction_wrapper(source, targets, moco_is_nonlinear=None, fname="f_topup_sphinx.nii.gz"):
    """
    :param source:
    :param targets:
    :param fname:
    :param moco_is_nonlinear
    :return:

    """
    source = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in source]
    out = [os.path.join(os.path.dirname(f), 'moco.nii.gz') for f in source]
    subj_root, project_root = _env_setup()
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    if moco_is_nonlinear is None:
        moco_is_nonlinear = config["reg_settings"]["nonlinear_moco"]

    best_disp = 9999999
    besp_disp_idx = -1
    if isinstance(targets, str):
        targets = [targets]
    for i, t in enumerate(targets):
        preprocess.motion_correction(source, t, check_rms=False, fname=fname, outname='temp_moco.nii.gz')
        disp = 0
        for run_dir in source:
            disp_file = os.path.join(os.path.dirname(run_dir), 'temp_moco.nii.gz_abs.rms')
            disp_vec = np.loadtxt(disp_file)
            run_disp = np.mean(disp_vec.flatten())
            disp += run_disp
        disp /= len(source)
        if disp < best_disp:
            best_disp = disp
            besp_disp_idx = i
            for s in source:
                sd = os.path.dirname(s)
                moco_file = os.path.join(sd, 'temp_moco.nii.gz')
                rename_moco = os.path.join(sd, 'moco.nii.gz')
                os.rename(moco_file, rename_moco)
    print("Best average displacement: ", str(best_disp))
    print("Target chosen: " + targets[besp_disp_idx])
    ref = targets[besp_disp_idx]
    if moco_is_nonlinear:
        print("running final nonlinear warp...")
        args = []
        for s in source:
            sd = os.path.dirname(s)
            mc = os.path.join(sd, 'moco.nii.gz')
            args.append((mc, ref, mc))
        with multiprocessing.Pool() as p:
            p.starmap(preprocess.nonlinear_moco, args)

    # update the target file to be an average of moco-ed images
    get_3d_rep(out, out_name=os.path.basename(ref))

    return out


def nordic_correction_wrapper(functional_dirs, fname='f.nii.gz'):
    """
    Wrapper for nordic denoising. See https://github.com/SteenMoeller/NORDIC_Raw
    and https://www.nature.com/articles/s41467-021-25431-8.

    """
    subj_root,project_root = _env_setup()
    functional_dirs = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in functional_dirs]
    full_func_dirs = [os.path.abspath(fdir) for fdir in functional_dirs]
    out = preprocess.NORDIC(full_func_dirs, 'f_nordic')
    out = [os.path.relpath(opath, project_root) for opath in out]
    return out


def topup_wrapper(functional_dirs, use_topup, fname='f_nordic.nii'):
    if use_topup:
        in_paths = [os.path.join(f, fname) if not os.path.isfile(f) else f for f in functional_dirs]
        image_1_path = input_control.dir_input('Enter the path to the first spin echo image.')
        image_1_enc = input_control.str_list_input('What phase encoding is this image? (HF, FH, RL, LR)')[0]
        image_2_path = input_control.dir_input('Enter the path to the second spin echo image.')
        image_2_enc = input_control.str_list_input('What phase encoding is this image? (HF, FH, RL, LR)')[0]
        number_images = input_control.int_input('How many images do you want to use topup with? (Recommended 2-3)')
        func_enc = input_control.str_list_input('What phase encoding are the functional images in? (HF, FH, RL, LR)')[0]
        out = preprocess.topup(in_paths, image_1_path, image_2_path, image_1_enc, image_2_enc, number_images, func_enc)
    else:
        print('Skipping Topup')
        out = functional_dirs

    return out


def time_series_order_vs_all_functional(functional_dirs, ima_order_data, paradigm_data, target_condition, output_dir,
                                        fname='epi_masked', pre_onset_blocks=1, post_offset_blocks=3):
    """
    Wrapper for analysis.get_condition_time_series_comparison that grabs needed input parameters
    from a paradigm json file.
    :param functional_dirs:
    :param ima_order_data:
    :param paradigm_data:
    :param target_condition:
    :param output_dir:
    :param fname:
    :param pre_onset_blocks:
    :param post_offset_blocks:
    :return:
    """
    subj_root, project_root = _env_setup()
    conditions = paradigm_data['condition_integerizer']
    order_def = paradigm_data['order_number_definitions']
    block_length = paradigm_data['block_length_trs']
    pre_onset_trs = pre_onset_blocks * block_length
    post_offset_trs = post_offset_blocks * block_length
    para_name = paradigm_data['name']
    output = os.path.relpath(
        os.path.join(output_dir,
                     para_name + '_condition_' + conditions[str(target_condition)] + '_timecourse_comparison.nii'),
        project_root)
    return analysis.get_condition_time_series_comparision(functional_dirs, block_length, ima_order_data, order_def,
                                                          target_condition, output, fname=fname,
                                                          pre_onset_trs=pre_onset_trs, post_offset_trs=post_offset_trs)


def get_vol_rois_time_series(vol_rois: dict, ts_dict: dict, ds_t1_path):
    """
    designed to work in context of control flow infrastructure
    Takes path to directory containing in volume roi niis (as binary masks), then creates a template functional file
    for a desrired target condition(s), extracts the ts for the roi, and plots.
    :param ds_t1:
    :param fine_transform:
    :param manual_transform:
    :param ts_dict:
    :param vol_rois:
    :param functional_dirs:
    :param ima_order_file:
    :param paradigm_file:
    :return:
    """
    options = list(vol_rois.keys())
    subj_root, project_root = _env_setup()
    subj = os.path.basename(subj_root)
    config_path = 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("which paradigm was used to define this roi set? ")
    para_options = list(config['paradigms'].keys())
    para_id = input_control.select_option_input(para_options)
    paradigm = para_options[para_id]
    paradigm_file = config['paradigms'][paradigm]
    with open(paradigm_file, 'r') as f:
        paradigm_data = json.load(f)
    sessions_dict = config['data_map'][paradigm_data['name']][subj]

    print('choose which roi set to generate time series comparisons for:')
    choice_idx = input_control.select_option_input(options)
    roi_set_name = options[choice_idx]
    roi_dir = vol_rois[roi_set_name]
    conditions = paradigm_data['condition_integerizer']
    print(
        'select condition(s) of interest for time series comparison (i.e. conditions where divergence from noise is expected)')
    coi = []
    cond_opt = list(conditions.values())
    idxs = list(conditions.keys())

    pre_block = 1
    post_block = 3

    ts_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set_name), 'time_series')

    while True:
        choice_idx = input_control.select_option_input(cond_opt)
        coi.append(int(idxs[choice_idx]))
        if not input_control.bool_input("add another condition of interest?"):
            break
    roi_paths = [os.path.join(roi_dir, f) for f in os.listdir(roi_dir) if '.nii' in f]

    # plt init
    fig, axs = plt.subplots(len(roi_paths))

    cond_descs = [conditions[str(i)] for i in coi]
    roi_total_ts = [None for i in range(len(roi_paths))]

    for condition in coi:
        for session_id in sessions_dict:
            ima_order_file = sessions_dict[session_id]['ima_order_map']
            with open(ima_order_file, 'r') as f:
                ima_order_data = json.load(f)
            sess_dir = os.path.dirname(sessions_dict[session_id]['session_net'])
            manual_transform = os.path.join(sess_dir, 'itkManual.txt')
            fine_forward_transform = os.path.join(sess_dir, 'Composite.h5')
            functional_dirs = [os.path.join(sess_dir, str(ima)) for ima in sessions_dict[session_id]['imas_used']]
            ts_func_path = time_series_order_vs_all_functional(functional_dirs, ima_order_data, paradigm_data,
                                                               condition, sess_dir,
                                                               fname='epi_masked.nii', pre_onset_blocks=pre_block,
                                                               post_offset_blocks=post_block)
            ts_func_path = apply_warp(ts_func_path, ds_t1_path, manual_transform, fine_forward_transform, type_code=3,
                                      dim=3)
            ts_data = nibabel.load(ts_func_path).get_fdata()
            ts_data = ts_data - np.mean(ts_data, axis=3)[:, :, :, None]

            for i, roi_path in enumerate(roi_paths):
                roi_nii = nibabel.load(roi_path)
                roi_name = os.path.basename(roi_path).split('.')[0]
                roi_data = roi_nii.get_fdata()
                # block order here is to ensure proper plotting, 1 indicates target condition present
                ts, _, colors = analysis._get_roi_time_course(roi_data, 1, ts_data, axs[i],
                                                              block_length=paradigm_data['block_length_trs'],
                                                              block_order=[0] * pre_block + [1] + [0] * post_block,
                                                              colors=plt.get_cmap('ocean'),
                                                              roi_name=roi_name,
                                                              ts_name=conditions[str(condition)])
                if roi_total_ts[i] is None:
                    roi_total_ts[i] = ts
                else:
                    roi_total_ts[i] += ts
        for i, roi_path in enumerate(roi_paths):
            roi_name = os.path.basename(roi_path).split('.')[0]
            ts_path = os.path.join(ts_dir, roi_name + '_'.join(cond_descs) + '_condition_ts.npy')
            ts = roi_total_ts[i] / len(sessions_dict)
            np.save(ts_path, ts)
    fig.suptitle(' '.join(cond_descs) + ' condition vs all time series')
    fig.set_size_inches(8, 1.5 * len(roi_paths))
    fig.tight_layout()
    fig.savefig(os.path.join(subj_root, 'rois', roi_set_name, 'ts_plot.png'))
    ts_dict[roi_set_name] = os.path.relpath(ts_dir, project_root)
    plt.legend(loc='upper right')
    plt.show()
    return ts_dict


def binary_masks_from_int_atlas(atlas: str,
                                desired_indices: Union[None, List[Union[int, Tuple[int]]]] = None, index_names=None,
                                out_dir=None):
    """
    takes a integer indexed atlas nifti path and converts it into seperate binary masks. Creates mask for each index in
    desired_indices, or for all indices in the atlas if desired_indices is None. If a desired index is a tuple these will
    be combined into a singular maskOptional index names allows user to define outut names for each roi
    :param out_dir:
    :param index_names:
    :param indexed_mask:
    :param desired_indices:
    :return:
    """
    subj_root, project_root = _env_setup()
    atlas_nii = nibabel.load(atlas)
    atlas_data = np.array(atlas_nii.get_fdata())
    out_paths = []
    if out_dir is None:
        out_dir = os.path.dirname(atlas)
    elif not os.path.isdir(out_dir):
        raise ValueError
    if desired_indices is None:
        desired_indices = list(np.unique(atlas_data))
    if index_names is not None and len(index_names) != len(desired_indices):
        raise ValueError("if names are provided must be same length as desired index list")
    for i, idx in enumerate(desired_indices):
        if type(idx) is tuple:
            mask = np.zeros_like(atlas_data)
            for lidx in idx:
                mask += (atlas_data == lidx).astype(int)
        else:
            mask = (atlas_data == idx).astype(int)
        if index_names is None:
            name = str(idx)
        else:
            name = index_names[i]
        mask_nii = nibabel.Nifti1Image(mask, header=atlas_nii.header, affine=atlas_nii.affine)
        out_path = os.path.join(out_dir, name + '.nii')
        nibabel.save(mask_nii, out_path)
        out_paths.append(os.path.relpath(out_path, project_root))
    return out_paths
