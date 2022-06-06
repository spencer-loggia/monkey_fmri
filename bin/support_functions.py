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

import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color='brgcmyk')

# subject support function must be operating with working directory appropriately set


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


def get_epis(*argv):
    session_id = argv[0]
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    from_dicom = input_control.bool_input("load new session dicoms? (otherwise must import niftis from folder)")
    f_dir = preprocess._create_dir_if_needed(subj_root, 'sessions')
    if from_dicom:
        dicom_dir = input_control.dir_input('Enter directory containing DICOMS from scans we are trying to analyze')
        run_numbers = input_control.int_list_input(
            "Enter whitespace separated list of valid epi run IMA numbers. (e.g. '2 4 5')")
        SOURCE = unpack.unpack_run_list(dicom_dir, f_dir, run_numbers, session_id, 'f')
        SOURCE = [os.path.relpath(s, project_root) for s in SOURCE]
        sessDir = os.path.dirname(SOURCE[0])
        print("Created the following functional run directories: \n ",
              SOURCE,
              " \n each containing a raw functional file 'f.nii.gz'")
    else:
        nifti_source = input_control.dir_input(
            "Enter path to session directory containing run subdirectories containing niftis")
        nifti_name = input("Enter name of nifti files to transfer (should be unprocessed versions")
        sess_target_dir = os.path.join(f_dir, str(session_id))
        if nifti_source != sess_target_dir:
            shutil.copytree(nifti_source, sess_target_dir, ignore=include_patterns(nifti_name))
        else:
            preprocess._create_dir_if_needed(f_dir, str(session_id))
        SOURCE = [os.path.relpath(os.path.join(sess_target_dir, f), project_root) for f in os.listdir(sess_target_dir)
                  if f.isnumeric()]
        if nifti_name != 'f.nii.gz':
            for run_dir in SOURCE:
                raw_nifti = os.path.join(run_dir, nifti_name)
                target_nifti = os.path.join(run_dir, 'f.nii.gz')
                shutil.move(raw_nifti, target_nifti)
    check = input_control.bool_input("Check functional time series lengths? ")
    if check:
        expected = int(input("What is the expected number of trs? "))
        SOURCE = preprocess.check_time_series_length(SOURCE, fname='f.nii.gz', expected_length=expected)
    return SOURCE


def downsample_anatomical(inpath, factor=2, out_dir=None, affine_scale=1, resample='interpolate'):
    """
    Resample should be either 'interpolate' or 'nearest'
    :param inpath:
    :param factor:
    :param out_dir:
    :param resample:
    :return:
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    source_dir = os.path.dirname(inpath)
    name = os.path.basename(inpath)
    out_name = 'ds_' + name.split('.')[0] + '.nii'
    if out_dir is None:
        output_dir = preprocess._create_dir_if_needed(subj_root, 'mri')
    else:
        output_dir = out_dir
    output = os.path.relpath(os.path.join(output_dir, out_name), project_root)
    preprocess.create_low_res_anatomical(source_dir, name, output,
                                         factor=factor, affine_scale=affine_scale, resample=resample)
    return output


def downsample_vol_rois(roi_dict, ds_roi_dict, factor=2, affine_scale=1, resample='nearest', output_dir=None):
    options = list(roi_dict)
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    choice = input_control.select_option_input(options)
    roi_set_name = options[choice]
    roi_set_path = roi_dict[roi_set_name]
    if output_dir is None:
        output_dir = os.path.relpath(preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set_name), 'ds_roi_vols'), project_root)
    for roi_file in os.listdir(roi_set_path):
        if '.nii' in roi_file:
            out = os.path.join(output_dir, roi_file)
            preprocess.create_low_res_anatomical(roi_set_path, roi_file, out, factor=factor, affine_scale=affine_scale, resample=resample)
    ds_roi_dict[roi_set_name] = output_dir
    return ds_roi_dict


def downsample_vol_rois_cmdline_wrap(dir, factor=2, affine_scale=1, resample='nearest', output_dir=None):
    """
    So we can use outside of project structure
    """
    roi_dict = {os.path.basename(dir): dir}
    out_dict = {}
    return downsample_vol_rois(roi_dict, out_dict, factor=factor, affine_scale=affine_scale, resample=resample, output_dir=output_dir)


def coreg_wrapper(source_space_vol_path, target_space_vol_path):
    """

    :param source_space_vol_path:
    :param target_space_vol_path:
    :return: forward_transform_path, inverse_transform_path
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    out = os.path.join(os.path.dirname(source_space_vol_path), 'coreg_3df.nii')
    return tuple([os.path.relpath(f, project_root) for f in preprocess.antsCoReg(os.path.abspath(target_space_vol_path),
                                                                           os.path.abspath(source_space_vol_path),
                                                                           outP=os.path.abspath(out),
                                                                           ltrns=['Affine', 'SyN'],
                                                                           n_jobs=2)])


def apply_warp(source, vol_in_target_space, forward_gross_transform_path=None, fine_transform_path=None, type_code=0, dim=3, interp='Linear'):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    if type(source) is not list:
        source = [source]
    transforms = [fine_transform_path, forward_gross_transform_path]
    transforms = [t for t in transforms if t is not None]
    to_invert = [False] * len(transforms)
    out_paths = []
    for s in source:
        out = os.path.join(os.path.dirname(s), 'reg_' + os.path.basename(s))
        preprocess.antsApplyTransforms(s, vol_in_target_space, out, transforms, interp,
                                       img_type_code=type_code, dim=dim, invertTrans=to_invert)
        out_paths.append(os.path.relpath(out, project_root))
    if len(out_paths) == 1:
        out_paths = out_paths[0]
    return out_paths


def apply_warp_4d(source, vol_in_target_space, forward_gross_transform_path=None, fine_transform_path=None):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    return os.path.relpath(apply_warp(source, vol_in_target_space, forward_gross_transform_path=forward_gross_transform_path,
                      fine_transform_path=fine_transform_path, type_code=3, dim=3), project_root)


def apply_warp_inverse(source, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path, out=None):
    """
    :param source:
    :param vol_in_target_space:
    :param forward_gross_transform_path:
    :param reverse_fine_transform_path:
    :param out:
    :return:
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    if out is None:
        out = os.path.join(os.path.dirname(vol_in_target_space), 'inverse_trans_' + os.path.basename(source))
    inverse_transforms = [os.path.abspath(forward_gross_transform_path), os.path.abspath(reverse_fine_transform_path)]
    to_invert = [True, False]
    preprocess.antsApplyTransforms(os.path.abspath(source), os.path.abspath(vol_in_target_space), os.path.abspath(out), inverse_transforms, 'Linear',
                                   img_type_code=0, invertTrans=to_invert)
    return os.path.relpath(out, project_root)


def apply_warp_inverse_vol_roi_dir(ds_vol_roi_dict, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path, func_space_rois_dict):
    """
    Slated for removal
    :return:
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    options = list(ds_vol_roi_dict)
    choice = input_control.select_option_input(options)
    roi_set_name = options[choice]
    roi_set_path = ds_vol_roi_dict[roi_set_name]
    output_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois'), 'func_space_rois')
    for roi_file in os.listdir(roi_set_path):
        if '.nii' in roi_file:
            f = os.path.join(roi_set_path, roi_file)
            out = os.path.join(output_dir, roi_file)
            apply_warp_inverse(f, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path, out=out)
    func_space_rois_dict[roi_set_name] = os.path.relpath(output_dir, project_root)
    return func_space_rois_dict


def apply_binary_mask_functional(source, mask, fname='moco.nii.gz'):
    for run_dir in source:
        src_nii = nibabel.load(os.path.join(run_dir, fname))
        mask_nii = nibabel.load(mask)
        masked_nii = preprocess._apply_binary_mask_3D(src_nii, mask_nii)
        nibabel.save(masked_nii, os.path.join(run_dir, 'epi_masked.nii'))
    return source


def apply_binary_mask_vol(src_vol, mask):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    out = os.path.join(os.path.dirname(src_vol), 'masked_vol.nii')
    src_nii = nibabel.load(src_vol)
    mask_nii = nibabel.load(mask)
    masked_nii = preprocess._apply_binary_mask_3D(src_nii, mask_nii)
    nibabel.save(masked_nii, out)
    return os.path.relpath(out, project_root)


def create_slice_overlays(function_reg_vol, anatomical, reg_contrasts):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    out_paths = []
    sig_thresh = float(input("enter significance threshold (std. dev.): "))
    sig_sat = float(input("enter significance saturation point (std. dev.): "))
    if sig_sat < sig_thresh or sig_thresh < 0:
        raise ValueError("Saturation point must be greater than threshold, threshold must be positive")
    if type(reg_contrasts) is not list:
        reg_contrasts = [reg_contrasts]
    for contrast in reg_contrasts:
        out_paths.append(os.path.relpath(analysis.create_slice_maps(function_reg_vol, anatomical, contrast,
                                                    sig_thresh=sig_thresh, saturation=sig_sat), project_root))
    return out_paths


def get_3d_rep(src: Union[List[str], str]):
    """
    :param src:
    :return:
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    return [os.path.relpath(f, project_root) for f in preprocess.sample_frames(src, 5)]


def convert_to_sphinx_vol_wrap(src):
    if type(src) is str:
        src = [src]
    paths = []
    for path in src:
        subj_root = os.environ.get('FMRI_WORK_DIR')
        project_root = os.path.join(subj_root, '..', '..')
        os.chdir(project_root)
        dirname = os.path.dirname(path)
        fname = os.path.basename(path)
        out = preprocess.convert_to_sphinx(input_dirs=[dirname], fname=fname)[0]
        out_path = os.path.join(out, fname.split('.')[0] + '_sphinx.nii')
        paths.append(os.path.relpath(out_path, project_root))
    return paths


def itk_manual(source_vol, template):
    """

    :param source_vol: source vol
    :param template: target vol
    :return: tuple(transform_path, transformed_nii_path)
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    out_dir = os.path.dirname(source_vol)
    return tuple([os.path.relpath(p, project_root) for p in preprocess._itkSnapManual(template, source_vol, out_dir)])


def _define_contrasts(condition_integerizers, base_index):
    print(
        "Need to define contrasts to preform to continue with analysis. Enter indexes of positive and negative conditions"
        "for each contrast. If multidexed conditions are being used, may enter tuples of indexes. ")
    print("Recall: conditions are mapped as follows ", ['condition set ' + str(i) + ' : ' + condition_integerizers[cond_int] for i, cond_int in enumerate(condition_integerizers)])
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


def _create_paradigm():
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    proj_config = 'config.json'
    with open(proj_config, 'r') as f:
        proj_data = json.load(f)
    para_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, '../..'), 'paradigms')
    print('Constructing new paradigm definition json')
    para_def_dict = {}
    name = input('what is this paradigm called? ')
    para_def_dict['name'] = name
    num_trs = int(input('how many trs are in each run ? '))
    para_def_dict['trs_per_run'] = num_trs
    predefine_orders = input_control.bool_input("Predefine stimuli presentation orders? (highly recommended for your "
                                                "future convenience and reproducibility, unless your orders are in "
                                                "someway stochastic)")
    para_def_dict['is_runtime_defined'] = not predefine_orders

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
        print("WARNING: When filling in order numbeer maps with multiple order sets - if order definition lengths are not equal, "
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
                        'enter the block order if block design, or event sequence if event related for order number ' + str(i))
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
            contrasts_id, contrast_desc = _define_contrasts(para_def_dict['condition_integerizer'], para_def_dict['base_case_condition'])
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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    sess_dir = os.path.dirname(source[0])
    omap_path = os.path.relpath(os.path.join(sess_dir, 'ima_order_map.json'), project_root)
    if os.path.exists(omap_path):
        if input_control.bool_input("IMA -> order number map already defined for this session. Use existing?"):
            return omap_path
    ima_order_map = {}
    one_index_source = input_control.bool_input("Are order numbers 1 indexed (in experiment logs)? ")
    for s in source:
        ima = os.path.basename(s).strip()
        order_num = int(input("Enter order number (as in log) for ima " + ima))
        if one_index_source:
            order_num -= 1
        ima_order_map[ima] = order_num
    with open(omap_path, 'w') as f:
        json.dump(ima_order_map, f, indent=4)
    print("Successfully saved ima to order number mapping json at", omap_path)
    return omap_path


def get_beta_matrix(source, paradigm_path, ima_order_map_path, mion, fname='epi_masked.nii'):
    """
    TODO: Still needs some work with regards to nd design matrices.
    :param source:
    :param paradigm_path:
    :param ima_order_map_path:
    :param mion:
    :param fname:
    :return:
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    with open(ima_order_map_path, 'r') as f:
        ima_order_map = json.load(f)
    if mion is None:
        mion = False
    assert type(mion) is bool
    sess_dir = os.path.dirname(source[0])
    design_matrices = []
    base_conditions = [paradigm_data['base_case_condition']]
    condition_names = paradigm_data['condition_integerizer']
    block_length = int(paradigm_data['block_length_trs'])
    num_blocks = int(paradigm_data['trs_per_run'] / block_length)
    num_conditions = int(paradigm_data['num_conditions'])
    is_block_design = paradigm_data['is_block']
    runtime_order_defs = paradigm_data['is_runtime_defined']
    sess_name = os.path.basename(sess_dir)
    if runtime_order_defs:
        print("Stimuli orders are defined manually at runtime for this paradigm. "
              "Please load order csv file (rows are trs, cols are IMAs)")
        ima_order = pd.read_csv(input_control.dir_input("Path to csv file: "))
        for c_name, c in ima_order.iteritems():
            clist = c.tolist()
            # check if integers
            if False in [str(item).isnumeric() for item in clist]:
                cond_int = {v: k for k, v in condition_names.items()}
                clist = [int(cond_int[cond_name]) for cond_name in clist]
            design_matrices.append(analysis.design_matrix_from_run_list(clist,
                                                                        num_conditions,
                                                                        base_conditions))
            if sess_name in paradigm_data['order_number_definitions']:
                paradigm_data['order_number_definitions'][os.path.basename(sess_dir)][c_name] = clist
            else:
                paradigm_data['order_number_definitions'][os.path.basename(sess_dir)] = {c_name: clist}
            with open(paradigm_path, 'w') as f:
                json.dump(paradigm_data, f)

    else:
        for run_dir in source:
            ima = os.path.basename(run_dir).strip()
            order_num = ima_order_map[ima]
            order = list(paradigm_data['order_number_definitions'][str(order_num)])
            if is_block_design:
                design_matrix = analysis.design_matrix_from_order_def(block_length, num_blocks, num_conditions, order,
                                                                  base_conditions)
            else:
                design_matrix = analysis.design_matrix_from_run_list(order, num_conditions, base_conditions)
            design_matrices.append(design_matrix)

    print("using mion: ", mion)
    print('source fname: ', fname)
    _, beta_path, _ = analysis.get_beta_coefficent_matrix(source, design_matrices, base_conditions, output_dir=sess_dir,
                                                          fname=fname, mion=mion, use_python_mp=False, auto_conv=False,
                                                          tr=3)
    print("Beta Coefficient Matrix created at: " + str(beta_path))
    return os.path.relpath(beta_path, project_root)


def create_contrast(beta_path, paradigm_path):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    sess_dir = os.path.relpath(os.path.dirname(beta_path), project_root)
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
        contrast_def = np.array(paradigm_data['desired_contrasts'], dtype=float).T
    contrast_descriptions = paradigm_data['contrast_descriptions']

    _, contrast_paths = analysis.create_contrasts(beta_path, contrast_def, contrast_descriptions, output_dir=sess_dir)
    print("contrasts created at: " + str(contrast_paths))
    return contrast_paths


def construct_complete_beta_file(para):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    proj_config_path = 'config.json'
    with open(para, 'r') as f:
        paradigm_data = json.load(f)
    subject = os.path.basename(subj_root)
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    para_name = paradigm_data['name']
    sessions_dict = proj_config['data_map'][paradigm_data['name']][subject]
    total_runs = 0
    total_affine = None
    total_beta = None
    header = None
    if len(sessions_dict) == 0:
        print("No Betas Have Been Promoted From Session to Subject Level.")
        return None, None
    for key in sessions_dict:
        session_betas = sessions_dict[key]['beta_file']
        num_runs = len(sessions_dict[key]['imas_used'])
        total_runs += num_runs
        beta = nibabel.load(session_betas)
        if total_beta is None:
            total_beta = (np.array(beta.get_fdata()) * num_runs)
            total_affine = (beta.affine * num_runs)
            header = beta.header
        else:
            total_beta += (np.array(beta.get_fdata()) * num_runs)
            total_affine += (beta.affine * num_runs)

    total_path = os.path.relpath(os.path.join(subj_root, 'analysis', para_name + '_subject_betas.nii'), project_root)
    subject_betas = total_beta / total_runs
    subject_affine = total_affine / total_runs
    total_nii = nibabel.Nifti1Image(subject_betas, affine=subject_affine, header=header)
    nibabel.save(total_nii, total_path)
    print('Constructed subject beta matrix for paradigm', para_name, 'using ', total_runs, ' individual runs')
    return total_path


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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    session_id = str(argv[0])
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    para_name = paradigm_data['name']
    proj_config_path = 'config.json'
    subject = os.path.basename(subj_root)
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    session_net_path = os.path.relpath(os.path.join(subj_root, 'sessions', session_id, 'session_net.json'), project_root)

    session_info = {'imas_used': [os.path.basename(f) for f in functional],
                    'session_net': session_net_path,
                    'ima_order_map': ima_order_path,
                    'beta_file': reg_beta}
    proj_config['data_map'][para_name][subject][session_id] = session_info
    with open(proj_config_path, 'w') as f:
        json.dump(proj_config, f, indent=4)


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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    preprocess._create_dir_if_needed(subj_root, 'rois')
    roi_set = input("Name the roi set you are creating. e.g. face-areas: ")
    preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois'), roi_set)
    preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set), 'ds_vol_rois')
    print("Manual ROI labeling on volumes must be done graphically in freeview. \n"
          "1. Open freeview and load " + os.path.abspath(ds_t1) + "\n"
          "2. Load desired contrast volumes to guide roi creation"
          "3. create masks for each roi and give an informative name."
          "4. save the rois in \'" + os.path.join(subj_root, "rois", roi_set, "ds_vol_rois") + "\'")
    ds_vol_rois[roi_set] = os.path.relpath(os.path.join(subj_root, 'rois', roi_set, 'ds_vol_rois'), project_root)
    input('done? ')
    return ds_vol_rois


def automatic_volume_rois(paradigm_file, paradigm_contrasts: List[str], ds_vol_rois):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    cur_t1_path = input_control.dir_input("enter path to t1: ")
    cur_t1_mask_path = input_control.dir_input("enter path to t1 mask: ")
    t1_name = os.path.basename(cur_t1_path)
    t1_mask_name = os.path.basename(cur_t1_mask_path)
    t1_proj_path = os.path.relpath(os.path.join(subj_root, 'mri', t1_name))
    t1_mask_proj_path = os.path.relpath(os.path.join(subj_root, 'mri', t1_mask_name))
    shutil.copy(cur_t1_path, t1_proj_path)
    shutil.copy(cur_t1_mask_path, t1_mask_proj_path)
    return t1_proj_path, t1_mask_proj_path


def load_white_surfs():
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    surfs = []
    for hemi in ['left', 'right']:
        cur_surf_path = input_control.dir_input("enter path to " + hemi + " surf: ")
        surf_name = os.path.basename(cur_surf_path)
        surf_proj_path = os.path.relpath(os.path.join(subj_root, 'surf', surf_name), project_root)
        shutil.copy(cur_surf_path, surf_proj_path)
        surfs.append(surf_proj_path)
    return surfs


def order_corrected_functional(functional_dirs, ima_order_data, paradigm_data, output, deconv_weight, fname='epi_masked') -> np.ndarray:
    """
    This method is not used by current control flow module, but kept cause it kinda cool.
    The defualt maximal dynamic regressor does produce a deconv npy file to be used with this
    By order correcting the batch averaged function via deconvolution, we can compare all block simultaneously,
    giving a better sense of the true nature of the data.
    Create a averaged 4d time series where stimuli presentation order is corrected for,
    by de-convolving, rearranging, stacking, then re-convolving.
    :return:
    """
    conditions = paradigm_data['condition_integerizer']
    order_def = paradigm_data['order_number_definitions']
    block_length = paradigm_data['block_length_trs']
    w, h, d, _ = nibabel.load(os.path.join(functional_dirs[0], fname)).get_fdata().shape
    num_conditions = len(conditions)
    corrected_arr = torch.zeros((w, h, d, block_length, num_conditions))
    weight = torch.zeros(len(conditions))
    deconv_weight = torch.from_numpy(deconv_weight)
    deconv = torch.nn.Conv1d(kernel_size=deconv_weight.shape[0], padding=torch.floor(deconv_weight.shape[0] / 2), in_channels=1, out_channels=1, bias=False)
    deconv.weight = torch.nn.Parameter(deconv_weight, requires_grad=False)
    for func_dir in functional_dirs:
        ima = os.path.basename(func_dir)
        order_num = ima_order_data[ima]
        order = order_def[str(order_num)]
        func_nii = nibabel.load(os.path.join(func_dir, fname))
        func_data = func_nii.get_fdata()
        shape = func_data.shape
        func_data = torch.from_numpy(func_data.reshape(w * h * d, 1, -1))  # reshape to voxels (batch), channels, time for use with torch conv operator
        func_data = deconv(func_data)
        func_data = func_data.reshape(list(shape[:3]) + [block_length, len(order)]) # so we can easily index into individual blocks
        for block, condition in enumerate(order):
            weight[condition] += 1
            corrected_arr[:, :, :, :, condition] += func_data[:, :, :, :, block]
    corrected_arr = (corrected_arr.T / weight[:, None, None, None, None]).T
    corrected_arr = corrected_arr.reshape((w, h, d, block_length * num_conditions)) # reshape to real functional
    corrected_arr = corrected_arr.numpy()
    corrected_nii = nibabel.Nifti1Image(corrected_arr, affine=func_nii.affine, header=func_nii.header)
    nibabel.save(corrected_nii, output)
    return output


def motion_correction_wrapper(source, targets, fname='f_sphinx.nii'):
    """
    :param source:
    :param targets:
    :param fname:
    :return:
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    best_disp = 9999999
    besp_disp_idx = -1
    for i, t in enumerate(targets):
        preprocess.motion_correction(source, t, check_rms=False, fname='f_sphinx.nii', outname='temp_moco.nii.gz')
        sess_dir = os.path.dirname(source[0])
        disp = 0
        for run_dir in source:
            disp_file = os.path.join(run_dir, 'temp_moco.nii.gz_abs.rms')
            disp_vec = np.loadtxt(disp_file)
            run_disp = np.mean(disp_vec.flatten())
            disp += run_disp
        disp /= len(source)
        if disp < best_disp:
            best_disp = disp
            besp_disp_idx = i
            for s in source:
                moco_file = os.path.join(s, 'temp_moco.nii.gz')
                rename_moco = os.path.join(s, 'moco.nii.gz')
                os.rename(moco_file, rename_moco)
    print("Best average displacement: ", str(best_disp))
    print("Target chosen: " + targets[besp_disp_idx])
    return source, targets[besp_disp_idx]


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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
    conditions = paradigm_data['condition_integerizer']
    order_def = paradigm_data['order_number_definitions']
    block_length = paradigm_data['block_length_trs']
    pre_onset_trs = pre_onset_blocks * block_length
    post_offset_trs = post_offset_blocks * block_length
    para_name = paradigm_data['name']
    output = os.path.relpath(
        os.path.join(output_dir, para_name + '_condition_' + conditions[str(target_condition)] + '_timecourse_comparison.nii'),
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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
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
    print('select condition(s) of interest for time series comparison (i.e. conditions where divergence from noise is expected)')
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

    #plt init
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
            fine_forward_transform = os.path.join(sess_dir, 'antsRegComposite.h5')
            functional_dirs = [os.path.join(sess_dir, str(ima)) for ima in sessions_dict[session_id]['imas_used']]
            ts_func_path = time_series_order_vs_all_functional(functional_dirs, ima_order_data, paradigm_data, condition, sess_dir,
                                                               fname='epi_masked.nii', pre_onset_blocks=pre_block, post_offset_blocks=post_block)
            ts_func_path = apply_warp(ts_func_path, ds_t1_path, manual_transform, fine_forward_transform, type_code=3, dim=3)
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
            ts_path = os.path.join(ts_dir,  roi_name + '_'.join(cond_descs) + '_condition_ts.npy')
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
                                desired_indices:Union[None, List[Union[int, Tuple[int]]]] = None, index_names = None, out_dir = None):
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
    subj_root = os.environ.get('FMRI_WORK_DIR')
    project_root = os.path.join(subj_root, '..', '..')
    os.chdir(project_root)
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


def segment_contrast_time_course(contrast_files, functional_dirs, ima_order_map, hrf_file, deconvolution_file,  paradigm_file, fname='epi_masked.nii'):
    out_paths = []
    with open(paradigm_file, 'r') as f:
        para_data = json.load(f)
    with open(ima_order_map, 'r') as f:
        ima_data = json.load(f)
    block_length = para_data['block_length_trs']
    block_orders = para_data['order_number_definitions']
    num_conditions = len(para_data['condition_integerizer'])
    hrf = np.load(hrf_file)
    deconv = np.load(deconvolution_file)
    correct_order = False
    if len(para_data['order_number_definitions']) > 1:
        correct_order = True
    sess_dir = os.path.dirname(functional_dirs[0])
    if correct_order:
        func = os.path.join(sess_dir, 'avg_func.nii')
        func = order_corrected_functional(functional_dirs, ima_data, para_data, func, deconv, fname=fname)
        order_def = list(range(num_conditions))
    else:
        imas = [None]
        while False in [ima in ima_data and ima_data[ima] == ima_data[imas[0]] for ima in imas]:
            imas = input_control.int_list_input("enter IMA number of run(s) to use for timeseries. Must use the same order number. ")
            imas = [str(ima) for ima in imas]
        if len(imas) > 1:
            func = os.path.join(sess_dir, 'avg_func.nii')
            analysis.average_functional_data([f for f in functional_dirs if os.path.basename(f) in imas],
                                             fname=fname,
                                             output=func)
        else:
            func = os.path.join(sess_dir, imas[0], fname)

        order_def = block_orders[str(ima_data[imas[0]])]

    if type(contrast_files) is not list:
        contrast_files = [contrast_files]

    all_contrast = input_control.bool_input("Create auto rois for all contrasts?")
    if not all_contrast:
        print('select contrast to use:')
        names = para_data['contrast_descriptions']
        option = input_control.select_option_input(names)
        contrast_files = [c for c in contrast_files if names[option] in c]
    for contrast in contrast_files:
        out_c = analysis.segment_get_time_course(contrast, func, block_length, order_def)
        path = os.path.join(os.path.dirname(contrast), 'cleaned_' + os.path.basename(contrast))
        nibabel.save(out_c, path)
        out_paths.append(path)
    plt.show()
    return out_paths
