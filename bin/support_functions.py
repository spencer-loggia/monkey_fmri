
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
    from_dicom = input_control.bool_input("load new session dicoms? (otherwise must import niftis from folder)")
    f_dir = preprocess._create_dir_if_needed(subj_root, 'sessions')
    if from_dicom:
        dicom_dir = input_control.dir_input('Enter directory containing DICOMS from scans we are trying to analyze')
        run_numbers = input_control.int_list_input(
            "Enter whitespace separated list of valid epi run IMA numbers. (e.g. '2 4 5')")
        SOURCE = unpack.unpack_run_list(dicom_dir, f_dir, run_numbers, session_id, 'f')
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
        SOURCE = [os.path.join(sess_target_dir, f) for f in os.listdir(sess_target_dir) if f.isnumeric()]
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
    source_dir = os.path.dirname(inpath)
    name = os.path.basename(inpath)
    out_name = 'ds_' + name.split('.')[0] + '.nii'
    if out_dir is None:
        output_dir = preprocess._create_dir_if_needed(subj_root, 'mri')
    else:
        output_dir = out_dir
    output = os.path.join(output_dir, out_name)
    preprocess.create_low_res_anatomical(source_dir, name, output, factor=factor, affine_scale=affine_scale, resample=resample)
    return output


def downsample_vol_rois(roi_dict, ds_roi_dict, factor=2, affine_scale=1, resample='nearest', output_dir=None):
    options = list(roi_dict)
    choice = input_control.select_option_input(options)
    roi_set_name = options[choice]
    roi_set_path = roi_dict[roi_set_name]
    subj_root = os.environ.get('FMRI_WORK_DIR')
    if output_dir is None:
        output_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set_name), 'ds_roi_vols')
    for roi_file in os.listdir(roi_set_path):
        if '.nii' in roi_file:
            out = os.path.join(output_dir, roi_file)
            preprocess.create_low_res_anatomical(roi_set_path, roi_file, out, factor=factor, affine_scale=affine_scale, resample=resample)
    ds_roi_dict[roi_set_name] = output_dir
    return ds_roi_dict


def downsample_vol_rois_cmdline_wrap(dir, factor=2, affine_scale=1, resample='nearest', output_dir=None):
    roi_dict = {os.path.basename(dir): dir}
    out_dict = {}
    return downsample_vol_rois(roi_dict, out_dict, factor=factor, affine_scale=affine_scale, resample=resample, output_dir=output_dir)


def coreg_wrapper(source_space_vol_path, target_space_vol_path):
    """

    :param source_space_vol_path:
    :param target_space_vol_path:
    :return: forward_transform_path, inverse_transform_path
    """
    out = os.path.join(os.path.dirname(source_space_vol_path), 'coreg_3df.nii')
    return preprocess.antsCoReg(target_space_vol_path,
                                source_space_vol_path,
                                outP=out,
                                ltrns=['Affine', 'SyN'],
                                n_jobs=2)


def apply_warp(source, vol_in_target_space, forward_gross_transform_path=None, fine_transform_path=None, type_code=0, dim=3):
    if type(source) is not list:
        source = [source]
    transforms = [fine_transform_path, forward_gross_transform_path]
    transforms = [t for t in transforms if t is not None]
    to_invert = [False] * len(transforms)
    out_paths = []
    for s in source:
        out = os.path.join(os.path.dirname(s), 'reg_' + os.path.basename(s))
        preprocess.antsApplyTransforms(s, vol_in_target_space, out, transforms, 'Linear',
                                       img_type_code=type_code, dim=dim, invertTrans=to_invert)
        out_paths.append(out)
    if len(out_paths) == 1:
        out_paths = out_paths[0]
    return out_paths


def apply_warp_4d(source, vol_in_target_space, forward_gross_transform_path=None, fine_transform_path=None):
    return apply_warp(source, vol_in_target_space, forward_gross_transform_path=forward_gross_transform_path,
                      fine_transform_path=fine_transform_path, type_code=3, dim=3)


def apply_warp_inverse(source, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path, out=None):
    """
    TODO: TEST TRANSFORM ORDRING!!
    :param source:
    :param vol_in_target_space:
    :param forward_gross_transform_path:
    :param reverse_fine_transform_path:
    :param out:
    :return:
    """
    if out is None:
        out = os.path.join(os.path.dirname(vol_in_target_space), 'inverse_trans_' + os.path.basename(source))
    inverse_transforms = [reverse_fine_transform_path, forward_gross_transform_path]
    to_invert = [False, True]
    preprocess.antsApplyTransforms(source, vol_in_target_space, out, inverse_transforms, 'Linear',
                                   img_type_code=0, invertTrans=to_invert)
    return out


def apply_warp_inverse_vol_roi_dir(ds_vol_roi_dict, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path, func_space_rois_dict):
    """
    Slated for removal
    :return:
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
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
    func_space_rois_dict[roi_set_name] = output_dir
    return func_space_rois_dict

def apply_binary_mask_functional(source, mask, fname='moco.nii.gz'):
    for run_dir in source:
        src_nii = nibabel.load(os.path.join(run_dir, fname))
        mask_nii = nibabel.load(mask)
        masked_nii = preprocess._apply_binary_mask_3D(src_nii, mask_nii)
        nibabel.save(masked_nii, os.path.join(run_dir, 'epi_masked.nii'))
    return source


def apply_binary_mask_vol(src_vol, mask):
    out = os.path.join(os.path.dirname(src_vol), 'masked_vol.nii')
    src_nii = nibabel.load(src_vol)
    mask_nii = nibabel.load(mask)
    masked_nii = preprocess._apply_binary_mask_3D(src_nii, mask_nii)
    nibabel.save(masked_nii, out)
    return out


def create_slice_overlays(function_reg_vol, anatomical, reg_contrasts):
    out_paths = []
    sig_thresh = float(input("enter significance threshold (std. dev.): "))
    sig_sat = float(input("enter significance saturation point (std. dev.): "))
    if sig_sat < sig_thresh or sig_thresh < 0:
        raise ValueError("Saturation point must be greater than threshold, threshold must be positive")
    if type(reg_contrasts) is not list:
        reg_contrasts = [reg_contrasts]
    for contrast in reg_contrasts:
        out_paths.append(analysis.create_slice_maps(function_reg_vol, anatomical, contrast,
                                                    sig_thresh=sig_thresh, saturation=sig_sat))
    return out_paths


def get_3d_rep(src: Union[List[str], str]):
    from_epi = input_control.bool_input("Select 3d rep as frame from epis? Otherwise select volume file (in epi space)")
    if from_epi:
        return preprocess.get_middle_frame(src)
    else:
        return input_control.dir_input("Select volume file in epi space (probably called t2.nii.gz)")


def convert_to_sphinx_vol_wrap(src):
    dirname = os.path.dirname(src)
    fname = os.path.basename(src)
    out = preprocess.convert_to_sphinx(input_dirs=[dirname], fname=fname)[0]
    out_path = os.path.join(out, fname.split('.')[0] + '_sphinx.nii')
    return out_path


def itk_manual(source_vol, template):
    """

    :param source_vol: source vol
    :param template: target vol
    :return: tuple(transform_path, transformed_nii_path)
    """
    out_dir = os.path.dirname(source_vol)
    return preprocess._itkSnapManual(template, source_vol, out_dir)


def _define_contrasts(condition_integerizer, num_conditions):
    print(
        "Need to define contrasts to preform to continue with analysis. Enter length number conditions vector defining "
        "contribution for each condition. For example [0, 1, 0, -1] defines a direct comparison between a positive "
        "condition number 2 and negative condition number 4")
    print("Recall: conditions are mapped as follows ", condition_integerizer)
    add_contrast = True
    contrasts = []
    contrast_descriptions = []
    while add_contrast:
        contrast = input_control.int_list_input("Enter vector defining constrast: ")
        if len(contrast) != num_conditions:
            print('contrast vector must have length equal to the number of conditions.')
            continue
        contrast_descriptions.append(input("Name this contrast: "))
        contrasts.append(contrast)
        add_contrast = input_control.bool_input("Add another contrast? ")
    return contrasts, contrast_descriptions


def _create_paradigm():
    subj_root = os.environ.get('FMRI_WORK_DIR')
    proj_config = os.path.join(subj_root, '..', 'config.json')
    with open(proj_config, 'r') as f:
        proj_data = json.load(f)
    para_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, '../..'), 'paradigms')
    print('Constructing new paradigm definition json')
    para_def_dict = {}
    name = input('what is this paradigm called? ')
    para_def_dict['name'] = name
    num_trs = int(input('how many trs are in each run ? '))
    para_def_dict['trs_per_run'] = num_trs
    block_design = input_control.bool_input('is this paradigm a block design? ')
    para_def_dict['is_block'] = block_design
    num_orders = int(input('how many unique orders are there? '))
    if block_design:
        para_def_dict['block_length_trs'] = int(input('TRs per block? '))
    else:
        para_def_dict['block_length_trs'] = 1
    para_def_dict['num_orders'] = num_orders
    num_conditions = int(input('how many uniques conditions are there? '))
    para_def_dict['num_conditions'] = num_conditions
    condition_map = {}
    for i in range(num_conditions):
        condition_map[str(i)] = input('description for condition #' + str(i))
    para_def_dict['condition_integerizer'] = condition_map
    order_def_map = {}
    for i in range(num_orders):
        order_def_map[str(i)] = input_control.int_list_input(
            'enter the block order if block design, or event sequence if event related.')
    para_def_dict['order_number_definitions'] = order_def_map
    contrasts_id, contrast_desc = _define_contrasts(condition_map, num_conditions)
    para_def_dict['desired_contrasts'] = contrasts_id
    para_def_dict['contrast_descriptions'] = contrast_desc
    para_def_dict['num_runs_included_in_contrast'] = 0
    config_file_path = os.path.join(para_dir, name + '_experiment_config.json')
    with open(config_file_path, 'w') as f:
        json.dump(para_def_dict, f, indent=4)
    proj_data['data_map'][name] = {subj: {} for subj in proj_data['subjects']}
    with open(proj_config, 'w') as f:
        json.dump(proj_data, f, indent=4)
    print("Successfully saved experiment / paradigm configuration json at", config_file_path)
    return para_def_dict, config_file_path


def create_load_paradigm(add_new_option=True):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    proj_config = os.path.abspath(os.path.join(subj_root, '../..', 'config.json'))
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
    sess_dir = os.path.dirname(source[0])
    omap_path = os.path.join(sess_dir, 'ima_order_map.json')
    if os.path.exists(omap_path):
        if input_control.bool_input("IMA -> order number map already defined for this session. Use existing?"):
            return omap_path
    ima_order_map = {}
    one_index_source = input_control.bool_input("Are order numbers 1 indexed (in experiment logs)? ")
    for s in source:
        ima = os.path.basename(s).strip()
        order_num = int(input("Enter order number (as in log) for ima" + ima))
        if one_index_source:
            order_num -= 1
        ima_order_map[ima] = order_num
    with open(omap_path, 'w') as f:
        json.dump(ima_order_map, f, indent=4)
    print("Successfully saved ima to order number mapping json at", omap_path)
    return omap_path


def create_beta_matrix(source, paradigm_path, ima_order_map_path, mion, fname='epi_masked.nii'):
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    with open(ima_order_map_path, 'r') as f:
        ima_order_map = json.load(f)
    if mion is None:
        mion = False
    assert type(mion) is bool
    sess_dir = os.path.dirname(source[0])
    design_matrices = []
    for run_dir in source:
        ima = os.path.basename(run_dir).strip()
        order_num = ima_order_map[ima]
        block_length = int(paradigm_data['block_length_trs'])
        num_blocks = int(paradigm_data['trs_per_run'] / block_length)
        num_conditions = int(paradigm_data['num_conditions'])
        order = list(paradigm_data['order_number_definitions'][str(order_num)])
        design_matrix = analysis.design_matrix_from_order_def(block_length, num_blocks, num_conditions, order,
                                                              convolve=False)
        design_matrices.append(design_matrix)
    print("using mion: ", mion)
    print('source fname: ', fname)
    _, beta_path = analysis.get_beta_coefficent_matrix(source, design_matrices, output_dir=sess_dir, fname=fname,
                                                       mion=mion, use_python_mp=True, auto_conv=True, tr=3)
    print("Beta Coefficient Matrix created at: " + str(beta_path))
    return beta_path


def create_contrast(beta_path, paradigm_path):
    sess_dir = os.path.dirname(beta_path)
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
        contrast_def = np.array(paradigm_data['desired_contrasts'], dtype=float).T
    contrast_descriptions = paradigm_data['contrast_descriptions']

    _, contrast_paths = analysis.create_contrasts(beta_path, contrast_def, contrast_descriptions, output_dir=sess_dir)
    print("contrasts created at: " + str(contrast_paths))
    return contrast_paths


def construct_complete_beta_file():
    subj_root = os.environ.get('FMRI_WORK_DIR')
    para = create_load_paradigm(add_new_option=False)
    with open(para, 'r') as f:
        paradigm_data = json.load(f)
    proj_config_path = os.path.join(subj_root, '../..', 'config.json')
    subject = os.path.basename(subj_root)
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    para_name = paradigm_data['name']
    sessions_dict = proj_config['data_map'][paradigm_data['name']][subject]
    total_runs = 0
    total_affine = None
    total_beta = None
    header = None
    for key in sessions_dict:
        session_betas = sessions_dict[key]['beta_file']
        num_runs = len(sessions_dict[key]['imas_used'])
        total_runs += num_runs
        beta = nibabel.load(session_betas)
        if total_beta is None:
            total_beta = np.array(beta.get_fdata())
            total_affine = beta.affine
            header = beta.header
        else:
            total_beta += np.array(beta.get_fdata())
            total_affine += beta.affine

    total_path = os.path.join(subj_root, 'analysis', para_name + '_subject_betas.nii')
    subject_betas = total_beta / total_runs
    subject_affine = total_affine / total_runs
    total_nii = nibabel.Nifti1Image(subject_betas, affine=subject_affine, header=header)
    nibabel.save(total_nii, total_path)
    print('Constructed subject beta matrix for paradigm', para_name, 'using ', total_runs, ' individual runs')
    return total_nii, total_path


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
    session_id = str(argv[0])
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    para_name = paradigm_data['name']
    proj_config_path = os.path.join(subj_root, '..', '..', 'config.json')
    subject = os.path.basename(subj_root)
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    session_net_path = os.path.join(subj_root, 'sessions', session_id, 'session_net.json')

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


def generate_subject_overlays(source, white_surfs, t1_path, ds_t1_path):
    """
    :param source: The total list of sessions
    :param white_surfs:
    :param t1_path:
    :param ds_t1_path:
    :return:
    """
    para = create_load_paradigm(add_new_option=False)
    with open(para, 'r') as f:
        para_data = json.load(f)
    para_name = para_data['name']
    subj_root = os.environ.get('FMRI_WORK_DIR')
    os.chdir(subj_root)
    full_subject = input_control.bool_input('Create Full Subject Contrast? (otherwise specify individual session)')
    if full_subject:
        contrasts = [os.path.join(subj_root, 'analysis', f) for f in os.listdir(os.path.join(subj_root, 'analysis')) if 'contrast.nii' in f and para_name in f]
    else:
        session_id = ''
        valid_ids = [os.path.basename(os.path.dirname(s)) for s in source]
        while session_id not in valid_ids:
            session_id = input('enter session id: ')
        contrasts = [os.path.join(subj_root, 'sessions', session_id, f) for f in os.listdir(os.path.join(subj_root, 'sessions', session_id)) if
                     'contrast.nii' in f and 'reg' in f]
    if len(contrasts) == 0:
        print('no contrasts have been created')
    for contrast_path in contrasts:
        for i, hemi in enumerate(['lh', 'rh']):
            # freesurfer has the dumbest path lookup schema so we have to be careful here
            print(subj_root)
            analysis.create_contrast_surface(white_surfs[i],
                                             contrast_path,
                                             ds_t1_path,
                                             t1_path,
                                             hemi=hemi, subject_id='.')


def define_surface_rois(surface_overlays: list, surf_labels: dict):
    """
    expects surface labels to be an roi_set -> surf_label_dir dict
    :param surface_overlays:
    :param surf_labels:
    :return:
    """
    subj_root = os.environ.get('FMRI_WORK_DIR')
    preprocess._create_dir_if_needed(subj_root, 'rois')
    roi_set = input("Name the roi set you are creating. e.g. face-areas: ")
    preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois'), roi_set)
    preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set), 'surf_labels')
    print("ROI labeling on surface must be done graphically in freeview. \n"
          "1. Open freeview and load this subjects rh.inflated surface and lh.infalated surface. \n"
          "2. Load surface overlays to guide roi creation"
          "3. create labels for each roi and give an informative name."
          "4. save the rois in \'" + subj_root + "/rois/" + roi_set + "/surf_labels\'")
    surf_labels[roi_set] = os.path.join(subj_root, 'rois', roi_set, 'surf_labels')
    input('done? ')
    return surf_labels


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
    roi_options = list(surf_labels.keys())
    choice = input_control.select_option_input(roi_options)
    roi_set_name = roi_options[choice]
    roi_surf_dir = surf_labels[roi_set_name]
    hemis = ['lh', 'rh']
    out_dir = preprocess._create_dir_if_needed(os.path.join(subj_root, 'rois', roi_set_name), 'vol_rois')
    for hemi in hemis:
        analysis.labels_to_roi_mask(roi_surf_dir, hemi, out_dir, t1, subject_id='.')
    vol_rois[roi_set_name] = out_dir
    return vol_rois


def load_t1_data():
    subj_root = os.environ.get('FMRI_WORK_DIR')
    cur_t1_path = input_control.dir_input("enter path to t1: ")
    cur_t1_mask_path = input_control.dir_input("enter path to t1 mask: ")
    t1_name = os.path.basename(cur_t1_path)
    t1_mask_name = os.path.basename(cur_t1_mask_path)
    t1_proj_path = os.path.join(subj_root, 'mri', t1_name)
    t1_mask_proj_path = os.path.join(subj_root, 'mri', t1_mask_name)
    shutil.copy(cur_t1_path, t1_proj_path)
    shutil.copy(cur_t1_mask_path, t1_mask_proj_path)
    return t1_proj_path, t1_mask_proj_path


def load_white_surfs():
    subj_root = os.environ.get('FMRI_WORK_DIR')
    surfs = []
    for hemi in ['left', 'right']:
        cur_surf_path = input_control.dir_input("enter path to " + hemi + " surf: ")
        surf_name = os.path.basename(cur_surf_path)
        surf_proj_path = os.path.join(subj_root, 'surf', surf_name)
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
    conditions = paradigm_data['condition_integerizer']
    order_def = paradigm_data['order_number_definitions']
    block_length = paradigm_data['block_length_trs']
    pre_onset_trs = pre_onset_blocks * block_length
    post_offset_trs = post_offset_blocks * block_length
    para_name = paradigm_data['name']
    output = os.path.join(output_dir, para_name + '_condition_' + conditions[str(target_condition)] + '_timecourse_comparison.nii')
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
    subj = os.path.basename(subj_root)
    config_path = os.path.join(subj_root, '..', '..', 'config.json')
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
    ts_dict[roi_set_name] = ts_dir
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
        out_paths.append(out_path)
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
