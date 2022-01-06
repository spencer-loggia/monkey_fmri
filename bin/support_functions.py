from typing import List

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

# subject support function must be operating with working directory appropriately set
subj_root = os.path.abspath(os.getcwd())

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
        sess_target_dir = preprocess._create_dir_if_needed(f_dir, str(session_id))
        shutil.copytree(nifti_source, sess_target_dir, ignore=include_patterns(nifti_name))
        SOURCE = [f for f in os.listdir(sess_target_dir) if f.isnumeric()]
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


def downsample_anatomical(inpath):
    source_dir = os.path.dirname(inpath)
    name = os.path.basename(inpath)
    output_dir = preprocess._create_dir_if_needed(subj_root, 'mri')
    output = os.path.join(output_dir, 'ds_t1.nii.gz')
    return preprocess.create_low_res_anatomical(source_dir, name, output)


def coreg_wrapper(source_space_vol_path, target_space_vol_path):
    out = os.path.dirname(target_space_vol_path)
    return preprocess.antsCoReg(target_space_vol_path,
                                source_space_vol_path,
                                outP=out,
                                ltrns=['Affine', 'SyN'],
                                n_jobs=2)


def apply_warp(source, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path):
    if type(source) is not list:
        source = [source]
    transforms = [forward_gross_transform_path, reverse_fine_transform_path]
    to_invert = [False, False]
    out_paths = []
    for s in source:
        out = os.path.join(os.path.dirname(s), 'reg_' + os.path.basename(s))
        preprocess.antsApplyTransforms(source, vol_in_target_space, out, transforms, 'Linear',
                                       img_type_code=0, invertTrans=to_invert)
    return out_paths


def apply_warp_inverse(source, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path):
    out = os.path.join(os.path.dirname(vol_in_target_space), 'inverse_trans_' + os.path.basename(source))
    inverse_transforms = [reverse_fine_transform_path, forward_gross_transform_path]
    to_invert = [False, True]
    preprocess.antsApplyTransforms(source, vol_in_target_space, out, inverse_transforms, 'Linear',
                                   img_type_code=0, invertTrans=to_invert)
    return out


def apply_binary_mask_functional(source, mask, fname='moco.nii.gz'):
    sess_dir = os.path.dirname(source[0])
    for run_dir in source:
        src_nii = nibabel.load(os.path.join(run_dir, fname))
        mask_nii = nibabel.load(mask)
        masked_nii = preprocess._apply_binary_mask_3D(src_nii, mask_nii)
        nibabel.save(masked_nii, os.path.join(sess_dir, 'epi_masked.nii'))
    return source


def apply_binary_mask_vol(src_vol, mask):
    out = os.path.join(os.path.dirname(src_vol), 'masked_vol.nii')
    src_nii = nibabel.load(src_vol)
    mask_nii = nibabel.load(mask)
    masked_nii = preprocess._apply_binary_mask_3D(src_nii, mask_nii)
    nibabel.save(masked_nii, out)
    return out


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
    return contrasts, contrast_descriptions


def _create_paradigm():
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
    print("Successfully saved experiment / paradigm configuration json at", config_file_path)
    return para_def_dict, config_file_path


def create_load_paradigm():
    proj_config = os.path.join(subj_root, '../..', 'config.json')
    with open(proj_config, 'r') as f:
        config = json.load(f)
    paradigms = config['paradigms']
    key_integerizer = list(paradigms.keys())
    choice = input_control.select_option_input(key_integerizer + ['Define new paradigm...'])
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
    for s in source:
        ima = os.path.basename(s).strip()
        order_num = int(input("Enter order number for ima" + ima))
        ima_order_map[ima] = order_num
    with open(omap_path, 'w') as f:
        json.dump(ima_order_map, f, indent=4)
    print("Successfully saved ima to order number mapping json at", omap_path)
    return omap_path


def create_load_session(ds_t1, ds_t1_mask, ds_t1_masked):
    pass


def create_project_config():
    project_config = {}
    project_config['project_name'] = input('project name: ')
    project_config['paradigms'] = []
    project_config['data_map'] = {}  # subjects -> paradigms -> session_ids / run nums


def create_contrast(source, paradigm_path, ima_order_map_path, fname='epi_masked.nii'):
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    with open(ima_order_map_path, 'r') as f:
        ima_order_map = json.load(f)
    contrast_def = np.array(paradigm_data['desired_contrasts'], dtype=float).T
    contrast_descriptions = paradigm_data['contrast_descriptions']
    num_runs = len(source)
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
    contrast_imgs = analysis.intra_subject_contrast(source, design_matrices, contrast_def, contrast_descriptions,
                                                    output_dir=sess_dir, fname=fname,
                                                    mode='maximal_dynamic', use_python_mp=True)
    contrast_paths = [os.path.join(sess_dir, s + '_contrast.nii') for s in contrast_descriptions]
    print("contrasts created at: " + str(contrast_paths))
    return contrast_paths


def add_to_subject_contrast(reg_sontrasts: List[str], paradigm_path, *argv):
    session_id = str(argv[0])
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    para_name = paradigm_data['name']
    for con_path in reg_sontrasts:
        contrast_file_name = os.path.basename(con_path)
        total_path = os.path.join(subj_root, 'analysis', para_name + '_' + contrast_file_name)
        local_nii = nibabel.load(con_path)
        l_data = np.array(local_nii.get_fdata())
        if os.path.exists(total_path):
            total_nii = nibabel.load(total_path)
            t_data = np.array(total_nii.get_fdata())
            new_total = nibabel.Nifti1Image(t_data + l_data, affine=total_nii.affine, header=total_nii.header)
            nibabel.save(new_total, total_path)
        else:
            nibabel.save(local_nii, total_path)
    proj_config_path = os.path.join(subj_root, '../..', 'config.json')
    subject = os.path.basename(subj_root)
    with open(proj_config_path, 'r') as f:
        proj_config = json.load(f)
    if paradigm_data['name'] in proj_config['data_map'][subject]:
        proj_config['data_map'][subject][paradigm_data['name']]['sessions_included'].append(session_id)




