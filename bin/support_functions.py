import matplotlib.pyplot as plt
from typing import List, Union

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


def downsample_anatomical(inpath):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    source_dir = os.path.dirname(inpath)
    name = os.path.basename(inpath)
    factor = float(input("enter factor to downsample by (e.g. '2' will create a half sized anatomical on all dimmensions)"))
    out_name = 'ds_' + name.split('.')[0] + '.nii'
    output_dir = preprocess._create_dir_if_needed(subj_root, 'mri')
    output = os.path.join(output_dir, out_name)
    preprocess.create_low_res_anatomical(source_dir, name, output, factor=factor)
    return output


def coreg_wrapper(source_space_vol_path, target_space_vol_path):
    out = os.path.join(os.path.dirname(source_space_vol_path), 'coreg_3df.nii')
    return preprocess.antsCoReg(target_space_vol_path,
                                source_space_vol_path,
                                outP=out,
                                ltrns=['Affine', 'SyN'],
                                n_jobs=2)


def apply_warp(source, vol_in_target_space, forward_gross_transform_path, fine_transform_path):
    if type(source) is not list:
        source = [source]
    transforms = [forward_gross_transform_path, fine_transform_path]
    to_invert = [False, False]
    out_paths = []
    for s in source:
        out = os.path.join(os.path.dirname(s), 'reg_' + os.path.basename(s))
        preprocess.antsApplyTransforms(s, vol_in_target_space, out, transforms, 'Linear',
                                       img_type_code=0, invertTrans=to_invert)
        out_paths.append(out)
    if len(out_paths) == 1:
        out_paths = out_paths[0]
    return out_paths


def apply_warp_inverse(source, vol_in_target_space, forward_gross_transform_path, reverse_fine_transform_path):
    out = os.path.join(os.path.dirname(vol_in_target_space), 'inverse_trans_' + os.path.basename(source))
    inverse_transforms = [reverse_fine_transform_path, forward_gross_transform_path]
    to_invert = [False, True]
    preprocess.antsApplyTransforms(source, vol_in_target_space, out, inverse_transforms, 'Linear',
                                   img_type_code=0, invertTrans=to_invert)
    return out


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
    if type(reg_contrasts) is not list:
        reg_contrasts = [reg_contrasts]
    for contrast in reg_contrasts:
        out_paths.append(analysis.create_slice_maps(function_reg_vol, anatomical, contrast))
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


def itk_manual(epi_rep, template):
    out_dir = os.path.dirname(epi_rep)
    return preprocess._itkSnapManual(template, epi_rep, out_dir)


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


def create_contrast(source, paradigm_path, ima_order_map_path, mion, fname='epi_masked.nii'):
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    with open(ima_order_map_path, 'r') as f:
        ima_order_map = json.load(f)
    if mion is None:
        mion = False
    assert type(mion) is bool
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
    print("using mion: ", mion)
    print('source fname: ', fname)
    contrast_imgs = analysis.intra_subject_contrast(source, design_matrices, contrast_def, contrast_descriptions,
                                                    output_dir=sess_dir, fname=fname,
                                                    mode='maximal_dynamic', mion=mion, use_python_mp=True, auto_conv=True, tr=3)
    contrast_paths = [os.path.join(sess_dir, s + '_contrast.nii') for s in contrast_descriptions]
    print("contrasts created at: " + str(contrast_paths))
    return contrast_paths


def add_to_subject_contrast(reg_sontrasts: List[str], paradigm_path, *argv):
    subj_root = os.environ.get('FMRI_WORK_DIR')
    session_id = str(argv[0])
    with open(paradigm_path, 'r') as f:
        paradigm_data = json.load(f)
    para_name = paradigm_data['name']
    if type(reg_sontrasts) not in [list, tuple]:
        reg_sontrasts = [reg_sontrasts]
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


def segment_contrast_time_course(contrast_files, functional_dirs, ima_order_map, paradigm_file, fname='epi_masked.nii'):
    out_paths = []
    with open(paradigm_file, 'r') as f:
        para_data = json.load(f)
    with open(ima_order_map, 'r') as f:
        ima_data = json.load(f)
    block_length = para_data['block_length_trs']
    block_orders = para_data['order_number_definitions']
    print("Cannot use all runs to generate time series because of different orderings. While we could re-arrange and "
          "stack the sequences, this will introduce artifacts that reduce the usefulnesss of looking at the time course.")
    imas = [None]
    while False in [ima in ima_data and ima_data[ima] == ima_data[imas[0]] for ima in imas]:
        imas = input_control.int_list_input("enter IMA number of run(s) to use for timeseries. Must use the same order number. ")
        imas = [str(ima) for ima in imas]

    sess_dir = os.path.dirname(functional_dirs[0])
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