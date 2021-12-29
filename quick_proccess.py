# An interactive script to quickly proccess and get a sense for a single run of fmri data
from copy import copy

import json
import numpy as np

from preprocess import *
from preprocess import _create_dir_if_needed, _apply_binary_mask_3D
from analysis import *
import nibabel as nib
import unpack
import os
from multiprocessing import Pool


def _dir_input(msg: str):
    path = "/not_a_path"
    while not os.path.exists(path):
        path = input(msg)
    return path


def _bool_input(msg: str):
    tkn = ''
    while tkn not in ['y', 'n']:
        tkn = input(msg + ' (y / n) ').strip().lower()
    return tkn == 'y'


def _int_list_input(msg: str):
    grammatical = False
    nums = []
    while not grammatical:
        sent = input(msg)
        tkns = sent.split()
        try:
            nums = [int(t) for t in tkns]
        except ValueError:
            try:
                nums = [float(t) for t in tkns]
            except ValueError:
                continue
        break
    return nums


def interactive_program():
    print("Running Interactive Intrasubject fMRI Preprocessing and First Level Analysis Code. \n"
          "Preprocessing Requires the following pre-existing files:\n"
          "- full head t1 for this animal \n"
          "- brain mask for above t1\n"
          "- functional runs from one scan session only \n", )

    root = _dir_input("Enter the project root dir (i.e. the folder containing subfolders 'functional', 'mri', etc.)")
    f_dir = _create_dir_if_needed(root, 'functional')
    ana_dir = _create_dir_if_needed(root, 'mri')

    print('Loaded project.')

    import_dicoms = _bool_input('Do you wish to import new dicoms??')

    if import_dicoms:
        dicom_dir = _dir_input('Enter directory containing DICOMS from scans we are trying to analyze')
        run_numbers = _int_list_input("Enter whitespace separated list of valid epi run IMA numbers. (e.g. '2 4 5')")
        SOURCE = unpack.unpack_run_list(dicom_dir, f_dir, run_numbers, 'f')
        sessDir = os.path.dirname(SOURCE[0])
        print("Created the following functional run directories: \n ",
              SOURCE,
              " \n each containing a raw functional file 'f.nii.gz'")

    else:
        print("Not importing dicoms")
        sessDir = _dir_input("enter a session directory containing run dirs containing source niftis")
        run_dirs = [f for f in
                    os.listdir(sessDir)
                    if f.isnumeric()]
        SOURCE = [os.path.join(sessDir, f) for f in run_dirs]

    check = _bool_input("Check functional time series lengths? ")
    if check:
        expected = int(input("What is the expected number of trs? "))
        SOURCE = check_time_series_length(SOURCE, fname='f.nii.gz', expected_length=expected)

    print('Functional data specified / loaded.')

    t1_path = "/not_a_path"
    t1_mask_path = '/not_a_path'
    t1_name = ''
    t1_mask_name = ''
    print("Anatomical data is assumed to be located in [root]/mri, in this case", ana_dir)
    while not os.path.exists(t1_path) and not os.path.exists(t1_mask_path):
        t1_name = input('enter name of whole head t1.')
        t1_mask_name = input('enter name of t1 brain mask.')
        t1_path = os.path.join(ana_dir, t1_name)
        t1_mask_path = os.path.join(ana_dir, t1_mask_name)

    process_head = 'f.nii.gz'
    print("The following initial preprocessing steps will now be preformed. \n"
          "- sphinx position correction \n"
          "- linear motion correction \n"
          "- creation of low resolution anatomical and mask")
    do_init_preprocess = _bool_input("do you wish to proceed ?")

    ds_t1_name = 't1_lowres.nii'
    ds_t1_path = os.path.join(ana_dir, ds_t1_name)
    ds_t1_mask_name = 't1_lowres_mask.nii'
    ds_t1_mask_path = os.path.join(ana_dir, ds_t1_mask_name)

    # create the masked t1
    # save masked t1
    t1_masked = _apply_binary_mask_3D(in_img=nib.load(ds_t1_path), mask=nib.load(ds_t1_mask_path))
    ds_t1_masked_path = os.path.join(ana_dir, 'ds_t1_masked.nii')
    nib.save(t1_masked, ds_t1_masked_path)

    if do_init_preprocess:
        convert_to_sphinx(SOURCE, fname=process_head, scan_pos='HFP')
        process_head = 'f_sphinx.nii'
        SOURCE = motion_correction(SOURCE, fname=process_head, abs_threshold=1., var_threshold=.3)
        process_head = 'moco.nii.gz'
        create_low_res_anatomical(source_dir=ana_dir,
                                  fname=t1_name,
                                  factor=2.,
                                  output=ds_t1_path)
        create_low_res_anatomical(source_dir=ana_dir,
                                  fname=t1_mask_name,
                                  factor=2.,
                                  output=ds_t1_mask_path)

        print("initial preprocessing complete")
    else:
        print("skipping initial preprocessing...")
        process_head = input("What is the basename if the function inputs to use going forward?")

    print("Next step is to preform an initial manual registration from function to anatomical in ITK-Snap. \n"
          "We can either do a manual registration for every epi (probably better, but time consuming), \n "
          "Or we can register the average functional and apply the transform to each epi")
    do_quick_reg = _bool_input("Do you want to do registration?")
    ref_img = None

    if do_quick_reg:
        average_f_man_reg_stripped = None
        average_f = None
        do_man_reg = _bool_input("do manual registration? ")
        if do_man_reg:
            average_f = 'average_functional.nii'
            avg_func = average_functional_data(SOURCE,
                                               output=os.path.join(sessDir, average_f),
                                               fname=process_head,
                                               through_time=True)
            res = manual_itksnap_registration([sessDir], fname=average_f, template_file=ds_t1_path)
            print(res)
            man_transform_path = res['transform_path']
            average_f_man_reg = os.path.basename(res['registered_path'])
            ref_img = copy(average_f_man_reg)
            print('manual transform saved')

            print('Now the anatomical brain mask must be applied to the manually registered functional data files.')
            if _bool_input('Do you wish to apply the brain mask to average functional ?'):
                f_masked = _apply_binary_mask_3D(in_img=nib.load(os.path.join(sessDir, average_f_man_reg)),
                                                 mask=nib.load(ds_t1_mask_path))
                average_f_man_reg_stripped = 'avg_f_stripped.nii'
                nib.save(f_masked, os.path.join(sessDir, average_f_man_reg_stripped))

            print('Now we can automatically register the stripped epis to the t1')
        else:
            man_transform_path = _dir_input("Enter path to grass spatial transform (i.e. the one that would have been "
                                            "defined manually)")
        if _bool_input("Do you want to do automatic coregistration?"):
            if not average_f_man_reg_stripped:
                average_f_man_reg_stripped = _dir_input(
                    'enter path to 3D averaged representation of functional data. If you select'
                    ' a file that has een transformed out of the native space, you must include'
                    ' that transform in the transform stack when applying registration.')
            f3d_reg_out = 'fine_grained_transform.nii.gz'
            fine_transform_path, fine_transform_inverse_path = antsCoReg(ds_t1_masked_path,
                                       average_f_man_reg_stripped,
                                       outP=os.path.join(sessDir, f3d_reg_out),
                                       ltrns=['Affine', 'SyN'],
                                       n_jobs=2)
        else:
            fine_transform_path = _dir_input("Enter path to fined grained forward transform (should be an h5 file "
                                             "defined using ants coregistration: ")
            fine_transform_inverse_path = _dir_input("Enter path to inverse fine grained transform (h5 file): ")

        print("We can use the inverse of the registration transforms to create a brain mask in the native functional "
              "space. These masked files will be used for contrast creation")

        mask_functional = _bool_input("mask the functional data? ")
        if mask_functional:
            if not average_f:
                average_f = _dir_input('specify path to 3d functional average unstripped in native space: ')
            # first apply the tranform to the average for sanity check

            f_mask = os.path.join(sessDir, 'functional_mask.nii.gz')
            inverse_transforms = [fine_transform_inverse_path, man_transform_path]
            to_invert = [False, True]  # manual transform affine matrix is invertable by applywarp,
                                       # nonlinear requires seperate transform file, so is not inverted again by
                                       # applywarp
            antsApplyTransforms(ds_t1_mask_path, average_f, f_mask, inverse_transforms, 'Linear',
                                img_type_code=0, invertTrans=to_invert)
            print('mask the functional data ')
            mask = nib.load(f_mask)
            stripped_head = 'f_moco_stripped.nii'
            for run_dir in SOURCE:
                in_img = nib.load(os.path.join(run_dir, process_head))
                stripped_img = _apply_binary_mask_3D(in_img, mask)
                nib.save(stripped_img, os.path.join(run_dir, stripped_head))
            process_head = stripped_head

        # if len(transforms) > 0:
        #     reg_args = []
        #     if not ref_img:
        #         ref_img = _dir_input('specify path to 3d reference (img generated from gross transform): ')
        #     if not average_f:
        #         average_f = _dir_input('specify path to 3d functional average unstripped in native space: ')
        #     # first apply the tranform to the average for sanity check
        #     out = os.path.join(sessDir, 'registered_avg_f.nii.gz')
        #     antsApplyTransforms(os.path.join(sessDir, average_f), ref_img, out, transforms, 'Linear', img_type_code=0)
        #     for run_dirs in SOURCE:
        #         out = os.path.join(run_dirs, 'registered.nii.gz')
        #         reg_args.append((os.path.join(run_dirs, process_head), ref_img, out, transforms, 'Linear'))
        #     process_head = 'registered.nii.gz'
        #     with Pool() as p:
        #         p.starmap(antsApplyTransforms, reg_args)
        else:
            process_head = input("Enter file name for functional data process head: ")
        #
        # if _bool_input("mask functional data head? "):
        #     mask = nib.load(ds_t1_mask_path)
        #     stripped_head = 'stripped_registered.nii'
        #     for run_dir in SOURCE:
        #         in_img = nib.load(os.path.join(run_dir, process_head))
        #         stripped_img = _apply_binary_mask_3D(in_img, mask)
        #         nib.save(stripped_img, os.path.join(run_dir, stripped_head))
        #     process_head = stripped_head
        print(
            'the functional data has been registered to the t1 successfully. Please check the results in freeview or '
            'fsl eyes')
    else:
        man_transform_path = _dir_input("Enter path to grass spatial transform (i.e. the one that would have been "
                                        "defined manually)")
        fine_transform_path = _dir_input("Enter path to fined grained forward transform (should be an h5 file "
                                         "defined using ants coregistration: ")

    if _bool_input("Continue to first level analysis?? "):
        print('*' * 40)
        print('Beginning First Level Analysis ')
        print("To complete all steps in the First Level Analysis we need the following files.")
        print("- functional data registered to a t1 \n"
              "- white matter and inflated surfaces for the t1\n"
              "- Paradigm version definition file, specifying the block order for associated with each version number."
              " Should be a json file with order number keys and a list of stimuli descriptions or identifiers as values."
              "- A txt file defining the contrasts to be preformed. Each row should be a vector of length equal to the number of "
              "unique stimuli conditions.")
        print('*' * 40)

        if _bool_input("do you already have a paradigm definition json file?"):
            para_json = _dir_input("Enter the path to the existing paradigm order definition json. ")
            with open(para_json, 'r') as f:
                para_def_dict = json.load(f)
        else:
            print('Constructing new paradigm definition json')
            para_def_dict = {}
            name = input('what is this paradigm called? ')
            para_def_dict['name'] = name
            num_trs = int(input('how many trs are in each run ? '))
            para_def_dict['trs_per_run'] = num_trs
            block_design = _bool_input('is this paradigm a block design? ')
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
                order_def_map[str(i)] = _int_list_input(
                    'enter the block order if block design, or event sequence if event related.')
            para_def_dict['order_number_definitions'] = order_def_map
            config_dir = _create_dir_if_needed(root, 'config')
            config_file_path = os.path.join(config_dir, 'experiment_config.json')
            with open(config_file_path, 'w') as f:
                json.dump(para_def_dict, f)
            print("Successfully saved experiment / paradigm configuration json at", config_file_path)

        interactive_ima_map = _bool_input(
            'Specify IMA to order number mapping interactively? (provide json file otherwise)')

        ima_order_map = {}
        if interactive_ima_map:
            for s in SOURCE:
                ima = os.path.basename(s).strip()
                order_num = int(input("Enter order number for ima" + ima))
                ima_order_map[ima] = order_num
            save_path = os.path.join(os.path.dirname(SOURCE[0]), 'ima_order_map.json')
            with open(save_path, 'w') as f:
                json.dump(ima_order_map, f)
            print("Successfully saved ima to order number mapping json at", save_path)
        else:
            dir_input = _dir_input("enter path to ima to order number json: ")
            with open(dir_input, 'r') as f:
                ima_order_map = json.load(f)

        # need to define contrasts we want to preform
        print(
            "Need to define contrasts to preform to continue with analysis. Enter length number conditions vector defining "
            "contribution for each condition. For example [0, 1, 0, -1] defines a direct comparison between a positive "
            "condition number 2 and negative condition number 4")
        print("Recall: conditions are mapped as follows ", para_def_dict['condition_integerizer'])
        add_contrast = True
        contrasts = []
        contrast_descriptions = []
        while add_contrast:
            contrast = _int_list_input("Enter vector defining constrast: ")
            if len(contrast) != para_def_dict['num_conditions']:
                print('contrast vector must have length equal to the number of conditions.')
                continue
            contrast_descriptions.append(input("Name this contrast: "))
            contrasts.append(contrast)
            add_contrast = _bool_input('add another contrast? ')
        contrasts = np.array(contrasts).astype(float).T

        print("Generating Contrasts...")
        design_matrices = []
        analysis_dir = _create_dir_if_needed(root, 'analysis')
        for run_dir in SOURCE:
            ima = os.path.basename(run_dir).strip()
            order_num = ima_order_map[ima]
            block_length = int(para_def_dict['block_length_trs'])
            num_blocks = int(para_def_dict['trs_per_run'] / block_length)
            num_conditions = int(para_def_dict['num_conditions'])
            order = list(para_def_dict['order_number_definitions'][str(order_num)])
            design_matrix = design_matrix_from_order_def(block_length, num_blocks, num_conditions, order,
                                                         convolve=False)
            design_matrices.append(design_matrix)
        contrast_imgs = intra_subject_contrast(SOURCE, design_matrices, contrasts, contrast_descriptions,
                                               output_dir=analysis_dir, fname=process_head,
                                               mode='maximal_dynamic', use_python_mp=True)
        contrast_paths = [os.path.join(analysis_dir, s + '_contrast.nii') for s in contrast_descriptions]
        print("contrasts created at: " + str(contrast_paths))
    else:
        contrast_paths = []
        print("Manually Define previously created contrasts: ")
        add_contrast = True
        while add_contrast:
            contrast_paths.append(_dir_input("enter path to contrast: "))
            add_contrast = _bool_input("add another contrast file? ")

    if _bool_input('generate contrast surfaces? '):
        print('Registering contrasts to anatomical space...')
        if not ref_img:
            ref_img = _dir_input("Enter path to target registration image in anatomical space (ideally an already"
                                 " registered average functional, though a true anatomical should also work ok): ")
        reg_contrast_paths = []
        for contrast_path in contrast_paths:
            base = os.path.dirname(contrast_path)
            name = 'reg_' + os.path.basename(contrast_path)
            out = os.path.join(base, name)
            transforms = [man_transform_path, fine_transform_path]
            antsApplyTransforms(contrast_path, ref_img, out, transforms, 'Linear', img_type_code=0, invertTrans=False)
            reg_contrast_paths.append(out)
        contrast_paths = reg_contrast_paths

        print('next step is to generate contrast surfaces... assuming surfaces are located in anatomical dir and called '
              'rh.white, lh.white, rh.inflated, lh.inflated')

        generate_surfaces = _bool_input("Do you want to generate contrast surfaces?")

        if generate_surfaces:
            print("Gererating Surfaces...")

            for contrast_path in contrast_paths:
                for hemi in ['rh', 'lh']:
                    # freesurfer has the dumbest path lookup schema so we have to be careful here
                    print(root)
                    create_contrast_surface(os.path.join(root, 'surf', hemi + '.white'),
                                            contrast_path,
                                            ds_t1_path,
                                            t1_path,
                                            hemi=hemi, subject_id='/')


if __name__ == '__main__':
    interactive_program()
