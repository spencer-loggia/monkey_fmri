# An interactive script to quickly proccess and get a sense for a single run of fmri data
import json

from preprocess import *
from preprocess import _create_dir_if_needed, _apply_binary_mask
from analysis import *
import nibabel as nib
import unpack
import os


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
        except TypeError:
            continue
        grammatical = True
    return nums


print("Running Interactive Intrasubject fMRI Preprocessing and First Level Analysis Code. \n"
      "Preprocessing Requires the following pre-existing files:\n"
      "- full head t1 for this animal \n"
      "- brain mask for above t1\n"
      "- functional runs from one scan session only \n",)

root = _dir_input("Enter the project root dir (i.e. the folder containing subfolders 'functional', 'mri', etc.)")
sessDir = _create_dir_if_needed(root, 'functional')
ana_dir = _create_dir_if_needed(root, 'mri')

print('Loaded project.')

import_dicoms = _bool_input('Do you wish to import new dicoms??')

if import_dicoms:
    dicom_dir = _dir_input('Enter directory containing DICOMS from scans we are trying to analyze')
    run_numbers = _int_list_input("Enter whitespace separated list of valid epi run IMA numbers. (e.g. '2 4 5')")
    SOURCE = unpack.unpack_run_list(dicom_dir, sessDir, run_numbers, 'f')
    print("Created the following functional run directories: \n ",
          SOURCE,
          " \n each containing a raw functional file 'f.nii.gz'")

else:
    print("Not importing dicoms")
    run_dirs = [f for f in
                os.listdir(_dir_input("enter a session directory containing run dirs containing source niftis"))
                if f.isnumeric()]
    SOURCE = [os.path.join(sessDir, f) for f in run_dirs]

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

print("Next step is to preform an initial manual registration from function to anatomical in ITK-Snap. \n"
      "We can either do a manual registration for every epi (probably better, but time consuming), \n "
      "Or we can register the average functional and apply the transform to each epi")
do_quick_reg = _bool_input("Do you want to do registration?")

if do_quick_reg:
    average_nii_name = 'average_functional.nii'
    avg_func = average_functional_data(SOURCE,
                                       output=os.path.join(sessDir, average_nii_name),
                                       fname=process_head)
    res = manual_itksnap_registration([sessDir], fname=average_nii_name, template_file=ds_t1_path)
    print(res)
    transform_path = res['transform']
    for run_dir in SOURCE:
        antsApplyTransforms(inP=os.path.join(run_dir, process_head),
                            refP=ds_t1_path,
                            outP=os.path.join(run_dir, 'manual_registered.nii'),
                            lTrns=[transform_path])
    process_head = 'manual_registered.nii'
    print('functional data was transformed according to manual specification')

# create the masked t1
# save masked t1
t1_masked = _apply_binary_mask(ts_in=nib.load(ds_t1_path), mask=nib.load(ds_t1_mask_path))
ds_t1_masked_path = os.path.join(ana_dir, 'ds_t1_masked.nii')
nib.save(t1_masked, ds_t1_masked_path)

print('Now the anatomical brain mask must be applied to the manually registered functional data files.')
if _bool_input('Do you wish to apply the brain mask to the head of your functional data?'):
    for run_dir in SOURCE:
        f_masked = _apply_binary_mask(ts_in=nib.load(os.path.join(run_dir, process_head)),
                                      mask=nib.load(ds_t1_mask_path))
        nib.save(f_masked, os.path.join(run_dir, 'f_stripped.nii'))
    process_head = 'f_stripped.nii'

print('Now we can automatically register the stripped epis to the t1')

transform_name = None
if _bool_input("Do you want to do automatic affine registration?"):
    linear_affine_registration(SOURCE, template_file=ds_t1_masked_path, fname=process_head, dof=6)
    transform_name = 'f_stripped_flirt.mat'

nonlinear_warp_name = None
if _bool_input("Do you want to do nonlinear coregistration of each functional to the t1? Might be slow, ~15 minutes per batch, "
               "where a batch is runs equal to number of available cpus - 1."):
    nonlinear_registration(functional_input_dirs=SOURCE,
                           transform_input_dir=SOURCE,
                           template_file=ds_t1_masked_path,
                           affine_fname=transform_name,
                           source_fname=process_head,
                           output='reg_tensor.nii.gz')
    transform_name = 'reg_tensor.nii.gz'

preform_nifti_registration(functional_input_dirs=SOURCE, transform_input_dir=SOURCE,
                           template_file=ds_t1_masked_path, source_fname=process_head,
                           transform_fname=transform_name, output='f_registered.nii.gz')

print('the functional data has been registered to the t1 successfully. Please check the results in freeview or fsl eyes')
if not _bool_input("Continue to first level analysis?? "):
    exit(0)

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

if _bool_input("do you already have a paradigm definition json file?" ):
    para_json = _dir_input("Enter the path to the existing paradigm order definition json. ")
    para_def_dict = json.loads(para_json)
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
    for i in range(1, num_conditions + 1):
        condition_map[i] = input('description for condition #' + str(i))
    para_def_dict['condition_integerizer'] = condition_map
    order_def_map = {}
    for i in range(1, num_orders + 1):
        order_def_map[i] = _int_list_input('enter the block order if block design, or event sequence if event related.')
    para_def_dict['order_number_definitions'] = order_def_map
    config_dir = _create_dir_if_needed(root, 'config')
    config_file_path = os.path.join(config_dir, 'experiment_config.json')
    with open(config_file_path, 'w') as f:
        json.dump(para_def_dict, f)
    print("Successfully saved experiment / paradigm configuration json at", config_file_path)

interactive_ima_map = _bool_input('Specify IMA to order number mapping interactively? (provide json file otherwise)')

ima_order_map = {}
if interactive_ima_map:
    for s in SOURCE:
        ima = os.path.basename(s)
        order_num = int(input("Enter order number for ima" + ima))
        ima_order_map[int(ima)] = order_num
    save_path = os.path.join(os.path.dirname(SOURCE), 'ima_order_map.json')
    with open(save_path, 'w') as f:
        json.dump(ima_order_map, f)
    print("Successfully saved ima to order number mapping json at", save_path)
else:
    input = _dir_input("enter path to ima to order number json: ")
    ima_order_map = json.loads(input)







