from __future__ import print_function
from __future__ import division
from typing import List, Union, Tuple, Dict

import numpy as np
from nipype.interfaces import freesurfer
from builtins import str
from builtins import range

import os  # system functions

import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.algorithms.modelgen as model  # model generation
import nipype.algorithms.rapidart as ra  # artifact detection
import nibabel as nib

from matplotlib import pyplot as plt


def _create_dir_if_needed(base: str, name: str):
    out_dir = os.path.join(base, name)
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass
    return out_dir


def convert_to_sphinx(input_dirs: List[str], output: Union[None, str], ) -> str:
    """
    Convert to sphinx
    :param input_dirs: paths to dirs with input nii files, (likely in the MION dir)
    :param output: output directory to create or populate (if None put in same dirs as input)
    :return: path to output directory
    """

    for scan_dir in input_dirs:
        files = os.listdir(scan_dir)
        os.environ.setdefault("SUBJECTS_DIR", scan_dir)
        cvt = freesurfer.MRIConvert()
        for f in files:
            if len(f) > 3 and f[-4:] == '.nii':
                cvt.inputs.in_file = os.path.join(scan_dir, f)
                if not output:
                    local_out = os.path.join(scan_dir, 'f.nii')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(scan_dir))
                    local_out = os.path.join(out_dir, 'f_sphinx.nii')
                cvt.inputs.out_file = local_out
                cvt.inputs.args = "--sphinx"
                cvt.inputs.out_type = 'nii'
                cvt.cmdline
                cvt.run()

    return output


def motion_correction(input_dirs: List[str], output: Union[None, str], fname='f_sphinx.nii', check_rms=True) -> List[List[Tuple[str]]]:
    """
    preform fsl motion correction
    :param output:
    :param input_dirs:
    :param fname:
    :param check_rms:
    :return:
    """
    all_outputs = []
    out_dirs = []
    for source_dir in input_dirs:
        outputs = []
        if os.path.isfile(os.path.join(source_dir,fname)):
            source = os.path.join(source_dir, fname)
            mcflt = fsl.MCFLIRT()
            if not output:
                out_dir = source_dir
                local_out = os.path.join(source_dir, 'moco.nii.gz')
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(out_dir, 'moco.nii.gz')
            path = os.path.join(source_dir, source)
            mcflt.inputs.in_file = path
            mcflt.inputs.cost = 'mutualinfo'
            mcflt.inputs.out_file = local_out
            mcflt.inputs.save_plots = True
            mcflt.inputs.save_rms = True
            mcflt.cmdline
            out = mcflt.run()
            outputs.append(out)
            out_dirs.append(out_dir)
            all_outputs.append(outputs)
        else:
            print("could not find file")
            exit(1)
    if check_rms:
        if not output:
            output = './'
        plot_moco_rms_displacement(out_dirs, output)
    return all_outputs


def plot_moco_rms_displacement(transform_file_dirs: List[str], save_loc: str, threshold=.5) -> Dict[str, bool]:
    fig, ax = plt.subplots(1)
    good_moco = {}
    for transform_file_dir in transform_file_dirs:
        for f in os.listdir(transform_file_dir):
            if f == 'moco.nii.gz_abs.rms':
                disp_vec = np.loadtxt(os.path.join(transform_file_dir, f))
                label = os.path.basename(transform_file_dir)
                ax.plot(disp_vec, label=label)
                is_good_moco = all(disp_vec <= threshold)
                good_moco[label] = is_good_moco
    ax.set_title("MOCO RMS Vector Translation (mm)")
    ax.set_xlabel("frame number")
    fig.legend(loc='upper left')
    fig.show()
    fig.savefig(save_loc + '/moco_rms_displacement.eps', format='eps')
    return good_moco


def linear_affine_registration(functional_input_dirs: List[str], template_file: str, fname: str ='moco.nii', output: str = None):
    flt = fsl.FLIRT()
    outputs = []
    for source_dir in functional_input_dirs:
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        for source in files:
            if fname == source:
                if not output:
                    local_out = os.path.join(source_dir, 'moco_flirt.nii'.gz)
                    mat_out = os.path.join(source_dir, 'moco_flirt.mat')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    local_out = os.path.join(out_dir, 'moco_flirt.nii.gz')
                    mat_out = os.path.join(out_dir, 'moco_flirt.mat')
                flt.inputs.in_file =  os.path.join(source_dir, source)
                flt.inputs.reference = template_file
                flt.inputs.out_file = local_out
                flt.inputs.out_matrix_file = mat_out
                flt.cmdline
                out = flt.run()
                outputs.append(out)
    return outputs


def nonlinear_registration(functional_input_dirs: List[str], template_file: str, fname: str ='moco_flirt.nii.gz', output: str = None):
    fnt = fsl.FNIRT()
    outputs = []
    for source_dir in functional_input_dirs:
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        for source in files:
            if fname in source:
                if not output:
                    local_out = os.path.join(source_dir, 'nirt.nii.gz')
                    mat_out = os.path.join(source_dir, 'nirt_jacobian.mat')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    local_out = os.path.join(out_dir, 'nirt.nii.gz')
                    mat_out = os.path.join(out_dir, 'nirt_jacobian.nii.gz')
                fnt.inputs.in_file = os.path.join(source_dir, source)
                fnt.inputs.ref_file = template_file
                fnt.inputs.jacobian_file = mat_out
                fnt.inputs.warped_file = local_out
                fnt.cmdline
                out = fnt.run()
                outputs.append(out)
    return outputs


def fix_nii_headers(input_dirs: List[str], output: str, fname: str = 'nirt.nii.gz', tr=2000):
    for scan_dir in input_dirs:
        files = os.listdir(scan_dir)
        os.environ.setdefault("SUBJECTS_DIR", scan_dir)
        cvt = freesurfer.MRIConvert()
        for f in files:
            if len(f) > 3 and f == fname:
                cvt.inputs.in_file = os.path.join(scan_dir, f)
                if not output:
                    local_out = os.path.join(scan_dir, 'fixed.nii')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(scan_dir))
                    local_out = os.path.join(out_dir, 'fixed.nii')
                cvt.inputs.out_file = local_out
                cvt.inputs.tr = tr
                cvt.inputs.out_type = 'nii'
                cvt.cmdline
                cvt.run()


def smooth(input_dirs: List[str], output: str, fname: str = 'fixed.nii', bright_tresh=1000.0, fwhm=4.0):
    sus = fsl.SUSAN()
    outputs = []
    for source_dir in input_dirs:
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        for source in files:
            if fname in source:
                if not output:
                    local_out = os.path.join(source_dir, 'smoothed.nii')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    local_out = os.path.join(out_dir, 'smoothed.nii')
                sus.inputs.in_file = os.path.join(source_dir, source)
                sus.inputs.fwhm = fwhm
                sus.smoothed_file = local_out
                sus.inputs.brightness_threshold = bright_tresh
                sus.cmdline
                out = sus.run()
                outputs.append(out)
        return outputs


def normalize(input_dirs: List[str], output: str, fname='smooth.nii'):
    for source_dir in input_dirs:
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        for source in files:
            if fname in source:
                if not output:
                    local_out = os.path.join(source_dir, 'smoothed.nii.gz')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    local_out = os.path.join(out_dir, 'smoothed.nii.gz')
                    nifti = nib.load(os.path.join(source_dir, source))
                    print(nifti.header)
    raise NotImplementedError

