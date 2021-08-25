from __future__ import print_function
from __future__ import division

import shutil
import sys
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


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    mean = np.mean(arr.reshape(-1))
    std = np.std(arr.reshape(-1))
    new_arr = (arr - mean) / std
    return new_arr


def _apply_binary_mask(ts_in: nib.Nifti1Image, mask: nib.Nifti1Image) -> nib.Nifti1Image:
    n_data = np.array(ts_in.get_fdata())
    mask_arr = np.array(mask.get_fdata())[:, :, :, None]
    tiled_mask = np.tile(mask_arr, (1, 1, 1, n_data.shape[3]))
    n_data[tiled_mask == 0] = 0
    new_nifti = nib.Nifti1Image(n_data, affine=ts_in.affine, header=ts_in.header)
    return new_nifti


def _extract_frame(nii: nib.Nifti1Image, loc: Union[None, int] = None):
    """
    extracts a 3D reference frame from 4D nifti object and generates a 3D nifti object
    :param nii:
    :param loc:
    :return:
    """
    n_data = np.array(nii.get_fdata())
    if not loc:
        loc = int(n_data.shape[2] / 2)
    n_data = n_data[:, :, :, loc]
    new_nifti = nib.Nifti1Image(n_data, affine=nii.affine, header=nii.header)
    return new_nifti


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
                    local_out = os.path.join(out_dir, 'f.nii')
                cvt.inputs.out_file = local_out
                cvt.inputs.args = "--sphinx"
                cvt.inputs.out_type = 'nii'
                cvt.cmdline
                cvt.run()

    return output


def motion_correction(input_dirs: List[str], output: Union[None, str], check_rms=True) -> List[List[Tuple[str]]]:
    """
    preform fsl motion correction
    :param output:
    :param input_dirs:
    :param check_rms:
    :return:
    """
    all_outputs = []
    out_dirs = []
    for source_dir in input_dirs:
        outputs = []
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        mcflt = fsl.MCFLIRT()
        for source in files:
            if source[-4:] == '.nii':
                if not output:
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
                flt.inputs.in_file = os.path.join(source_dir, source)
                flt.inputs.reference = template_file
                flt.inputs.dof = 12
                flt.inputs.args = '-omat ' + mat_out
                flt.cmdline
                out = flt.run()
                outputs.append(out)
    return outputs


def nonlinear_registration(functional_input_dirs: List[str], transform_input_dir: List[str], template_file: str,
                           source_fname: str = 'moco.nii.gz', affine_fname: str = 'moco_flirt.mat',
                           output: str = None):
    fnt = fsl.FNIRT()
    outputs = []
    try:
        sources = zip(functional_input_dirs, transform_input_dir)
    except Exception:
        print("function_input_dirs and transform_input_dirs must be lists of the same length.", sys.stderr)
        exit(-2)
    for source_dir, transform_dir in sources:
        try:
            files = os.listdir(source_dir)
            tfiles = os.listdir(transform_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        if source_fname in files and affine_fname in tfiles:
            if not output:
                local_out = os.path.join(source_dir, 'reg_tensor.nii.gz')
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(out_dir, 'reg_tensor.nii.gz')
            fnt.inputs.in_file = os.path.join(source_dir, source_fname)
            fnt.inputs.ref_file = template_file
            fnt.inputs.affine_file = os.path.join(source_dir, affine_fname)
            fnt.cmdline
            out = fnt.run()
            outputs.append(out)

            # workaround fsl fnirt cout error
            warp_file = os.path.join(source_dir, [s for s in os.listdir(source_dir) if '_warpcoef' in s][0])
            shutil.copy(warp_file, local_out)
            os.remove(warp_file)
        else:
            raise FileNotFoundError("The specified source nifti or affine transform matrix cannot be found.")
    return outputs


def preform_nifti_registration(functional_input_dirs: List[str], transform_input_dir: Union[None, List[str]] = None, template_file: str = None,
                    output: str = None, source_fname: str = 'moco.nii.gz', transform_fname: str = 'reg_tensor.nii.gz'):
    """
    Apply a registration file to a 4D nifti and save the output.
    :param functional_input_dirs: Directories where we expect to find scan directories containing 4D input niftis.
    :param transform_input_dir: (default: None) Directories where we expect to find scan directories containing combined lin,
                            non-lin transforms. If None assumed to be the same as input directories.
    :param output:  (default: None) Parent folder to place scan directories containing transformed 4D niftis. If None
                    same as source.
    :param input_fname:  (default: 'moco.nii.gz') expected base filename for input 4D niftis.
    :param transform_fname:  (default: 'register.dat'):  expected base filename for registration transforms.
    :return:
    """
    apw = fsl.ApplyWarp()
    outputs = []
    try:
        sources = zip(functional_input_dirs, transform_input_dir)
    except Exception:
        print("function_input_dirs and transform_input_dirs must be lists of the same length.", sys.stderr)
        exit(-2)
    for source_dir, transform_dir in sources:
        try:
            files = os.listdir(source_dir)
            tfiles = os.listdir(transform_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        if source_fname in files and transform_fname in tfiles:
            if not output:
                local_out = os.path.join(source_dir, 'registered.nii.gz')
            else:
                out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                local_out = os.path.join(out_dir, 'registered.nii.gz')
            apw.inputs.in_file = os.path.join(source_dir, source_fname)
            apw.inputs.ref_file = template_file
            apw.inputs.field_file = os.path.join(transform_dir, transform_fname)
            apw.inputs.out_file = local_out
            apw.cmdline
            res = apw.run()
            outputs.append(res)
        else:
            raise FileNotFoundError("The specified source nifti or nonlinear transform tensor cannot be found.")
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
                    local_out = os.path.join(source_dir, 'smooth.nii.gz')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    local_out = os.path.join(out_dir, 'smooth.nii.gz')
                sus.inputs.in_file = os.path.join(source_dir, source)
                sus.inputs.fwhm = fwhm
                sus.inputs.brightness_threshold = bright_tresh
                sus.cmdline
                out = sus.run()
                shutil.copy('./' + fname[:-4] + '_smooth.nii.gz', local_out)
                os.remove('./' + fname[:-4] + '_smooth.nii.gz')
                outputs.append(out)
        return outputs


def skull_strip_4d(input_dirs: List[str], output: str, fname: str = 'moco.nii.gz'):
    bet = fsl.BET()
    outputs = []
    for source_dir in input_dirs:
        if fname not in os.listdir(source_dir):
            print("could not find file")
            exit(1)
        if not output:
            local_out = os.path.join(source_dir, 'stripped.nii.gz')
        else:
            out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
            local_out = os.path.join(out_dir, 'stripped.nii.gz')
        nii = nib.load(os.path.join(source_dir, fname))
        # create reference 3D nifti
        s_nii = _extract_frame(nii)
        _create_dir_if_needed('./', 'tmp')
        nib.save(s_nii, './tmp/3d.nii')
        bet.inputs.in_file = './tmp/3d.nii'
        bet.inputs.mask = True
        bet.cmdline
        out = bet.run()
        outputs.append(out)
        mask_file = os.path.join('./', [s for s in os.listdir('./') if '_brain_mask' in s][0])
        mask = nib.load(mask_file)
        result = _apply_binary_mask(nii, mask)
        nib.save(result, local_out)
        return outputs


def normalize(input_dirs: List[str], output: str, fname='smooth.nii.gz'):
    """
    Centers and normalizes the intensity data using (X - mu) / std. Creates a normalized nifti
    :param input_dirs:
    :param output:
    :param fname:
    :return:
    """
    for source_dir in input_dirs:
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        for source in files:
            if fname in source:
                if not output:
                    local_out = os.path.join(source_dir, 'normalized.nii')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    local_out = os.path.join(out_dir, 'normalized.nii')
                nifti = nib.load(os.path.join(source_dir, source))
                n_data = np.array(nifti.get_fdata())
                n_data = _normalize_array(n_data)
                new_nifti = nib.Nifti1Image(n_data, affine=nifti.affine, header=nifti.header)
                nib.save(new_nifti, local_out)
                print(nifti.header)


def create_mask(input_dirs: List[str], output: str, fname='normalized.nii.gz', thresh: Union[float, None] = None):
    """
    creates a binary mask of a input nifti. By default sets threshold at 3 std devs above the mean.
    :param input_dirs:
    :param output:
    :param fname:
    :param thresh:
    :return:
    """
    bet = fsl.BET()
    for source_dir in input_dirs:
        try:
            files = os.listdir(source_dir)
        except (FileExistsError, FileNotFoundError):
            print("could not find file")
            exit(1)
        for source in files:
            if fname in source:
                if not output:
                    mask_out = os.path.join(source_dir, 'bin_mask.nii')
                    ss_out = os.path.join(source_dir, 'skull_stripped.nii')
                else:
                    out_dir = _create_dir_if_needed(output, os.path.basename(source_dir))
                    mask_out = os.path.join(out_dir, 'bin_mask.nii')
                    ss_out = os.path.join(out_dir, 'skull_stripped.nii')
                bet.inputs.in_file = os.path.join(source_dir, source)
                bet.inputs.functional = True
                bet.inputs.out_file = ss_out
                bet.cmdline
                bet.run()
                nifti = nib.load(os.path.join(source_dir, source))
                n_data = np.array(nifti.get_fdata())
                if not thresh:
                    u = np.mean(n_data.flatten())
                    s = np.std(n_data.flatten())
                    thresh = u + 3*s
                n_data[n_data > thresh] = 9999999
                n_data[n_data <= thresh] = 0
                n_data /= 9999999
                new_nifti = nib.Nifti1Image(n_data, affine=nifti.affine, header=nifti.header)
                nib.save(new_nifti, mask_out)


