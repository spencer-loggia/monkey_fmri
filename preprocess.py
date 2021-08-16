from __future__ import print_function
from __future__ import division
from typing import List, Union

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


def convert_to_sphinx(input_dirs: List[str], output: Union[None, str]) -> str:
    """
    Convert to sphinx
    :param input_dirs: paths to dirs with input nii files, (likely in the MION dir)
    :param output: output directory to create or populate (if None put in same dirs as input)
    :return: path to output directory
    """
    cvt = freesurfer.MRIConvert()
    for scan_dir in input_dirs:
        files = os.listdir(scan_dir)
        for f in files:
            if len(f) > 3 and f[-4:] == '.nii':
                cvt.inputs.in_file = os.path.join(scan_dir, f)
                if not output:
                    local_out = os.path.join(scan_dir, 'f.nii')
                else:
                    local_out = os.path.join(output, os.path.basename(scan_dir), 'f.nii')
                cvt.inputs.out_file = local_out
                cvt.inputs.args = "--sphinx"
                cvt.inputs.out_type = 'nii'
                cvt.cmdline
                cvt.run()

    return output


def motion_correction(source_dir: str):
    """
    preform fsl motion correction
    :param source_dir: subject dir containing scan directories with .nii files
    :return:
    """
    try:
        files = os.listdir(source_dir)
    except (FileExistsError, FileNotFoundError):
        print("could not find file")
        exit(1)
    outputs = []
    mcflt = fsl.MCFLIRT()
    for source in files:
        path = os.path.join(source_dir, source)
        mcflt.inputs.in_file = path
        mcflt.inputs.cost = 'mutualinfo'
        mcflt.inputs.out_file = 'moco.nii'
        mcflt.cmdline
        out = mcflt.run()
        outputs.append(out)
    return outputs

