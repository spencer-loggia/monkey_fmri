from typing import List, Union

from nipype.interfaces import freesurfer


def convert_dcm_to_sphinx(input_dirs: List[str], output: Union[None, str]) -> str:
    """
    Convert dicom to sphinx
    :param input_dirs: paths to dirs with input nii files, (likely in the MION dir)
    :param output: output directory to create or populate (if None put in same dirs as input)
    :return: path to output directory
    """
    cvt = freesurfer.DICOMConvert()
    for scan_dir in input_dirs:
        cvt.inputs.dicom_dir = scan_dir
        if not output:
            output = scan_dir
        cvt.inputs.base_output_dir = output
        cvt.inputs.args = "--sphinx"
        cvt.cmdline
    return output

def motion_correction
