import shutil
from typing import List, Union

import numpy as np
import pandas as pd
from subprocess import call
from preprocess import _create_dir_if_needed
import sys
import os
import glob
from multiprocessing import Pool
import re


def unpack(inDir,outDir,adj=False,dirdepth=5, nifti_name='f', ts_only=False):
    """
    unpack unpacks DICOM data from the scanner into the NIFTI format.
    :param nifti_name: what to name output nifti
    :param inDir: Input directory where the dicom files live.
    :param outDir: Directory where the NIFTI files will go.
    :param adj: True/False. Are images from the same series always in the same exp? Default False
    :param dirdepth: 1-9. How many folders of depth to convert NIFTI files? Default 5.
    :return: 'Completed'
    """
    assert isinstance(inDir,str), 'Parameter inDir={} not of <class: "str">'.format(inDir)
    assert isinstance(outDir,str), 'Parameter outDir={} not of <class: "str">'.format(outDir)
    assert isinstance(adj,bool), 'Parameter adj={} not of <class: "bool">'.format(adj)
    assert isinstance(dirdepth,int), 'Parameter dirdepth={} not of <class "int">'.format(dirdepth)

    if adj: # Prepare the 'a' argument for dcm2niix
        a = 'y'
    else:
        a = 'n'

    if ts_only:
        i = 'y'
    else:
        i = 'n'

    if nifti_name is not None:
        cmd = 'dcm2niix -o {} -i {} -d {} -z y -f {} {}'.format(outDir, i, dirdepth, nifti_name, inDir)
    else:
        cmd = 'dcm2niix -o {} -i {} -d {} -z y {}'.format(outDir, i, dirdepth, inDir)
    print(cmd)
    call(cmd, shell=True)
    return 'Completed'


def unpack_other_list(inDir: str, outDir: str, ima_numbers: List[int], session_id):
    """
    Unpack a list of images (non-runs) to a target directory.
    :param inDir: A directory from a day of scanning containing mulitiple runs
    :param outDir: target directory of session
    :param ima_numbers: a list of imas
    :param session_id: The session id
    :return:
    """
    ima_dirs = os.listdir(inDir)
    session_dir = os.path.join(outDir,str(session_id))
    image_out_dir = _create_dir_if_needed(session_dir,'images_other')
    to_unpack = []
    imgs = []
    tkn_idx = None

    print('ima numbers ', ima_numbers)
    for ima in ima_dirs:
        if ima[0] == '.':
            continue
        tkns = ima.split('_')
        if tkn_idx is None:
            print(tkns)
            tkn_idx = int(input("enter 0 indexed index of token denoting IMA number "
                                "(this is usually the 0th index but might not be always.)"))
        if len(tkns) < tkn_idx:
            continue
        try:
            this_ima_num = int(tkns[tkn_idx])
        except Exception:
            print("failed to grab", ima)
            continue

        if this_ima_num in ima_numbers:
            print(ima)
            imgs.append(ima)
            to_unpack.append((os.path.join(inDir,ima),image_out_dir,False,2,ima))
    with Pool() as p:
        p.starmap(unpack, to_unpack)
    return imgs


def unpack_run_list(inDir: str, outDir: str, run_numbers: List[int], session_id, nifti_name: str = 'f'):
    """
    Unpack a list of runs to target dir.
    :param nifti_name: what to name output nifti
    :param inDir: A directory from a day of scanning containing multiple runs
    :param outDir: target directory for new nifti folders
    :param run_numbers: list of good run number
    :return: a list of the dirs created containing nifti files
    """
    if os.path.exists("./" + str(session_id) + "tmp_unpack"):
        shutil.rmtree("./" + str(session_id) + "tmp_unpack")
    os.mkdir("./" + str(session_id) + "tmp_unpack")
    print(os.path.abspath("./" + str(session_id) + "tmp_unpack"))
    unpack(inDir, os.path.abspath("./" + str(session_id) + "tmp_unpack"), False, 2, None, True)

    unpacked_epis = os.listdir("./" + str(session_id) + "tmp_unpack")
    _create_dir_if_needed(outDir, str(session_id))
    tkn_idx = None
    fdirs = []
    for run in unpacked_epis:
        if run[0] == '.' or '.nii' not in run:
            continue
        tkns = re.split('-|_|\.', run)
        if tkn_idx is None:
            print(tkns)
            tkn_idx = int(input("enter 0 indexed index of token denoting run number "
                                "(usually is 0 but seems to unexpectedly change sometimes.)"))

        if len(tkns) < tkn_idx:
            continue
        try:
            this_run_num = int(tkns[tkn_idx])
        except Exception:
            print("Failed to grab", run)
            continue

        if this_run_num in run_numbers:
            _create_dir_if_needed(os.path.join(outDir, session_id), str(this_run_num))
            local_out = os.path.join(outDir, session_id, str(this_run_num))
            fdirs.append(local_out)
            shutil.copy(os.path.join("./" + str(session_id) + "tmp_unpack", run), os.path.join(local_out, nifti_name + '.nii.gz'))
    shutil.rmtree("./" + str(session_id) + "tmp_unpack")
    return fdirs

    # run_dirs = os.listdir(inDir)
    # to_unpack = []
    # f_dirs = []
    # _create_dir_if_needed(outDir, str(session_id))
    # tkn_idx = None
    # for run in run_dirs:
    #     if run[0] == '.' or '.dcm' not in run :
    #         continue
    #
    #     if '_' in run:
    #         tkns = run.split('_')
    #     elif '-' in run:
    #         tkns = run.split('-')
    #     else:
    #         tkns = run.split()
    #     if tkn_idx is None:
    #         print(tkns)
    #         tkn_idx = int(input("enter 0 indexed index of token denoting run number "
    #                             "(usually is 0 but seems to unexpectedly change sometimes.)"))
    #     if len(tkns) < tkn_idx:
    #         continue
    #     this_run_num = int(tkns[tkn_idx])
    #     if this_run_num in run_numbers:
    #         if 't2' in run:
    #             local_out = os.path.join(outDir, session_id)
    #             name = 't2'
    #         else:
    #             _create_dir_if_needed(os.path.join(outDir, session_id), str(this_run_num))
    #             local_out = os.path.join(outDir, session_id, str(this_run_num))
    #             f_dirs.append(local_out)
    #             name = nifti_name
    #         to_unpack.append((os.path.join(inDir, run), local_out, False, 2, name, True))
    # with Pool() as p:
    #     p.starmap(unpack, to_unpack)
    # return f_dirs


def scan_log(inF, outF, re_scan=True):
    if not os.path.exists(outF):
        os.mkdir(outF)
    if os.path.exists(os.path.join(outF, 'scan.info')) and not re_scan:
        print("dicom scan exists, aborting...", sys.stderr)
        return
    print("Generating scan.info from dicom headers...")
    scan_log_cmd(inF, outF)


def scan_log_cmd(inF, outF):

    cmd = 'dcmunpack -src %s -scanonly %s -index-out %s'%(inF,os.path.join(outF,'scan.info'),os.path.join(outF,'dcm.index.dat'))
    waitMsg = 'Please wait, scanning can take a handful of minutes'
    print(waitMsg)
    call(cmd,shell=True)
    print('Done')



def create_dir_structure(input_dir: str, output_root_dir: str) -> List[str]:
    """
    Takes a single directory full of all runs and converts it to freesurfer standard nested directory structure
    :return: a list of the new directories for each functional run
    """
    files = os.listdir(input_dir)
    count = 0
    functional_dirs = []
    _create_dir_if_needed(os.path.dirname(output_root_dir), os.path.basename(output_root_dir))
    _create_dir_if_needed(output_root_dir, 'mri')
    _create_dir_if_needed(output_root_dir, 'surf')
    _create_dir_if_needed(output_root_dir, 'stimuli')
    _create_dir_if_needed(output_root_dir, 'analysis_out')
    func_dir = os.path.join(output_root_dir, 'functional')
    _create_dir_if_needed(output_root_dir, 'functional')
    for f in files:
        if '.nii' in f:
            fid = None
            par_id = None
            items = f.split('_')
            for i in range(len(items) - 1):
                tkn_day = items[i]
                tkn_run = items[i + 1]
                if tkn_run.isnumeric() and tkn_day.isnumeric():
                    par_id = str(tkn_day)
                    fid = str(tkn_run)
            if not fid:
                fid = str(count)
                par_id = 'unlabelled'
                count -= 1
            par_dir = os.path.join(func_dir, par_id)
            _create_dir_if_needed(func_dir, par_id)
            run_dir = os.path.join(par_dir, fid)
            _create_dir_if_needed(par_dir, fid)
            ext = '.'.join(f.split('.')[1:])
            while len(ext) > 0:
                if ext not in ['nii', 'nii.gz']:
                    ext = ext[1:]
                    continue
                else:
                    break
            if len(ext) == 0:
                raise RuntimeError('Unpacked files must be of type .nii or .nii.gz, not ' + ext)
            func_path = os.path.join(run_dir, 'f.' + ext)
            shutil.copy(os.path.join(input_dir, f), func_path)
            functional_dirs.append(run_dir)
    return functional_dirs



        
    



