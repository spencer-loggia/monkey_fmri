
import numpy as np
import pandas as pd
from subprocess import call
import sys
import os
import glob


def scan_log(inF, outF, re_scan=True):
    if not os.path.isfile(outF):
        os.mkdir(outF)
    if os.path.isfile(os.path.join(outF, 'scan.info')) and not re_scan:
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


def unpack(inF,outF,runs,HDR):
    runlist = ''
    HDRs = ['mion','bold']
    if HDR.lower() in HDRs:
        for x in runs:
            runlist += '\ -run %s %s nii f.nii '%(x,HDR)
        inxF = os.path.join(outF,'dcm.index.dat')
        cmd = 'dcmunpack -src %s -index-in %s -targ %s %s'%(inF,inxF,outF,runlist)
        call(cmd,shell=True)
    else:
        print('Please enter either "MION" or "BOLD"')
            
        
    



