
import numpy as np
import pandas as pd
from subprocess import call
import sys
import os
import glob

def scan_log(inF,outF):
    if os.path.isfile(os.path.join(outF, 'scan.info')):
        while True:
            check = input('scan.info already exists. Would you like to scan the dicom files anyway? This may take several minutes. (y/n)')
            if check == 'y':
                scan_log_cmd(inF,outF)
                break
            elif check == 'n':
                return 'Dicom Scan Aborted'
            else:
                print('Please enter either \"y\" or \"n\".')


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
            runlist += '\ -run %s %s nii f.nii'%(x,HDR)
        end
        cmd = 'dcmunpack -src . -index-in'
            
        
    


# def unpack_tsv():

