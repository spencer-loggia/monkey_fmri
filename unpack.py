
import numpy as np
import pandas as pd
from subprocess import call
import sys
import os
import glob

def scan_log(inF,outF):
    if os.path.exist(os.path.join(outF, 'scan.info')):
        while True:
            check = input('scan.info already exists./n Would you like to scan the dicom files anyway? /n This may take several minutes. (y/n)')
            if check == 'y':
                scan_log_cmd(inF,outF)
                break
            elif check == 'n':
                return 'Dicom Scan Aborted'
            else:
                print('Please enter either \"y\" or \"n\".')


def scan_log_cmd(inF, outF):
    cmd = 'dcmunpack -src %s -scanonly %s -index-out dcm.index.dat'%(inF,outF)
    #call(cmd,shell=True)
    print(cmd)
    waitMsg = 'Please, wait, scanning can take a handful of minutes'
    print(waitMsg)

# def unpack_tsv():

