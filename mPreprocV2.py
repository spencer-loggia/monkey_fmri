#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:27:23 2020

@author: kurtb
"""
"""
At the center of the toolbox is the fmri_data object, which is loosely based
on the fmri_data object from the Canlab toolbox (https://github.com/canlab).
The primary role of this object is to load, mask, resample, and reshape fMRI
data, but many functions are designed to interact with it.
"""
import sys
sys.path.append('/Users/duffieldsj/Documents/Kurt')
import numpy as _np
import nipype
from subprocess import call #as _call
import os as _os
import nibabel as _nib
import glob as _glob
import matplotlib.pyplot as _plt
from tqdm import tqdm
import shutil
import numpy as np
import nilearn.plotting as _nip
import nilearn.image as _nli
import pandas as _pd
import json
from pprint import pprint as _pprint
import time
monkeyDeepBrainF = '/Users/duffieldsj/Documents/GitHub/PREEMACS/monkey_deepbrain'
import gzip
import shutil
#%%




def itkSnapManual(anatP,funcP,outF):
    '''manually register to anat'''
    print('-'*30)
    s = 'do manual reg with ITK SNAP: \n\nmain image: \n%s'%(anatP)
    s+='\n\nsecondaryimage: \n%s'%(funcP)
    step1NiiP = outF+'/itkManual.nii.gz'
    step1TxtP = outF+'/itkManual.txt'
    s+='\n\nsave nii as: \n%s'%(step1NiiP)
    s+='\n\nand transform as: \n%s'%(step1TxtP)
    print(s)
    done = 'y' in input('done? (y/n): ')
    return step1TxtP,step1NiiP


def antsCoReg(fixedP,movingP,outP,initialTrsnfrmP,ltrns=['Affine','SyN'],n_jobs=2):
    outF = _os.path.split(outP)[0]
    _os.chdir(outF)
    antsCoreg(fixedP,
                 movingP,
                 outP,
                 initialTrsnfrmP=None, # we are working on resampled img
                 across_modalities=True,
                 outPref='antsReg',
                 transforms=ltrns,
                 run=True,n_jobs=n_jobs)
    frwdTrnsP =  _glob.glob(outF+'/antsRegComposite.h5')[0]
    return frwdTrnsP


def antsCoreg(fixedP,movingP,outP,initialTrsnfrmP=None,
              across_modalities=False,outPref='antsReg',
              transforms=['Affine', 'SyN'],run=True,n_jobs=10):
    _os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = '%d'%n_jobs
    from nipype.interfaces.ants import Registration
    reg = Registration()
    reg.inputs.fixed_image = fixedP
    reg.inputs.moving_image = movingP
    reg.inputs.output_transform_prefix = outPref
    reg.inputs.interpolation= 'BSpline'
    reg.inputs.transforms = transforms
    reg.inputs.transform_parameters = [(2.0,), (0.25, 0.25, 0.0)]
    reg.inputs.number_of_iterations = [[400, 300], [300, 300, 50]][:len(transforms)]
    reg.inputs.dimension = 3
    if initialTrsnfrmP!=None:
        reg.inputs.initial_moving_transform = initialTrsnfrmP
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = [['Mattes','MI'][1]]*2
    reg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    reg.inputs.radius_or_number_of_bins = [32]*2
    reg.inputs.sampling_strategy = ['Random', None][:len(transforms)]
    reg.inputs.sampling_percentage = [0.05, None][:len(transforms)]
    reg.inputs.convergence_threshold = [1.e-8, 1.e-9][:len(transforms)]
    reg.inputs.convergence_window_size = [20]*2
    reg.inputs.smoothing_sigmas = [[1,0], [2,1,0]][:len(transforms)]
    reg.inputs.sigma_units = ['vox'] * 2
    reg.inputs.shrink_factors = [[2,1], [3,2,1]][:len(transforms)]
    reg.inputs.use_estimate_learning_rate_once = [True]*len(transforms)
    reg.inputs.use_histogram_matching = [across_modalities==False]*len(transforms) # This is the default
    reg.inputs.output_warped_image = outP
    reg.inputs
    print(reg.cmdline)
    if run:
        reg.run()


def readJson(jsonP):
    fid = open(jsonP, "r")
    dJson = json.load(fid)
    fid.close()
    return dJson

def vol2surf(subjDir, subj,hem,volP,outMghP):
    cmd = cmd = 'SUBJECTS_DIR=%s; '%subjDir
    cmd += 'mri_vol2surf --mov %s --regheader %s --projdist-avg 0 1 0.1 '%(volP,
                                                                   subj,)
    cmd += '--interp nearest --hemi %s --out %s --noreshape'%(hem,
                                                              outMghP)
    print(cmd)
    call(cmd,shell=True)

def freeview_surface_and_overlay_bothhemi(surfPs,overlayPs,thrMinMax=[3,10]):
    assert type(surfPs)==list,'surfPs must be a list of paths to surface files you wish to load'
    assert type(overlayPs)==list, 'overlayPs must be a list of paths to surface files you wish to load'
    if len(surfPs)>len(overlayPs):
        pad_len = len(surfPs)-len(overlayPs) # Need to pad overlayPs
        overlayPs.append(['']*pad_len)
    cmd = 'freeview -f'
    for surf, overlay in zip(surfPs,overlayPs):
        if len(overlay)>0:
            cmd+=' %s:overlay=%s:overlay_threshold=%d,%d'%(surf,
                                                                overlay,
                                                                thrMinMax[0],
                                                                thrMinMax[1]
                                                                )
        else:
            cmd+=' %s:overlay=%s'%surf

    print(cmd)
    call(cmd,shell=True)



def freeview_surface_and_overlay(surfP,overlayP,thrMinMax=[3,10]):
    cmd = 'freeview -f %s:overlay=%s:overlay_threshold=%d,%d'%(surfP,
                                                                    overlayP,
                                                                    thrMinMax[0],
                                                                    thrMinMax[1]
                                                                    )
    print(cmd)
    call(cmd,shell=True)


def slice_timing_spm(inP):
    from nipype.interfaces.spm import SliceTiming
    st = SliceTiming()
    st.inputs.in_files = inP
    st.inputs.num_slices = 32
    st.inputs.time_repetition = 6.0
    st.inputs.time_acquisition = 6. - 6./32.
    st.inputs.slice_order = list(range(32,0,-1))
    st.inputs.ref_slice = 1
    st.run()


def slice_timing_fsl(funcP,fJsP,TR,outP):
    '''b/c multband, we need to provide slice times. fsl expects in fraction TR.
    We read from json the slice times, divide by TR, subtract by .5,
    write to slice_timing file'''
    stcF,fN=_os.path.split(outP)
    a_file = open(fJsP, "r")
    json_object = json.load(a_file)
    a_file.close()
    sliceTimes = json_object["SliceTiming"]
    print('\nsliceTimes = ',sliceTimes)
    dfSlice = _pd.DataFrame(sliceTimes)
    dfSlice/=TR
    dfSlice-= .5
    txtP = stcF+'/sliceTimesFracTr_fsl_%s.txt'% _os.path.basename(fN)
    txtP = txtP.replace('.nii.gz','')
    dfSlice.to_csv(txtP,header=None,index=None)

    _os.chdir(stcF)

    import nipype.interfaces.fsl as fsl
    st = fsl.SliceTimer()
    st.inputs.in_file = funcP
    st.inputs.custom_timings = txtP
    st.inputs.out_file = outP
    st.inputs.time_repetition = TR
    print(st.cmdline)
    return st.run()


def coreg_spm(fP,anatP,lFollowers=None,jobtype='estwrite'):
    '''stimating cross-modality rigid body alignment (three translations
    and three rotations about the different axes)'''

    dsAnat = _nli.resample_img(anatP, _nib.load(fP).affine)

    anatF = _os.path.split(anatP)[0]
    dsAnatP = _os.path.join(anatF,'dsAnat.nii')
    dsAnat.to_filename(dsAnatP)
    # run
    import nipype.interfaces.spm as spm
    coreg = spm.Coregister()
    coreg.inputs.target = dsAnatP
    coreg.inputs.source = fP
    coreg.inputs.write_mask=True
    coreg.inputs.out_prefix = 'a'
    coreg.inputs.jobtype=jobtype
    if lFollowers != None:
        coreg.inputs.apply_to_files=list(lFollowers)

    print(coreg.cmdline)
    coreg.run()

    if not 'rite' in jobtype:
        # only the header has changed, no resampling, so we still have ims
        # preprended stih st_, and we don't have a meanP
    #        mcPs = glob.glob(mcF+'/st_*.nii') #[0]
    #        motionParamsPs = glob.glob(mcF+'/rp_st*.txt')#[0]
        dat = fmri_data([lFollowers[0]],'background')
        f,n = _os.path.split(lFollowers[0])
        meanP = '%s/mean_%s'%(f,n)
        dat.unmasker(dat.dat.mean(axis=0)).to_filename(meanP)


def antsCoreg(fixedP,movingP,outP,initialTrsnfrmP=None,
              across_modalities=False,outPref='antsReg',
              transforms=['Affine', 'SyN'],run=True,n_jobs=10):
    _os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = '%d'%n_jobs
    from nipype.interfaces.ants import Registration
    reg = Registration()
    reg.inputs.fixed_image = fixedP
    reg.inputs.moving_image = movingP
    reg.inputs.output_transform_prefix = outPref
    reg.inputs.interpolation= 'BSpline'
    reg.inputs.transforms = transforms
    reg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    #reg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]][:len(transforms)]
    reg.inputs.number_of_iterations = [[1500, 200], [200, 100, 60]][:len(transforms)]
    reg.inputs.dimension = 3
    if initialTrsnfrmP!=None:
        reg.inputs.initial_moving_transform = initialTrsnfrmP
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initialize_transforms_per_stage = False
    reg.inputs.metric = [['Mattes','MI'][1]]*2
    reg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    reg.inputs.radius_or_number_of_bins = [32]*2
    reg.inputs.sampling_strategy = ['Random', None][:len(transforms)]
    reg.inputs.sampling_percentage = [0.05, None][:len(transforms)]
    reg.inputs.convergence_threshold = [1.e-8, 1.e-9][:len(transforms)]
    reg.inputs.convergence_window_size = [20]*2
    reg.inputs.smoothing_sigmas = [[1,0], [2,1,0]][:len(transforms)]
    reg.inputs.sigma_units = ['vox'] * 2
    reg.inputs.shrink_factors = [[2,1], [3,2,1]][:len(transforms)]
    reg.inputs.use_estimate_learning_rate_once = [True]*len(transforms)
    reg.inputs.use_histogram_matching = [across_modalities==False]*len(transforms) # This is the default
    reg.inputs.output_warped_image = outP
    reg.inputs
    print(reg.cmdline)
    if run:
        reg.run()

def motion_correction_fsl(funcP,outP):
    from nipype.interfaces import fsl
    mcflt = fsl.MCFLIRT()
    mcflt.inputs.in_file=funcP
    mcflt.inputs.cost='mutualinfo'
    mcflt.inputs.out_file= outP
    mcflt.inputs.output_type = 'NIFTI_GZ'
    mcflt.inputs.save_plots = True
    mcflt.inputs.save_rms = False
    mcflt.inputs.stats_imgs = False
    mcflt.inputs.dof = 6
    res = mcflt.run()
    return res


def resamp_clip_anat_to_func(anatP,funcP):
    return _nli.resample_to_img(anatP,
                               funcP,
                               clip=True)





def write_md(ps,outP):
    ''' can be used as first step in generating html report'''
    fid=open(outP,'w')
    for p in ps:
        title = _os.path.basename(p)
        fid.write('## %s\n'%title)
        fid.write('![](%s)\n'%p)
        fid.write('\n')
    fid.close()


def qa_motion_movie(ps,qaF, outN=None):
    _plt.close('all')
    fig=_plt.figure(figsize=(8,3))
    ax=fig.add_subplot(1,1,1)

    tmpF = qaF+'/tmp'
    if _os.path.exists(tmpF):
        shutil.rmtree(tmpF)
    _os.makedirs(tmpF)

    i=-1
    for irun,p in enumerate(ps):
        dat = fmri_data([p],'background')
        im4d=dat.unmasker(dat.dat)
        del dat
        if not _os.path.exists(tmpF):
            _os.makedirs(tmpF)
        for img in tqdm(_nli.iter_img(im4d),desc='preparing r%.3d movie'%(irun+1)):
            i+=1
            fig.clear()
            ax.cla()
            if i==0:
                m=img.get_fdata()
                vmin,vmax=0,np.max(m)
                del m
                _nip.plot_epi(img,
#                             cut_coords=(0,0,0),
        #                     display_mode='y',
                             figure=fig,
                             output_file=tmpF+'/%.3d.png'%i,
                             title='run=%d, TR=%d'%(irun+1,i),
                             draw_cross=False,)
            else:
                _nip.plot_epi(img,
                             cut_coords=(0,0,0),
        #                     display_mode='y',
                             figure=fig,
                             output_file=tmpF+'/%.3d.png'%i,
                             title='run=%d, TR=%d'%(irun+1,i),
                             draw_cross=False,
                             vmin=vmin,
                             vmax=vmax,)
    if outN==None:
        n = _os.path.basename(p).replace('.nii','').replace('.gz','')
    else:
        n=outN
    cmd = 'cd %s; '%tmpF
    vidP = qaF+'/%s.mp4'%n
    if _os.path.exists(vidP):
        _os.remove(vidP)
    cmd+="ffmpeg -framerate FPS -pattern_type glob -i '*.png'   -c:v libx264 -pix_fmt yuv420p OUTPATH"
    cmd = cmd.replace('FPS','%d'%5)
    cmd = cmd.replace('OUTPATH',vidP)
    call(cmd, shell=True)
#    shutil.rmtree(tmpF)
    return vidP


def antsMC(antsMotF,inP):
    cmd = 'cd %s; ./antsMC.sh %s'%(antsMotF,inP)
    print(cmd)
    call(cmd,shell=True)


def plot_motion(df,title,outP):
    _plt.figure(figsize=(8,6))
    _plt.subplot(2,1,1)
    _plt.plot(df.loc[:,:2])
    _plt.legend(['x','y','z'])
    _plt.title('translations (mm)')

    _plt.subplot(2,1,2)
    _plt.plot(df.loc[:,3:])
    _plt.legend(['x','y','z'])
    _plt.title('rotations (deg)')
    _plt.suptitle(title)
    _plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _plt.savefig(outP)




def freeview_auto_anat(subjF,lVolPs):
    ''' used to be called freeview_auto'''
    lsurf = []
    for hem in ['lh','rh']:
        for sf in ['inflated','pial','white']:
            p = _os.path.join(subjF,'surf','%s.%s'%(hem,sf))
            lsurf.append(_glob.glob(p)[0])
    freeview(lVolPs,lsurf)


def ants_brainExtraction(t1P,templateP,templateProbMask,out_prefix='antsBrnExtrct'):
    f=_os.path.split(t1P)[0]
    _os.chdir(f)
    from nipype.interfaces.ants.segmentation import BrainExtraction
    brainextraction = BrainExtraction()
    brainextraction.inputs.dimension = 3
    brainextraction.inputs.anatomical_image =t1P
    brainextraction.inputs.brain_template = templateP
    brainextraction.inputs.brain_probability_mask = templateProbMask
#    brainextraction.inputs.out_prefix = out_prefix
    cmd =brainextraction.cmdline
    cmd = 'cd %s; %s'%(f,cmd)
    call(cmd,shell=True)
#    brainextraction.run()

def segment_atropos(anatP,priorProbImP):
    from nipype.interfaces.ants import Atropos
    at = Atropos(
        dimension=3, intensity_images=anatP,
        number_of_tissue_classes=2, likelihood_model='Gaussian', save_posteriors=True,
        mrf_smoothing_factor=0.2, mrf_radius=[1, 1, 1], icm_use_synchronous_update=True,
        maximum_number_of_icm_terations=1, n_iterations=5, convergence_threshold=0.000001,
        posterior_formulation='Socrates', use_mixture_model_proportions=True)

    at.inputs.initialization = 'PriorProbabilityImages'
    at.inputs.prior_image = priorProbImP
    at.inputs.bspline=100
    at.inputs.prior_weighting = 0.8
    at.inputs.prior_probability_threshold = 0.0000001
    at.cmdline
    at.run()

def make_spmBrainMask(cPs,outP,maskType='background'):
    '''using the c1, c2 and c3 images (or a subset), create full brain mask.
    write to outP'''
    dat = fmri_data(cPs,maskType)
    vsum = dat.dat.sum(axis=0)
    vmask = (vsum>.0).astype(np.float32)
    dat.unmasker(vmask).to_filename(outP)


def gzip_shutil(inP,outP=None):
    if outP == None:
        outP = inP+'.gz'
    with open(inP, 'rb') as f_in:
        with gzip.open(outP, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return outP


def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)
    return dest_filepath


def gzip_to_nii(gzP):
    ''' checks if unzipped version exists in dir. if not, creates it.'''
    niiP = gzP.replace('.gz','')
    if not _os.path.exists(niiP):
        with gzip.open(gzP, 'rb') as f_in:
            with open(niiP, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return niiP


def ksurfing_safari(inP,steps='precon_all',ants=True):
    '''
    inP: path to high quality t1

    steps:

        precon_all reconstructs the cortical surface from a
    single whole body image.

     precon_1 Only performs brain extraction i.e step 1

     precon_2 performs steps Denoising, Segmentation, WM fill and
     generates surfaces

     precon_3 performs only WM filling and Generates surfaces
     precon_art meant for art projects or figures in which the non_cort
     mask is included. Only run after precon_2

     precon art is not meant to actually  give any statistics or
     information on the surface. visualization purposes only
     '''
    f,n=_os.path.split(inP)
    _os.chdir(f)
    if ants:
        cmd = 'cd %s; surfing_safari.sh -i %s -r %s -a NIMH_mac  -n -s'%(f,inP,steps)
    else:
        cmd = 'cd %s; surfing_safari.sh -i %s -r %s -a NIMH_mac  -n'%(f,inP,steps)
    print(cmd)
    call(cmd,shell=True)

def mri_nu_correct(p):
    f,n= _os.path.split(p)
    outP = _os.path.join(f,'ro_%s'%n)
    cmd= 'mri_nu_correct.mni --i %s --o %s --distance 24'%(p,outP)
    call(cmd,shell=True)
    return outP


def average_structurals_freesurfer(ps,outP,nuCorrect=True):
    '''optionally mri_nu_correct, motion correct/ average'''
    if nuCorrect:
        motionInput = []
        for p in ps:
            motionInput.append(mri_nu_correct(p))
    else:
        motionInput=ps

    cmd="mri_motion_correct2 -o %s"%outP
    for p in motionInput:
        cmd+= ' -i %s'%p
    call(cmd,shell=True)


def flirt(movingP, fixedP, outP, transP=None, dof=12,outType="NIFTI_GZ"):
    from nipype.interfaces import fsl
    flt = fsl.FLIRT(bins=640, cost_func='mutualinfo')
    flt.inputs.in_file = movingP
    flt.inputs.reference = fixedP
    flt.inputs.out_file = outP
    flt.inputs.dof=dof
    if transP!=None:
        flt.inputs.in_matrix_file = transP
    flt.inputs.output_type = outType
    print(flt.cmdline)
    flt.run()


def combine_T1_T2(t1P,t2P,greyP,):
    '''  https://pubmed.ncbi.nlm.nih.gov/25533337

    Their algo results in v. noisy image! -- Don't use!

    "The combined image (CI) was calculated as
    CI = (T1w - sT2w)/(T1w + sT2w),
    where sT2w is the scaled T2-weighted image. The scaling factor
    was calculated to adjust the gray- matter (GM) voxel intensities
    in the T2w image so that their median value equaled that of the
    GM voxel intensities in the T1w image."

    t1P & t2P: should be preprocessed nu-corrected t1 and t2 images:
        - preproc= within each modality: nu-correction, averaging
        - The t1 an t2 should be coreghistered.

    greyp: grey-matter path calculated from the t1 beforhand'''
    makeAffinesMatch([t1P,t2P]) # upsample t2 to t1
    # make mask binary
    datGrey=fmri_data([greyP],'background')
    vMask = (datGrey.dat>.1).astype(_np.float32)
    imMask = datGrey.unmasker(vMask)

    # calculate scaling value
    '''he scaling factorwas calculated as MG_T1/MG_T2, where
    MG_T1and MG_T2are the median values of GM voxels of T1w and
    T2w,respectively. '''
    dat = fmri_data([t1P,t2P],imMask)
    _plt.figure()
#    _plt.subplot(2,1,1)
    _plt.hist(dat.dat[0,:],bins=30,density=True)
    _plt.hist(dat.dat[1,:],bins=30,density=True)

    medT1 = _np.median(dat.dat[0,:])
    medT2 = _np.median(dat.dat[1,:])
    scalingFactor = _np.divide(medT1,medT2)

    # test scaling
    vT2_scaled = dat.dat[1,:]*scalingFactor
#    _plt.subplot(2,1,2)
    _plt.hist(vT2_scaled,bins=30,density=True)
    _plt.legend(['t1 grey ','t2 grey (before scaling', 't2 grey after scaling'])


    # load both volumes full brain
    dat = fmri_data([t1P,t2P],'background')

    # apply scaling to t2 & save
    sT2w = dat.dat[1,:]*scalingFactor
    f,n=_os.path.split(t2P)
    sT2P = _os.path.join(f,'scaled_%s'%n)
    dat.unmasker(sT2w).to_filename(sT2P)

    # combine T1 and T2
    T1w = dat.dat[0,:]
    CI = (T1w - sT2w)/(T1w + sT2w)

    ciP = _os.path.join(f,'combined_avT1_avT2.nii.gz')
    dat.unmasker(CI).to_filename(ciP)
    return ciP



#def combine_nuCorrectedIms(lRoP,outP,targP=None,transforms='r'):
#    '''if targP is None, will align each volume to that with the largest
#    file-size...'''
#    if targP==None:
#        imax= np.argmax([_os.path.getsize(p) for p in lRoP])
#        targP = lRoP[imax]
#        lRoP = [p for p in lRoP if not p==targP]
#    lp = [targP]
#    for i in range(0,len(lRoP)):
#        lp.append(antsRegistrationSynQuick(targP,lRoP[i],transforms=transforms))
#
#    print('averaging bias-field corrected, coregistered T1s')
#    dat = fmri_data(lp,'background')
#    mnIm = dat.unmasker(dat.dat.mean(axis=0))
#    mnIm.to_filename(outP)


def combine_wInModality(ps,targP=None,nuCorrect=True,outF=None,imType=None,transforms='r'):
    '''to improve reconall, we average t1s within an animal. This
    function peforms 1. bias field correction. 2. coregistration
    3. averaging'''
    if outF==None:
        outF = _os.path.split(ps[1])[0]

    #  bias field correct
    if nuCorrect:
        coregInput=[]
        for p in ps:
            coregInput.append(mri_nu_correct(p))
    else:
        coregInput = ps


    if targP==None: # 2nd coreg to biggest vol
        imax= np.argmax([_os.path.getsize(p) for p in coregInput])
        targP = coregInput[imax]
        coregInput = [p for p in coregInput if not p==targP]
        lp = [targP]
    else:
        lp = []

    for i in range(0,len(coregInput)):
        lp.append(antsRegistrationSynQuick(targP,coregInput[i],transforms=transforms))

    # average
    print('averaging bias-field corrected, coregistered T1s')
    try:
        dat = fmri_data(lp,'background')
    except:
        import code
        code.interact(local=locals())
    if len(dat.dat.shape)>1:
        vmn = dat.dat.mean(axis=0)
    else:
        vmn=dat.dat
    mnIm = dat.unmasker(vmn)

    if imType == None:
        if 'T1' in lp[0]:
            outP = outF+'/avT1.nii.gz'
        elif 'T2' in lp[1]:
            outP = outF+'/avT2.nii.gz'
        else:
            raise ValueError('unknown: T1 or T2?')
    else:
        outP = outF+'/av%s.nii.gz'%imType

    mnIm.to_filename(outP)
    return outP


def N4BiasFieldCorrection(structuralP,output_image,outBiasP,num_threads=1):
    '''    cmd = 'N4BiasFieldCorrection -d 3 -b [100] -i ${bidsdir}/${subId}/anat/*.nii.gz  -o [bias_corrected.nii.gz,bias_image.nii.gz]
'''
    from nipype.interfaces.ants import N4BiasFieldCorrection
    n4 = N4BiasFieldCorrection()
    n4.inputs.dimension = 3
    n4.inputs.input_image = structuralP
    n4.inputs.bspline_fitting_distance = 100
#    n4.inputs.shrink_factor = 3
#    n4.inputs.n_iterations = [50,50,30,20]
    n4.inputs.save_bias = True
    n4.inputs.bias_image = outBiasP
    n4.inputs.output_image = output_image
    n4.inputs.num_threads = num_threads
    n4.cmdline
    n4.run()


def mask_an_image(in_file,mask_file,out_file,output_type='NIFTI_GZ'):
    '''input are string-like paths
    PREEMAC:
    fslmaths ${maskDir}/${subId}/brain_mask.nii.gz  -thr 0.0001 out_mask.nii.gz
    fslmaths bias_corrected.nii.gz -mul out_mask.nii.gz out_file.nii.gz

    fslmaths ${maskDir}/${subId}/brain_mask.nii.gz  -thr 0.0001 out_mask.nii.gz
    fslmaths bias_corrected.nii.gz -mul out_mask.nii.gz out_file.nii.gz'''
    im = _nib.load(mask_file)
    m = im.get_fdata()
    if not len(np.unique(m))==2:
        m[m>.0001] = 1.
        m[m<=.0001]=0.
        im = _nib.Nifti1Image(m,im.affine)
        im.to_filename(mask_file)

    from nipype.interfaces.fsl.maths import ApplyMask
    ma=ApplyMask()
    ma.inputs.in_file=in_file
    ma.inputs.mask_file=mask_file
    ma.inputs.nan2zeros = True
    ma.inputs.out_file=out_file
    ma.inputs.output_type=output_type
    ma.cmdline
    ma.run()


def mrVoxresize(in_file,out_file):
    '''${MRTRIX_DIR}/mrresize -voxel 0.5 $path_job/T1_preproc.nii.gz $path_job/T1_preproc.nii.gz -force
    '''
    import nipype.interfaces.mrtrix3 as mrt
    image_resize = mrt.MRResize()
    image_resize.inputs.in_file = in_file
    image_resize.inputs.voxel_size = (0.5, 0.5, 0.5)
    image_resize.inputs.out_file = out_file
    image_resize.cmdline
    image_resize.run()


def mrImresize(in_file,out_file):
    '''${MRTRIX_DIR}/mrresize -voxel 0.5 $path_job/T1_preproc.nii.gz $path_job/T1_preproc.nii.gz -force
'''
    import nipype.interfaces.mrtrix3 as mrt
    image_resize = mrt.MRResize()
    image_resize.inputs.in_file = in_file
    image_resize.inputs.image_size = (256, 256, 256)
    image_resize.inputs.out_file = out_file
    image_resize.cmdline
    image_resize.run()


def fsey(ps):
    '''open fsleyes with a list of images'''
    cmd = 'fsleyes'
    for p in ps:
        cmd+=' %s'%p
    print(cmd)
    call(cmd,shell=True)


def dnn_skull_strip(img_path, output_dir, p=0.5):
    '''use DNN to extract brain from n4 bias corrected brain.'''
    sys.path.append(monkeyDeepBrainF)
    from deepbrain import Extractor
    img = _nib.load(img_path)

    affine = img.affine
    img = img.get_fdata()

    extractor = Extractor()

    now = time.time()
    prob = extractor.run(img)
    print("Extraction time: {0:.2f} secs.".format(time.time() - now))
    mask = prob > p
    brain_mask = (1 * mask).astype(np.uint8)
    brain_mask = _nib.Nifti1Image(brain_mask, affine)
    maskP = _os.path.join(output_dir, "brain_mask_p%2f.nii"%p)
    _nib.save(brain_mask, maskP)

    brain = img[:]
    brain[~mask] = 0
    brain = _nib.Nifti1Image(brain, affine)
    brainP = _os.path.join(output_dir, "brain_p%2f.nii"%p)
    _nib.save(brain, brainP)

    return brainP, maskP



def spm_oldSegment(anatP,tmplP,gP,wP,csfP):
    import nipype.interfaces.spm as spm
    seg = spm.Segment()
    seg.inputs.data = anatP
    seg.inputs.gm_output_type = [False,False,True]
    seg.inputs.csf_output_type = [False,False,True]
    seg.inputs.wm_output_type = [False,False,True]
    seg.inputs.tissue_prob_maps = [gP,wP,csfP]
    seg.inputs.affine_regularization= 'none'
    seg.run()


def dilateMask(im, n_iter=1,outP=None):
    if isinstance(im,str):
        im = _nib.load(im)
    from scipy import ndimage
    m = im.get_data()
    dm = ndimage.binary_dilation(m,iterations=n_iter).astype(float)
    im = _nib.Nifti1Image(dm,im.affine)
    if outP is not None:
        im.to_filename(outP)
    return im


def charm_info_to_subj(subj,charm_txtP,outF,outNiiP):
    '''outNiiP is the '''


def split_SEG(segP,subj,outF=None,run=True):
    '''freesurfer requires separate segmentation files

    subj is subject ID (e.g., 'rick', 's001', etc)'''
    f,n = _os.path.split(segP)
    if outF==None:
        outF=f
    dPrm = {'CSF':[.9,1.1],
            'GM':[1.9,2.1],
            'WM':[2.9,4.1],
            # 'subCorticalWm':[2.9,3.1],
            # 'corticalWm':[3.9,4.1],
            'brainmask':[.2, 4.4],
        }
    for k in dPrm.keys():
        outP = '%s/%s_%s.nii.gz'%(outF,subj,k)
        cmd = 'fslmaths %s -thr %.1f -uthr %.1f -bin %s'%(segP,
                                                        dPrm[k][0],
                                                        dPrm[k][1],
                                                        outP)
        if run:
            call(cmd,shell=True)
        else:
            print(cmd)


def param_to_string(dParam):
    cmd = ''
    for k in dParam.keys():
        cmd+='%s %s '%(k,dParam[k])
    return cmd

#%%
def sort_niftis(topF,subj,dfSubj,blipForward='HF',rmOld=False,scan_types=['bold','fieldmap','TOPUP','T1','T2']):
    date = topF[-8:]
    # edit this dictionary to add scan types, or change search strings (srch),
    #  which are used to filter dcm filenames.
    dFTypes = {'bold':{'srch':'ep2d*bold*','minMB':5},
               'TOPUP':{'srch':'TOPUP','minMB':.01},
               'T1':{'srch':'MPRAGE','minMB':3},
               'fieldmap':{'srch':'field_mapping', 'minMB':0},
               'T2':{'srch':'t2','minMB':.5},
               # 'anatLocalizer_noInterest':{'srch':'MR_localizer','minMB':0},
        }

    idxDfSubj = _np.max(dfSubj.index)
    for scan_type in scan_types:

        print('-'*30)
        print('%s'%scan_type)
        newF = '%s/%s'%(topF,scan_type)
        fs = _np.sort(_glob.glob(topF+'/*%s*'%dFTypes[scan_type]['srch']))
        fs = [f for f in fs if  (not 'Noise' in f) and  (not 'NOISE' in f) \
              and (not 'noise' in f)]
        fs = np.sort([f for f in fs if (not 'pre_norm' in f)])
        # sometimes, we will have aborted runs.
        # - display the file size, let the user indicate which ones should be included

        dfF = _pd.DataFrame(columns=['og','shrtOg','scan_numb','run_name','MB','newName','F','p','jp'])
        i=-1
        for f in (fs):
            # get size of nifti file inside f
            ps = _glob.glob(f+'/sph*.nii*')
            if len(ps)==0:
                ok = input('%s: no nifti files found, ignore? (y/n): '%scan_type)
                if not 'y' in ok:
                    raise ValueError('Stopping')
            for p in ps:
                i+=1
                n= _os.path.basename(p)
                dfF.loc[i,'og']= n#n[n.find('x')]
                dfF.loc[i,'shrtOg'] = n[-17:]
                dfF.loc[i,'F'] = f
                dfF.loc[i,'p'] = p
                dfF.loc[i, 'jp'] = _glob.glob(f+'/*%s*.json'%n[7:-7])[0]
                lastChars = n[-15:-7]
                if '_' in lastChars:
                    try: # t2 ends in ph
                        numb='%.3d'%int(lastChars[lastChars.find('_')+1:])
                    except:
                        numb='%s'%lastChars[lastChars.find('_')+1:]
                else:
                    numb='%.3d'%0
                dfF.loc[i,'scan_numb'] = numb

                dfF.loc[i,'MB'] = _os.path.getsize(p)/(1024*1024)

        dfF=dfF.sort_values('scan_numb').reset_index(drop=True)

        # setup initial run_names. Will have a chance to manually correct.
        if 'bold' in scan_type:
            dfF['run_name'] = ['r%.3d'%r for r in dfF.index+1]
        elif 'TOPUP' in scan_type:
            idxFrwd = np.array(['_%s_'%blipForward in n for n in dfF['og']])
            dfF.loc[idxFrwd,'run_name'] = 'forward'
            dfF.loc[idxFrwd==False,'run_name'] = 'reverse' #blipForward[::-1]
        elif 'fieldmap' in scan_type:
            dfF['run_name'] = dfF['scan_numb']
        else: # 'TOPUP' in scan_type:
            dfF['run_name'] = dfF.index+1


        if ('bold' in scan_type) or ('fieldmap' in scan_type):
            cs = ['shrtOg','scan_numb','run_name','MB']
        else:
            cs = ['og','scan_numb','run_name','MB']
        if len(dfF)>0:
            print(dfF[cs])
            ok = input('look ok? (y/n): ')
        if not 'y' in ok:
            drop = input('Do you want to skip any of these scans? (y/n): ')
            if 'y' in drop:
                flag=True
                dfF_ = dfF.copy()
                while flag:
                    idx_ = eval(input('enter indices of bad scans as list (e.g.: "[0,1,3]"): '))
                    idx = [v for v in dfF.index if not v in idx_]
                    dfF_ = dfF_.loc[idx,:].reset_index(drop=True)
                    print(dfF_[cs])
                    ok = input('Are the correct scans remaining? (y/n): ')
                    if 'y' in ok:
                        flag=False
                        dfF = dfF_
                        print(dfF[cs])
            reorder = input('Do you want to change the run_name? (y/n): ')
            if 'y' in reorder:
                flag=True
                dfF_ = dfF.copy()
                while flag:
                    idx_ = eval(input('''enter new run_names as list (e.g. "['r001','r002']"): '''))
                    dfF_['run_name'] = idx_
                    print(dfF_[cs])
                    ok = input('Is run_name correct? (y/n): ')
                    if 'y' in ok:
                        flag=False
                        dfF = dfF_
                        print('-'*30)
                        print('Great. The final scan info:')
                        print(dfF[cs])
                        ok = input('Ready to rename files? (y/n): ')
                        if not 'y' in ok:
                            raise ValueError('oops.. (try renaming files by hand?)')
            else:
                print('\nfinal scan list:')
                print(dfF[['shrtOg','scan_numb','run_name','MB']])
                ok = input('Ready to rename files? (y/n): ')
                if not 'y' in ok:
                    raise ValueError('oops.. (try renaming files by hand?)')


        #% copy files
        if len(dfF)>0:
            subjRow = int(_np.nanmax([-1,_np.max(dfSubj.index)]))
            sessname = _os.path.split(topF)[1]
            date = sessname[sessname.rfind('_')+1:]
            if not _os.path.exists(newF):
                _os.makedirs(newF)
            shortNewFN = newF[newF.find('/fmri/')+1:]
            for i in dfF.index:
                oldP = dfF.loc[i,'p']
                oldJp = dfF.loc[i,'jp']
                newJn = '%s_%s_%s_%s.json'%(subj,date,scan_type,dfF.loc[i,'run_name'])
                newN = '%s_%s_%s_%s.nii.gz'%(subj,date,scan_type,dfF.loc[i,'run_name'])
                dfF.loc[i,'newName'] = newN
                newP = '%s/%s'%(newF,newN)
                shortNewP = '%s/%s'%(shortNewFN,newN)
                newJp = '%s/%s'%(newF,newJn)
                shutil.copy(oldP,newP)
                shutil.copy(oldJp,newJp)
                dfF.to_csv(newF+'/%s_renamed.csv'%scan_type)

                # write info to dfSubj:
                # - ['subj', 'date', 'coil', 'scanType', 'task', 'sessLabel', 'runLabel',
                #    'collectNote', 'procNote'],
                subjRow+=1
                dfSubj.loc[subjRow,'subj'] = subj
                dfSubj.loc[subjRow,'date'] = str(date)
                dfSubj.loc[subjRow,'scanType'] = scan_type
                dfSubj.loc[subjRow,'sessLabel'] = sessname
                dfSubj.loc[subjRow,'runLabel'] = dfF.loc[i,'run_name']
                dfSubj.loc[subjRow,'p'] = shortNewP

        #% erase old folders:
        if rmOld:
            for f in dfF['F']:
                shutil.rmtree(f)
    return dfSubj





def makeAffinesMatch(imgs,prefix='',interpolation='nearest',disp=True):
    '''resamples and reshapes each img in imgs to affine and shape of first img.


    output:
    -------------
    - imgs, prefix='r', to the original img directory.
    - prefix: default='r', if '', replaces original
    '''
    im1=_nib.load(imgs[0])
    cnt=0
    for im in imgs[1:]:
        m = _np.unique(_nib.load(im).get_fdata())
        isMask = _np.unique(m).shape[0]<=2
        if _np.any(_nib.load(im).affine!=im1.affine):
            cnt+=1
            if disp:
                print(im)
            if isMask:
                imOut=_nli.resample_to_img(im,im1,interpolation='nearest')
                if disp:
                    print('interpolation using "nearest"')
            else:
                imOut=_nli.resample_to_img(im,im1)
            p,n = _os.path.split(_os.path.realpath(im))
            imOut.to_filename(_os.path.join(p,prefix+n))


def image_orientation(p):
    im = _nib.load(p)
    return ''.join(_nib.aff2axcodes(im.affine))


def antsRegistrationSynQuick(fixedP, movingP, transforms='r',
                             outF=None,n_jobs=5,outPref='ants_'):
    '''setup with rigid+ affine transform as default (a). Use this
        for coreg within same subject, s (rigid + affine + deformable
       syn)  isthe default'''
    from nipype.interfaces.ants import RegistrationSynQuick

    f,n = _os.path.split(movingP)
    n = n[:n.find('.')]
    if outF == None:
        outF = f

    _os.chdir(outF)
    reg = RegistrationSynQuick()
    reg.inputs.fixed_image = fixedP
    reg.inputs.spline_distance=10 # reset from 26
#    reg.inputs.dimenson=3 # reset from 26
    reg.inputs.moving_image = movingP
    reg.inputs.num_threads = n_jobs
    reg.inputs.transform_type = transforms
    reg.inputs.output_prefix = outPref+'%s'%n
    print(reg.cmdline)
    reg.run()
    outP = _glob.glob('%s/%sWarped.nii*'%(outF,outPref+'%s'%n))[0]
    return outP






def recon_all(p, outF, parallel=True):
    '''anatomical to surface'''
    subDir = _os.path.split(outF)[0]
    n = _os.path.basename(outF)
    _os.environ["SUBJECTS_DIR"] = subDir
    cmd = 'echo $SUBJECTS_DIR'
    call(cmd,shell=True)

    if not len(_os.listdir(outF)) > 0:
        if parallel:
            cmd = 'recon-all -i %s -s %s -all -parallel' % (p, n)
        else:
            cmd = 'recon-all -i %s -s %s -all' % (p, n)
    else:
        if parallel:
            cmd = 'recon-all  -s %s -all -no-isrunning -parallel ' % (n)
        else:
            cmd = 'recon-all  -s %s -all -no-isrunning' % (n)
    call(cmd,shell=True)


def plot_carpet(funcP,outP,segP=None,segLabels=None):
    if segP is None:
        _nip.plot_carpet(funcP,
                        output_file=outP)
#        if segLabels==None:
#            segLabels = {'wm':4,'csf':1,'cortical grey':2}
    else:
        _nip.plot_carpet(funcP,
                         segP,
                         mask_labels=segLabels,
                         output_file=outP)


def tsnr(inP,outF,outN,polyDeg=0):
    from nipype.algorithms.confounds import TSNR
    tsnr = TSNR()
    tsnr.inputs.in_file = inP# dCnd[cnd][i]
    tsnr.inputs.mean_file = outF+'/%s_m.nii.gz'%outN
    tsnr.inputs.stddev_file = outF+'/%s_sd.nii.gz'%outN
    tsnr.inputs.detrended_file = outF+'/%s_detrend.nii.gz'%outN
    tsnr.inputs.tsnr_file = outF+'/%s_tsnr.nii.gz'%outN
    if polyDeg!=0:
        tsnr.inputs.regress_poly = polyDeg
    res = tsnr.run()

    return res,tsnr.inputs.tsnr_file

def dcm2niix(inF,outF):
    '''-f : filename (%a=antenna (coil) name, %b=basename, %c=comments,
    %d=description, %e=echo number, %f=folder name, %i=ID of patient,
    %j=seriesInstanceUID, %k=studyInstanceUID, %m=manufacturer,
    %n=name of patient, %o=mediaObjectInstanceUID, %p=protocol,
    %r=instance number, %s=series number, %t=time, %u=acquisition number,
    %v=vendor, %x=study ID; %z=sequence name; default '%f_%p_%t_%s')
    '''
    cmd = "dcm2niix -f '#p_#t_#s' -o %s -z y %s"%(outF,inF) # #s=series number
    cmd = "dcm2niix -f '#s_#p' -o %s -z y %s"%(outF,inF) # #s=series number
    cmd=cmd.replace('#','%')
#    print(cmd)
    call(cmd,shell=True)


#def biasfield_fix(inP,outP=None):
#    if outP == None:
#        f,n = _os.path.split(inP)
#        if len(f) ==0:
#            outP = '%s_%s'%('ro_',n)
#    cmd='mri_nu_correct.mni --i %s --o %s'%(inP,outP)
#    print(cmd)
#    call(cmd,shell=True)
    # -o /users/cr/outdir/ -z y ~/dicomdircall(cmd,shell=True)


def tkmedit(p):
    cmd='tkmedit -f %s'%p
    call(cmd,shell=True)



# def swap_axes(inP):
#     f,n = _os.path.split(inP)
#     outP = f+'/swapAxes_%s'%n
#     nii = _nib.load(inP)
#     m,aff =  nii.get_fdata().astype(_np.float64), nii.affine
#     m2 = _np.swapaxes(m, 0, 2)
#     im2 = _nib.Nifti1Image(m2,aff)
#     im2.to_filename(outP)
#     tkmedit(outP)
#     return outP


#def check_reg_tkmedit(p):
#    print('-'*50)
#    print('''A correctly specified volume loaded in tkmedit will produce the
#    following views:\n''' )
#    print('''coronal: (c(0) is posterior, c(255) is anterior)
#        X: LEFT right RIGHT left (so the subjects right hemisphere is on the left)
#        Y: TOP superior, BOTTOM inferior''')
#    print('''horizontal: (h(0) is superior, h(255) is inferior)
#        X: LEFT right RIGHT left (so the subjects right hemisphere is on the left)
#        Y: TOP anterior, BOTTOM posterior''')
#    print('''sagital: (s(0) is right, s(255) is left)
#        X: LEFT posterior, RIGHT anterior
#        Y: TOP superior, BOTTOM inferior
#        ''')
#    cmd = 'tkmedit %s'%p
#    call(cmd,shell='zsh)')
#    tkmedit(p)

#def freeview_surfs(volP,surfs,subjdir):


def freeview(volPs,surfPs=None):
    cmd = 'freeview '
    for p in volPs:
        cmd+=' -v %s'%p
    if surfPs!=None:
        cmd+=' -f'
        for p in surfPs:
            cmd+=' %s'%p
    print(cmd)
    call(cmd,shell=True)


def sphinx_fix(inP,toStd=False,outP=None,pos_string='LPS',visualize=False):
    '''from https://surfer.nmr.mgh.harvard.edu/fswiki/MonkeyData

    this uses mri_convert, and thgewn fslorient2std to correct sphinx
        '''
    if outP == None:
        f,n = _os.path.split(inP)
        outP = f+'/%s_%s'%('sphnx',n)
    cmd='mri_convert -i %s -o %s --in_orientation %s --sphinx'%(inP,outP,pos_string)
#    print(cmd)
    call(cmd,shell=True)

    if toStd: # if we run this, fsleyes has labelled correctly, but upside down
        f,n = _os.path.split(outP)
        cmd = 'fslreorient2std %s/%s %s/%s'%(f,n,f,n)
#        print(cmd)
        call(cmd,shell=True)
    if visualize:
#        cmd = 'mri_info %s'%outP
        print(image_orientation(outP))
        call(cmd,shell=True)
        freeview([outP])
    return outP


def check_registration(baseP,layerP):
    fig = _nip.plot_anat(baseP)
    fig.add_edges(layerP)


def image_header(im):
    for k in im.header.keys():
        print(k,im.header[k])


def runBrex(f,inFileName):
    '''all input files must be in the same directory'''
    print('-'*50)
    print('''run the following commands in the terminal. 3 fsleyes windows
    will open. If one looks good, select 1,2 or 3. Otherwise, select 4
    (and enter your own frac value (0.:1.):\n''')
    _os.chdir(f)
    b='ss_macaque_25_model-MNI.nii.gz'
    nb='macaque_25_model-MNI.nii.gz'
    h = inFileName
    print('cd %s\n'%f)
    cmd = 'bash atlasBREX.sh -b %s -nb %s -h %s -f n -nrm 1 -reg 2'%(b,nb,h)
    print(cmd)
    print('-'*50)


def createMaskFromBrex(normF,outN='T2_mask.nii.gz'):
    '''atlasBREX outputs only an ss image, so we create a mask'''
    p = _glob.glob(normF+'/*brain.nii.gz')
    dat = fmri_data([p],'background')
    vMaskIm = dat.masker(dat.mask['niimg']).astype(_np.float32)
    maskIm = dat.unmasker(vMaskIm)
    _nip.plot_anat(maskIm)
    maskIm.to_filename(normF+'/%s'%outN)
    return maskIm


def mcflirt(inP,outP):
    from nipype.interfaces import fsl
    mcflt = fsl.MCFLIRT(in_file=inP, cost='mutualinfo')
    mcflt.inputs.save_plots=True
    mcflt.inputs.out_file = outP
    print(mcflt.cmdline)
    res = mcflt.run()


def fs_mri_convert(dcmF,outF):
    from nipype.interfaces.freesurfer import DICOMConvert
    cvt = DICOMConvert()
    cvt.inputs.dicom_dir = dcmF
    cvt.inputs.base_output_dir = outF
    cvt.inputs.args = '--in_type siemens --sphinx --out_type nii'
    cvt.inputs.file_mapping = [('nifti', '*.nii'), ('info', 'dicom*.txt'), ('dti', '*dti.bv*')]
    print(cvt.cmdline)
    cvt.run()


def antsMotionCorr(inP,outF,outPrefix):
    ''' affine and deformable transformation
    first calc mean ts, then use this as fixed image for each vol in 4d ts'''
    # from nipype.interfaces.ants.preprocess import MotionCorr
    # ants_avg = MotionCorr()
    # ants_avg.inputs.average_image = inP
    # ants_avg.inputs.output_average_image = outP
    # ants_avg.cmdline
    # ants_avg.run()
    # f = _os.path.basename(inP)

    cmd = 'antsMotionCorr  -d 3 -a %s -o %s '%(inP,outF+'/%s_avg.nii.gz'%(outPrefix))# averate ts'
    print(cmd)
    call(cmd,shell=True)

    # do affine and deformable correction
    _os.chdir(outF)
    cmd = 'antsMotionCorr  -d 3 -o [${out},${out}.nii.gz,${out}_avg.nii.gz]'
    cmd+= ' -m gc[ ${out}_avg.nii.gz , $in , 1 , 1 , Random, 0.05  ]'
    cmd+= ' -t Affine[ 0.005 ] -i 20 -u 1 -e 1 -s 0 -f 1 '
    cmd+= ' -m CC[  ${out}_avg.nii.gz , $in , 1 , 2 ] '
    cmd+= ' -t SyN[0.15,3,0.5] -i 20 -u 1 -e 1 -s 0 -f 1 -n 10'
    cmd = cmd.replace('${out}',outPrefix).replace('$in',inP)
    print(cmd)
    call(cmd,shell=True)



def plot_fsl_motion_parameters(parameter_file, outP):
    """ Plot motion parameters obtained with FSL software

    Parameters
    ----------
    parameter_file: string
        path of file containing the motion parameters.
    outfname: string
        output filename for storing the plotted figure.

    from pypreclin
    """
    # Load parameters
    motion = _np.loadtxt(parameter_file)
    motion[:, :3] *= (180. / _np.pi)

    # do plotting
    _plt.figure()
    fig, (ax1, ax2) = _plt.subplots(2, 1)
    ax1.plot(motion[:, 3:])
    ax1.set_xlabel("time(scans)")
    ax1.set_ylabel("Estimated motion (mm)")
    ax1.grid(True)
    ax2.plot(motion[:, :3])
    ax2.set_xlabel("time(scans)")
    ax2.set_ylabel("Estimated motion (degrees)")
    ax2.grid(True)
    _plt.legend(("TransX", "TransY", "TransZ", "RotX", "RotY", "RotZ"),
               loc="upper left", ncol=2)
    _plt.savefig(outP)
    _plt.close()


def bet(inP,frac=.5,outP=None):
        # cmd = 'bet %s -f %.f'%(p,f)
        # call(cmd,shell=True)
    import nipype.interfaces.fsl as fsl
    mybet = fsl.BET()
    mybet.inputs.in_file = inP
    if outP is not None:
        mybet.inputs.out_file = outP
    mybet.inputs.frac = frac
    print(mybet.cmdline)
    result = mybet.run()


#def fslSliceTimer(inP,TR,outP,interleaved=False,ascend=True,outType='NIFTI_GZ',sliceDrct=3):
#    from nipype.interfaces import fsl
#    st = fsl.SliceTimer()
#    st.inputs.in_file = inP
#    st.inputs.time_repetition = _np.float32(TR)
#    st.inputs.interleaved = interleaved
#    st.inputs.index_dir = ascend
#    st.inputs.out_file = outP
#    st.inputs.output_type = outType
#    st.inputs.slice_direction = sliceDrct
#    print(st.cmdline)
#    st.run()


def antsApplyTransforms(inP,refP,outP,lTrns,
                        interp='Linear',dim=3,invertTrans=False,inptype=0):
    ''' might want to try interp='BSpline'
    lTrans is a list of paths to transform files (.e.g, .h5)
    I think invert trans will just work...

    refP indicates the dimensions desired...
    '''
    from nipype.interfaces.ants import ApplyTransforms
    at = ApplyTransforms()
    at.inputs.dimension = dim
    at.inputs.input_image = inP
    at.inputs.reference_image = refP
    at.inputs.output_image = outP
    at.inputs.interpolation = interp
    at.inputs.transforms = lTrns
    at.inputs.input_image_type = inptype
    at.inputs.invert_transform_flags = [invertTrans for i in lTrns]
#    at.inputs.verbose = 1
    print(at.cmdline)
    at.run()


#def antsBrainExtraction(anatP,brainExtractTemplate,probMaskP,outputPrefix='ss', imageDimension=3):
#    cmd = '''antsBrainExtraction.sh -d imageDimension \
#              -a anatomicalImage \
#              -e brainExtractionTemplate \
#              -m brainExtractionProbabilityMask \
#              -o outputPrefix'''
#
#    cmd=cmd.replace("imageDimension",str(imageDimension)).replace('anatomicalImage',anatP)
#    cmd=cmd.replace('brainExtractionTemplate',brainExtractTemplate)
#    cmd = cmd.replace('brainExtractionProbabilityMask',probMaskP)
#    cmd=cmd.replace('outputPrefix',outputPrefix)
#    print(cmd)
#    call(cmd,shell=True)


def ants_brainExtraction(t1P,tmplP,tmplProbMask):
    f = _os.path.split(t1P)[0]
    _os.chdir(f)
    from nipype.interfaces.ants.segmentation import BrainExtraction
    brainextraction = BrainExtraction()
    brainextraction.inputs.dimension = 3
    brainextraction.inputs.anatomical_image =t1P
    brainextraction.inputs.brain_template = tmplP
    brainextraction.inputs.brain_probability_mask = tmplProbMask
    brainextraction.cmdline
    brainextraction.run()




def antsAtroposN4(t1P,maskP,outPrefix):
    cmd='''
      antsAtroposN4.sh \
      -d 3 \
      -a %s \
      -x %s \
      -c 3 \
      -y 2 \
      -y 3 \
      -w 0.25 \
      -o %s'''%(t1P,maskP,outPrefix)
    print(cmd+'\n')
    call(cmd,shell=True)


def antsCommonTemplate(t1s_fldr,n_jobs):
    '''cds to directory containing all anats and runs.'''

    _os.chdir(t1s_fldr)
    cmd = ["""inputPath=${PWD}/
           outputPath=${PWD}/TemplateSyN/

        ${ANTSPATH}/antsMultivariateTemplateConstruction.sh \
          -d 3 \
          -o ${outputPath}T_ \
          -i 4 \
          -g 0.2 \
          -j %d \
          -c 2 \
          -k 1 \
          -w 1 \
          -m 100x70x50x10 \
          -n 1 \
          -r 1 \
          -s CC \
          -t GR \
        ${inputPath}/*.nii"""%n_jobs
        ]
    call(cmd,shell=True)


def get_orient(p):
    im = _nib.load(p)
    return _nib.aff2axcodes(im.affine)




class fmri_data:
    '''General class for handling fmri data.

    Parameters:
    --------------
    - imgs: All images must have the same affine and shape. Accepts several formats:
     - list of image paths
     - list of 3d nibabel ('niimg') images.
     - a single 4d image
     - pandas series
    - mask: automatically resampled to match imgs. Accepts several formats:
     - niimg
     - path to mask: (str)
     - 'full': full brain (mni) mask
     - 'grey': grey matter (mni) mask
     - 'background': calculate background from imgs
     - 'epi'
    - [fwhm=scalar] (optional): smooth imgs before masking and transforming to 2d


    Returns
    --------------
    fmri_data object ('dat')
     - dat.dat: 2d matrix (volumes*voxels)
     - dat.volInfo: img parameters
     - dat.mask:  mask parameters

    Methods
    -----------------
    - cleaner
    - masker
    - unmasker
    - clusterPuller
    - antsApplyWarp (untested)

    Examples:
    ------------
    >>> # load and mask volumes:
    >>> imPs  = np.sort(glob.glob('*.nii'))
    >>> dat   =  fmri_data(imPs,'background')
    >>> # create 4d niimg:
    >>> niimg = dat.unmasker(dat.dat)
    >>> # standardize by run:
    >>> dat.sessions = runs # 1d array with unique label for each run
    >>> dat.cleaner(standardizeRun=True)


    '''

    def __init__(self, imgs=None,mask=None,fwhm=None):
        self.history    = []
        self.fwhm       = fwhm
        self.y          = None     # 1d numpy array
        self.clusterNib = None  # Nifti1Image object
        self.processMask= None  # path for extracting signals (e.g., defines volume for searchlight analysis)
        self.sessions   = None  # np array, same shape as y
        self.volInfo    = { 'imgPaths'  : imgs,
                        'dim'       : None,
                        'affine'    : None,
                        }
        self.mask       = { 'imgPaths'  : mask,
                        'fullPath'  : None,
                        'history'   : [],
                        'volInfo'   : { 'dim': None,'affine': None,},
                        }
        # input imgs :
        if imgs is None:
            self.volInfo['imgPaths'] = _guiLoad(title='Select functional imgs')
        elif not isinstance(imgs,list):
            imgs=[imgs]
        if isinstance(imgs[0],_nib.nifti1.Nifti1Image):
            imIsNiimg = 1
        else: #isinstance(imgs[0],str):
            imIsNiimg = 0
            self.volInfo['imgPaths'] = [_os.path.realpath(p) for p in self. volInfo['imgPaths']]

        if imIsNiimg==0:
            imgs = _nib.concat_images(self.volInfo['imgPaths'])
        else:
            if len(imgs[0].shape)>3:
                imgs = _nib.concat_images(imgs,axis=3)
            else:
                imgs = _nib.concat_images(imgs)
        imgs     = _nib.squeeze_image(imgs)

        # smooth, setup volInfo:
        if self.fwhm is not None:
            imgs = _nli.smooth_img(imgs,self.fwhm)
            print('smoothed with %dmm fwhm kernel'%self.fwhm)
            self.history.append('smoothed with %dmm fwhm isotropic kernel'%self.fwhm)
        self.volInfo['affine']  = imgs.affine
        self.volInfo['dim']     = imgs.shape

        # input mask :
        if isinstance(mask,_nib.nifti1.Nifti1Image):
            self.mask['niimg'] = mask
            self.mask['imgPaths'] = None
        elif 'grey' in mask:
            self.mask['imgPaths'] = greyMaskPath
            self.mask['niimg'] = _nib.load(self.mask['imgPaths'])
        elif 'full' in mask:
            self.mask['imgPaths'] = fullBrainMaskPath
            self.mask['niimg'] = _nib.load(self.mask['imgPaths'])
        elif 'background' in mask:
            self.mask['niimg'] = _compute_background_mask(self.volInfo['imgPaths'],
                                border_size=2, connected=False, opening=False)
            self.mask['imgPaths'] = 'background'
        elif 'epi' in mask:
            self.mask['niimg']=_compute_epi_mask(self.volInfo['imgPaths'])
            self.mask['imgPaths']= 'epi'
        elif mask is None:
            print('use gui to select mask')
            self.mask['imgPaths'] = _guiLoad(title='Select mask')
        else:
            self.mask['imgPaths'] = mask
            self.mask['niimg'] = _nib.load(self.mask['imgPaths'])



        if _np.any(imgs.affine!=self.mask['niimg'].affine):
            self.mask['niimg'] = _nli.resample_img(self.mask['niimg'],
                        target_affine=self.volInfo['affine'],
                        target_shape=self.volInfo['dim'][:3],
                        interpolation='nearest')
            self.history.append('mask resampled to func(s)')

        self.mask['volInfo'] = {'affine':   self.mask['niimg'].affine,
                                'dim':      self.mask['niimg'].shape,
                                }

        # load, mask, convert to 2d:
        def masker(imgs,interpolation='nearest',selfmask=self.mask['niimg']):
            return _apply_mask(imgs,selfmask)
        self.masker=masker

        def unmasker(dat,selfmask=self.mask['niimg']):
            return _unmask(dat,selfmask)
        self.unmasker=unmasker

        self.dat = self.masker(imgs,self.mask['niimg']).squeeze()
        self.history.append('imgs masked by %s, & converted to 2d'%self.mask['imgPaths'])


    def clusterPuller(self,slowAndMni=0):
        '''General method for pulling information from an roi.
        Automatically resamples clusterIm (or mask) to space of dat object
         - Returns:
            - roiSummary: a dictionary with key = AAL region name. Fields:
            - subRoiMeans: mean (across voxels) of the roi, 1 value per image
            - maskNib: binary Nifti1Image roi mask
            - clusterMapNumb
            - roiDat: data from all roi voxels'''
        if self.clusterNib=='None':
            raise Exception( 'dat.clusterNib not defined')
        return pullRoi(self,self.clusterNib,slowAndMni=slowAndMni)


    def antsApplyWarp(self,t1FList,funcPath,toMni=1):
        '''using previously estimated transformation files stored in
        the [t1FList], apply transform to imgs stored in
        self.volInfo['imgPaths']

        input
        --------
        - t1FList: list of directories where the transformation files are stored. should match order of self.volInfo['impPaths']
        - funcPath: the image to be warped.
        - toMni: when toMni=1, warps to mni. when toMni=0, warps from template to native

        output
        --------
        - 3d images are writtin to the dat.outFldr, each new image is prepended with 'w'
        - new dat object with rewarped images loaded and masked with the self.mask strategy.
        '''
        if self.outFldr is None:
            raise Exception('dat.outFldr not specified')

        outImList=[]
        for i,im in enumerate(self.volInfo['imgPaths']):
            t1F = t1FList[i]
            p,n = _os.path.split(im)
            outPath = _os.path.join(self.outFldr,'w'+n)
            outImList.apend(outPath)
            antsWarpFunc(t1F,im,toMni=toMni,outPath=outPath)

        datOut=fmri_data(outImList,self.mask['imgPaths'])
        return datOut


    def cleaner(self,detrend=False,standardizeRun=False,standardizeVox=False,confounds=None,
                        low_pass=None,high_pass=None,plot=0):
        '''If dat.sessions is provided, each run is cleaned indpendently.

        If dat.confounds is not empty, these signals are accounted for.

        default options:
        ------------
        detrend=False,standardizeRun=False,standardizeVox=False,confounds=None,
                        low_pass=None,high_pass=None,plot=0'''
        from nilearn.signal import clean
        assert(not(_np.logical_and(standardizeRun,standardizeVox)))

        if len(_np.unique(self.sessions))<=1:
            print('dat.sessions is empty or contains 1 unique value. Will clean dat.dat by entire columns ''')
            self.history.append(("dat.dat cleaned: detrend=%s,standardizeRun=%s"
                            "standardizeVox=%s, confounds=%s,"
                            "low_pass=%s, high_pass=%s"%(detrend,standardizeRun,
                            standardizeVox,_np.all(confounds is not None),low_pass,high_pass)))
        else:
            print('dat.dat cleaned by run')
            self.history.append(("dat.dat cleaned by run: detrend=%s,"
                            "standardizeRun=%s, standardizeVox=%s, confounds=%s,"
                            "low_pass=%s, high_pass=%s"%(detrend,standardizeRun,
                            standardizeVox,_np.all(confounds is not None),low_pass,high_pass)))
        if plot:
            _plt.figure()
            _plt.subplot(211)
            _plt.plot(self.dat[:,_np.arange(0,self.dat.shape[1],self.dat.shape[1]/4)])

        self.dat = clean(self.dat,sessions=self.sessions, detrend=detrend, \
                        standardize=standardizeVox, confounds=confounds, \
                        low_pass=low_pass, high_pass=high_pass)
        if standardizeRun:
            for run in _np.unique(self.sessions):
                idxRun = self.sessions==run
                X = self.dat[idxRun==1,:]
                X = zAllAxes(X)
                self.dat[idxRun==1,:] = X
        if plot:
            _plt.subplot(212)
            _plt.plot(self.dat[:,_np.arange(0,self.dat.shape[1],self.dat.shape[1]/4)])
            _plt.pause(.01)
