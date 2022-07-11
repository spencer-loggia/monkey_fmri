###
# Macaque anatomical processing pipeline.
# Authored by Kurt Braunlich 2021
# Added option to use macaque template to define white matter, put into
# Python notebook format, added additional comments--Stuart Duffield 2022
# DEPENDENCIES
# Advanced Normalization Tools: http://stnava.github.io/ANTs/
# Freesurfer: https://surfer.nmr.mgh.harvard.edu
# Precon_all: https://github.com/neurabenn/precon_all
# PREEMACS: https://github.com/pGarciaS/PREEMACS
# Need to set up paths past functions
###
#%% Imports
import sys
import nibabel as nib
import os
from subprocess import call
import glob
import numpy as np
import shutil
import nibabel as nib
import gzip
import nilearn.image as nli

#%% Set up the class fmri_data that many functions are dependent on
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
        if isinstance(imgs[0],nib.nifti1.Nifti1Image):
            imIsNiimg = 1
        else: #isinstance(imgs[0],str):
            imIsNiimg = 0
            self.volInfo['imgPaths'] = [os.path.realpath(p) for p in self. volInfo['imgPaths']]

        if imIsNiimg==0:
            imgs = nib.concat_images(self.volInfo['imgPaths'])
        else:
            if len(imgs[0].shape)>3:
                imgs = nib.concat_images(imgs,axis=3)
            else:
                imgs = nib.concat_images(imgs)
        imgs     = nib.squeeze_image(imgs)

        # smooth, setup volInfo:
        if self.fwhm is not None:
            imgs = nli.smooth_img(imgs,self.fwhm)
            print('smoothed with %dmm fwhm kernel'%self.fwhm)
            self.history.append('smoothed with %dmm fwhm isotropic kernel'%self.fwhm)
        self.volInfo['affine']  = imgs.affine
        self.volInfo['dim']     = imgs.shape

        # input mask :
        if isinstance(mask,nib.nifti1.Nifti1Image):
            self.mask['niimg'] = mask
            self.mask['imgPaths'] = None
        elif 'grey' in mask:
            self.mask['imgPaths'] = greyMaskPath
            self.mask['niimg'] = nib.load(self.mask['imgPaths'])
        elif 'full' in mask:
            self.mask['imgPaths'] = fullBrainMaskPath
            self.mask['niimg'] = nib.load(self.mask['imgPaths'])
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
            self.mask['niimg'] = nib.load(self.mask['imgPaths'])



        if _np.any(imgs.affine!=self.mask['niimg'].affine):
            self.mask['niimg'] = nli.resample_img(self.mask['niimg'],
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
            p,n = os.path.split(im)
            outPath = os.path.join(self.outFldr,'w'+n)
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
#%% functions
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
def mri_nu_correct(p):
    f,n= os.path.split(p)
    outP = os.path.join(f,'ro_%s'%n)
    cmd= 'mri_nu_correct.mni --i %s --o %s --distance 24'%(p,outP)
    call(cmd,shell=True)
    return outP
def antsRegistrationSynQuick(fixedP, movingP, transforms='r', outF=None,n_jobs=5,outPref='ants_'):
    '''setup with rigid+ affine transform as default (a). Use this
        for coreg within same subject, s (rigid + affine + deformable
       syn)  isthe default'''
    from nipype.interfaces.ants import RegistrationSynQuick

    f,n = os.path.split(movingP)
    n = n[:n.find('.')]
    if outF == None:
        outF = f

    os.chdir(outF)
    reg = RegistrationSynQuick()
    reg.inputs.fixed_image = fixedP
    reg.inputs.spline_distance=10
    reg.inputs.moving_image = movingP
    reg.inputs.num_threads = n_jobs
    reg.inputs.transform_type = transforms
    reg.inputs.output_prefix = outPref+'%s'%n
    print(reg.cmdline)
    reg.run()
    outP = glob.glob('%s/%sWarped.nii*'%(outF,outPref+'%s'%n))[0]
    return outP
def combine_wInModality(ps,targP=None,nuCorrect=True,outF=None,imType=None,transforms='r'):
    '''to improve reconall, we average t1s within an animal. This
    function peforms 1. bias field correction. 2. coregistration
    3. averaging'''
    if outF==None:
        outF = os.path.split(ps[1])[0]

    #  bias field correct
    if nuCorrect:
        coregInput=[]
        for p in ps:
            coregInput.append(mri_nu_correct(p))
    else:
        coregInput = ps


    if targP==None: # 2nd coreg to biggest vol
        imax= np.argmax([os.path.getsize(p) for p in coregInput])
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
def gzip_to_nii(gzP):
    ''' checks if unzipped version exists in dir. if not, creates it.'''
    niiP = gzP.replace('.gz','')
    if not os.path.exists(niiP):
        with gzip.open(gzP, 'rb') as f_in:
            with open(niiP, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return niiP
def dnn_skull_strip(img_path, output_dir, p=0.5):
    '''use DNN to extract brain from n4 bias corrected brain.'''
    sys.path.append(monkeyDeepBrainF)
    from deepbrain import Extractor
    img = nib.load(img_path)

    affine = img.affine
    img = img.get_fdata()

    extractor = Extractor()

    now = time.time()
    prob = extractor.run(img)
    print("Extraction time: {0:.2f} secs.".format(time.time() - now))
    mask = prob > p
    brain_mask = (1 * mask).astype(np.uint8)
    brain_mask = nib.Nifti1Image(brain_mask, affine)
    maskP = os.path.join(output_dir, "brain_mask_p%2f.nii"%p)
    nib.save(brain_mask, maskP)

    brain = img[:]
    brain[~mask] = 0
    brain = nib.Nifti1Image(brain, affine)
    brainP = os.path.join(output_dir, "brain_p%2f.nii"%p)
    nib.save(brain, brainP)

    return brainP, maskP
def mask_an_image(in_file,mask_file,out_file,output_type='NIFTI_GZ'):
    '''input are string-like paths
    PREEMAC:
    fslmaths ${maskDir}/${subId}/brain_mask.nii.gz  -thr 0.0001 out_mask.nii.gz
    fslmaths bias_corrected.nii.gz -mul out_mask.nii.gz out_file.nii.gz

    fslmaths ${maskDir}/${subId}/brain_mask.nii.gz  -thr 0.0001 out_mask.nii.gz
    fslmaths bias_corrected.nii.gz -mul out_mask.nii.gz out_file.nii.gz'''
    im = nib.load(mask_file)
    m = im.get_fdata()
    if not len(np.unique(m))==2:
        m[m>.0001] = 1.
        m[m<=.0001]=0.
        im = nib.Nifti1Image(m,im.affine)
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
def surfing_safari(inP,steps='precon_all',ants=True):
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
    f,n=os.path.split(inP)
    os.chdir(f)
    if ants:
        cmd = 'cd %s; surfing_safari.sh -i %s -r %s -a NIMH_mac  -n -s'%(f,inP,steps)
    else:
        cmd = 'cd %s; surfing_safari.sh -i %s -r %s -a NIMH_mac  -n'%(f,inP,steps)
    print(cmd)
    call(cmd,shell=True)
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
    outF = os.path.split(outP)[0]
    os.chdir(outF)
    antsCoreg(fixedP,
                 movingP,
                 outP,
                 initialTrsnfrmP=None, # we are working on resampled img
                 across_modalities=True,
                 outPref='antsReg',
                 transforms=ltrns,
                 run=True,n_jobs=n_jobs)
    frwdTrnsP =  glob.glob(outF+'/antsRegComposite.h5')[0]
    return frwdTrnsP
def antsApplyTransforms(inP,refP,outP,lTrns,interp='Linear',dim=3,invertTrans=False,inptype=0):
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
    print(at.cmdline)
    at.run()

#%% Set up paths
os.environ["OMP_NUM_THREADS"] = "12" # This sets the number of threads needed
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "12"
anatF = '/Volumes/bc7/projects/mTurk_FMRI/anatproc/images' # Where are the T1s
subj = '' # what is the subject name

#%% Find necessary starting files
t1Ps = list(np.sort(glob.glob(os.path.join(anatF, '*MPR*.nii.gz')))) # Find T1s. You need T1s
t2Ps = list(np.sort(glob.glob(os.path.join(anatF, '*T2*.nii.gz')))) # Find T2s, if they exist
print('\nOriginal images found:')
for p in t1Ps+t2Ps:
    print('%s: %.2f MB'%(os.path.basename(p),os.path.getsize(p)*1e-6))

roPs = glob.glob(anatF+'/ro_*.nii.gz')
print(roPs)

#%%  Basic Quality Control:
# check that the individual images look ok (e.g., signal from both coils).
# The images have not yet been aligned, so don't worry about that.
# Look for basic image quality. If you have a very high quality T1,
# you probably don't want to average in noisy images. If you have several
# noisy images, more data will probably be better.
for i in range(len(t1Ps)):
    mp.freeview([t1Ps[i]])
for i in range(len(t2Ps)):
    mp.freeview([t2Ps[i]])

avgT1P = combine_wInModality(t1Ps,
                                nuCorrect=True,
                                outF=anatF,
                                imType='T1',
                                transforms='r')

avgT1_niiP = gzip_to_nii(avgT1P) # avT1.nii is created--as avgT1P

#%% If you have multiple T2s, repeat this step
avgT1P = combine_wInModality(t2Ps,
                                nuCorrect=True,
                                outF=anatF,
                                imType='T2',
                                transforms='r')
avgT2_niiP = gzip_to_nii(avgT2P) # avT1.nii is created--as avgT1P

#%% Restart
# run this if restarting
avgT1P = os.path.join(anatF,'avT1.nii.gz')
avgT1_niiP = os.path.join(anatF,'avT1.nii')
avgT2P = os.path.join(anatF,'avT2.nii.gz')
avgT2_niiP = os.path.join(anatF,'avT2.nii')

#%% Register to T2
# Register to the T2 if taken. Hopefully will correct spatial distortions
antsRegistrationSynQuick(avgT1_niiP,avgT2_niiP,transforms='r')
freeview([avgT1_niiP, avgT2_niiP])

#%% Skullstrip
# Edit the p parameter to your liking for skull stripping
dnnBrainP,dnnMaskP = mp.dnn_skull_strip(avgT1_niiP, anatF, p=0.8)
# After skullstripping, you will want to manually inspect and edit the brainmask.
# You need to remove all of the mask outside of the brain, which will take about
# 30 minutes or so depending on the resolution of the image.
# You also want to make sure that as much of the dura matter and pia is stripped
# away. This can be somewhat difficult, and I would err on the side of preserving
# cortex than getting rid of pia. Save this new image as whatever you want to
# set maskP as (usually anatF/brainmask.nii)

#%% Edit
freeview([avgT1_niiP,dnnMaskP,dnnBrainP])

#%% What did you save the edited mask as?
maskP = anatF+'/brainmask.nii'

#%% Mask the T1
f,n = os.path.split(maskP)
maskedP = os.path.join(f,'%s.nii'%subj)
mask_an_image(avgT1_niiP,maskP,maskedP)
#%% Check the masked image, if you don't like it go back and edit the mask again
# remask
freeview([maskedP])

#%% Create surfaces
# This will take a while.
surfing_safari(maskedP, steps='precon_all')
subjF = anatF+'/%s'%subj
mriF = subjF+'/mri'
surfF = subjF+'/surf'
lInflatedP = surfF+'/lh.inflated'
rInflatedP = surfF+'/rh.inflated'
wmP = mriF+'/wm_orig.nii.gz'

# Look at the white matter generation. Is it satisfactory?
freeview([maskedP,wmP],[lInflatedP,rInflatedP])

#%% White matter edits.
# There are two approaches you can take to conduct your white matter edits. If the
# surface generation produced a white matter volume that is close to what you
# would like the white matter to look like, then it may be most prudent to
# edit this white matter volume directly and then proceed to the PRECON-3
# stage of the script. If it is not, then


# Edit white matter by registering wm to the brain
templateP ='/Volumes/bc7/projects/mTurk_FMRI/MTurk1/templates/NMT_v2.0_asym/NMT_v2.0_asym_05mm/NMT_v2.0_asym_05mm_SS.nii.gz'
segP = '/Volumes/bc7/projects/mTurk_FMRI/MTurk1/templates/NMT_v2.0_asym/NMT_v2.0_asym_05mm/NMT_v2.0_asym_05mm_segmentation.nii.gz'
outF = '/Volumes/bc7/projects/mTurk_FMRI/anatproc'
itkTxt, itkMan=mp.itkSnapManual(maskedP,templateP,outF)


NMTinSubj = anatF+'/NMTinSubj.nii.gz'
fwtrnsP = mp.antsCoReg(maskedP,itkMan,NMTinSubj,itkMan,n_jobs=10)
# What we're doing to do is take the original segmentation and parse it so that everything that isn't WM == 0 and WM = 1
segmentation = nib.load(segP)
seg_data = segmentation.get_fdata()
WMseg = np.array(np.where(seg_data == 4, 1,0),dtype=float)
WMseg_nii = nib.Nifti1Image(WMseg,affine=segmentation.affine)
WMseg_path = anatF+'/WMsegNMT.nii'
nib.save(WMseg_nii,WMseg_path)
mp.antsApplyTransforms(WMseg_path,maskedP,segInSubj,[fwtrnsP,itkTxt],interp = 'NearestNeighbor') # interp = 'NearestNeighbor'

mp.freeview([maskedP,segInSubj])


wmP = glob.glob(os.path.join(anatF,'*','mri','wm_orig.nii.gz'))[0]
f,n = os.path.split(wmP)
wmHandP = os.path.join(f,'wm_hand_edit.nii.gz')
shutil.copyfile(segInSubj,wmHandP)


mp.ksurfing_safari(maskedP, steps='precon_3')
