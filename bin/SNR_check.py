import sys

if len(sys.argv) < 9:
    print('Please supply subject name, path to session directory, SNR estimated on scan day, session MION dosage, number of used runs, receive coils (t for turquoise, o for orange, n for nif), path to roi directory, and True if you want measurements in whole brain AND rois')
    
else:  
    import numpy as np
    import pandas as pd
    import nibabel as nib
    import os
    import glob

    subject = str(sys.argv[1]) # wooster or jeeves
    session_dir = str(sys.argv[2]) # path to session directory
    scan_snr_est = float(sys.argv[3]) # SNR estimated on scan day, should be in scan notes
    amt_mion = float(sys.argv[4]) # mL MION given at session start
    num_runs = int(sys.argv[5]) # number of runs being used in analysis
    r_coils = str(sys.argv[6])
    rois_dir = str(sys.argv[7]) # path to folder with roi binary masks
    in_roi = sys.argv[8] # True or False 
    temp_arg = False
    
    # filename for file you are calculating tSNR with
    fname = 'reg_moco.nii.gz'
    
    # path to functional target background mask for tBSNR calc
    background_mask_path = os.path.join('/home/ssbeast/Projects/SS/monkey_fmri/MTurk1_12TB/subjects', subject, 'mri/functional_target_background_mask.nii.gz')
    background_mask = np.nonzero(nib.load(background_mask_path).get_fdata())
    
    # directory containing roi binary masks if you want to mask anything out
    roi_mask_paths = [f for f in os.listdir(rois_dir) if ".nii.gz" in f and "~" not in f and "reg" not in f] # don't grab the "~" backup files
    
    # First create a binarized version of the masked epi to get mask of brain for each ima
    # For each session
    
    sesh_tSNR = []
    sesh_tBSNR = []
    
    if temp_arg: #len(glob.glob(os.path.join(session_dir, '*/reg_moco.nii.gz')))>0:
        print('reg_moco files are present; tSNR and tBSNR will be calculated')
    
        
        # For each ima in the session
        for root,directory,file in os.walk(session_dir,topdown=True):
             for f in file:
                if f == 'epi_masked.nii.gz':
                    fullpath = os.path.join(root, f)
                    to_mask = nib.load(fullpath).get_fdata()
                    to_mask = to_mask[:,:,:,0] # discard time dimension by just taking mask at first time point
                    binary_mask = np.where(to_mask>0.0, 1.0, 0.0)
                    
                    binary_img = nib.Nifti1Image(binary_mask, nib.load(fullpath).affine)
                    binary_img_outpath = os.path.join(os.path.dirname(fullpath), 'epi_masked_binary.nii.gz')
                    nib.save(binary_img, binary_img_outpath)
                    
                    #to_mask_one = np.nonzero(np.where(to_mask>0.0, 1.0, 0.0))
                    #mask_loc = np.nonzero(to_mask)
                    
        # Second calculate SNR measures
        
        # sesh_name = str(session_dir).split("/")[-1] # session name    
        # initialize lists
        sesh_tSNR = []
        sesh_tSNR_val = []
        sesh_tBSNR = []
        sesh_tBSNR_val = []
    
        ima = []
        
        roi_names_tSNR = []
        roi_names_tBSNR = []
        roi_tSNR_avgs = []
        roi_tBSNR_avgs = []
        roi_ima = []
        
        scanner_snr = []
        mion_dose = []
        num_good_runs = []
        receive_coils = []
        
        # For each ima in the session
        for root,directory,file in os.walk(session_dir,topdown=True):
             for f in file:       
                if f == fname: # load ima reg_moco file
    
                    fullpath = os.path.join(root, f)
                    epi = nib.load(fullpath).get_fdata() # neural data from one run
                    t_epi_mean = epi.mean(axis=3) # mean signal across time at each voxel; shape (128, 100, 42)
                    t_epi_std = epi.std(axis=3) # standard dev signal across time at each voxel; shape (128, 100, 42)
                    
                    epi_mask_path = os.path.join(root, 'epi_masked_binary.nii.gz') # grab binary brain mask for ima
                    epi_mask = nib.load(epi_mask_path).get_fdata()
                    epi_mask = np.nonzero(epi_mask) # generate mask locations
                    
                    background_vox = epi[background_mask] # mask out background voxels
                    t_bg_std = background_vox.std() # standard dev of background signal across time; single value (scalar)
                    
                    # tSNR = mean signal / std signal at each voxel
                    #tSNR_epi = t_epi_mean/t_epi_std
                    
                    # below was originally meant to deal with div by 0 but I don't think actually necessary
                    tSNR_epi = np.divide(t_epi_mean, t_epi_std, out=np.zeros_like(t_epi_mean), where = t_epi_std != 0)  #avoid division by 0
                    sesh_tSNR.append(tSNR_epi)
                    # mask to get avg value across brain; not perfect
                    tSNR_epi_masked = tSNR_epi[epi_mask]
                    tSNR_brain_avg = tSNR_epi_masked.mean()
                    sesh_tSNR_val.append(tSNR_brain_avg)
    
                    # tBSNR = mean signal at each voxel / std background signal
                    tBSNR_epi = t_epi_mean/t_bg_std
                    sesh_tBSNR.append(tBSNR_epi)
                    # mask to get avg value across brain; not perfect
                    tBSNR_epi_masked = tBSNR_epi[epi_mask]
                    tBSNR_brain_avg = tBSNR_epi_masked.mean()
                    sesh_tBSNR_val.append(tBSNR_brain_avg)
                    
                    scanner_snr.append(scan_snr_est)
                    mion_dose.append(amt_mion)
                    num_good_runs.append(num_runs)
                    receive_coils.append(r_coils)
                    
                    if in_roi:
                        
                        for j, mask_path in enumerate(roi_mask_paths): 
                            roi_name = str(mask_path).split(".")[0]
                            roi_names_tSNR.append(roi_name+'_tSNR')
                            roi_names_tBSNR.append(roi_name+'_tBSNR')
                            roi_dat = nib.load(os.path.join(rois_dir, mask_path)).get_fdata()
                            roi_loc = np.nonzero(roi_dat) # coordinates of roi
                            
                            roi_tSNR = tSNR_epi[roi_loc]
                            roi_tSNR_avg = roi_tSNR.mean()
                            roi_tSNR_avgs.append(roi_tSNR_avg)
                            
                            roi_tBSNR = tBSNR_epi[roi_loc]
                            roi_tBSNR_avg = roi_tBSNR.mean()
                            roi_tBSNR_avgs.append(roi_tBSNR_avg)
                            
                            roi_ima.append(str(os.path.basename(root)))
    
                    ima.append(str(os.path.basename(root)))
                    print('finished ima ', str(os.path.basename(root)))
                    
        # Calculate session tSNR (averaged across imas)
        sesh_tSNR_map = np.array(sesh_tSNR).mean(axis=0)
        sesh_tSNR_map_img = nib.Nifti1Image(sesh_tSNR_map, nib.load(fullpath).affine)
        sesh_tSNR_outpath = os.path.join(session_dir, 'tSNR_map.nii.gz')
        nib.save(sesh_tSNR_map_img, sesh_tSNR_outpath)
        # Calculate average tSNR across brain across voxels
        sesh_tSNR_val_mean = np.array(sesh_tSNR_val).mean()
        sesh_tSNR_val.append(sesh_tSNR_val_mean)
        
        # Calculate session tBSNR (averaged across imas)
        sesh_tBSNR_map = np.array(sesh_tBSNR).mean(axis=0)
        sesh_tBSNR_map_img = nib.Nifti1Image(sesh_tBSNR_map, nib.load(fullpath).affine)
        sesh_tBSNR_outpath = os.path.join(session_dir, 'tBSNR_map.nii.gz')
        nib.save(sesh_tBSNR_map_img, sesh_tBSNR_outpath)
        # Calculate average tSNR across brain across voxels
        sesh_tBSNR_val_mean = np.array(sesh_tBSNR_val).mean()
        sesh_tBSNR_val.append(sesh_tBSNR_val_mean)
        
        ima.append('Avg')
        scanner_snr.append(scan_snr_est)
        mion_dose.append(amt_mion)
        num_good_runs.append(num_runs)
        receive_coils.append(r_coils)
        
        # create session df
        session_SNR_report = pd.DataFrame({'ima': ima, 'scanner_SNR': scanner_snr, 'MION': mion_dose, 'num_runs': num_good_runs, 'receive_coils': receive_coils, 'whole_brain_tSNR': sesh_tSNR_val, 'whole_brain_tBSNR': sesh_tBSNR_val})
        #session_SNR_report.to_csv(os.path.join(sesh, 'SNR_report.csv'))
        
        if in_roi:
            roi_df_tSNR = pd.DataFrame({'ima': roi_ima, 'roi': roi_names_tSNR, 'tSNR': roi_tSNR_avgs})
            roi_df_wide_tSNR = roi_df_tSNR.pivot(index='ima', columns='roi', values='tSNR')
            roi_df_wide_tSNR.loc['Avg'] = roi_df_wide_tSNR.mean()
            roi_df_wide_tSNR = roi_df_wide_tSNR.reset_index()
            
            roi_df_tBSNR = pd.DataFrame({'ima': roi_ima, 'roi': roi_names_tBSNR, 'tBSNR': roi_tBSNR_avgs})
            roi_df_wide_tBSNR = roi_df_tBSNR.pivot(index='ima', columns='roi', values='tBSNR')
            roi_df_wide_tBSNR.loc['Avg'] = roi_df_wide_tBSNR.mean()
            roi_df_wide_tBSNR = roi_df_wide_tBSNR.reset_index()
            
            full_SNR_report = session_SNR_report.merge(roi_df_wide_tSNR, how = 'inner', on = 'ima').merge(roi_df_wide_tBSNR, how = 'inner', on = 'ima')
        else:
            full_SNR_report = session_SNR_report
        full_SNR_report.to_csv(os.path.join(session_dir, 'SNR_report.csv'))
        print('SNR report and images can be found at ', str(session_dir))
    # but if you have a sterilized session and only the epi_masked files are available, you can still get tSNR, but not tBSNR
    else:
        print('no reg_moco files are present; only tSNR will be calculated')
        # For each ima in the session THIS STEP IS KIND OF REDUNDANT BUT IT IS FINE YOU NEED IT ANYWAY SORT OF
        for root,directory,file in os.walk(session_dir,topdown=True):
             for f in file:
                if f == 'epi_masked.nii.gz':
                    fullpath = os.path.join(root, f)
                    to_mask = nib.load(fullpath).get_fdata()
                    to_mask = to_mask[:,:,:,0] # discard time dimension by just taking mask at first time point
                    binary_mask = np.where(to_mask>0.0, 1.0, 0.0)
                    
                    binary_img = nib.Nifti1Image(binary_mask, nib.load(fullpath).affine)
                    binary_img_outpath = os.path.join(os.path.dirname(fullpath), 'epi_masked_binary.nii.gz')
                    nib.save(binary_img, binary_img_outpath)
                    
        # calculate SNR measures
        
        # sesh_name = str(session_dir).split("/")[-1] # session name    
        # initialize lists
        sesh_tSNR = []
        sesh_tSNR_val = []
    
        ima = []
        
        roi_names_tSNR = []
        roi_tSNR_avgs = []
        roi_ima = []
        
        scanner_snr = []
        mion_dose = []
        num_good_runs = []
        receive_coils = []
        
        # For each ima in the session
        for root,directory,file in os.walk(session_dir,topdown=True):
             for f in file:       
                if f == 'epi_masked.nii.gz': # load ima epi_masked file
    
                    fullpath = os.path.join(root, f)
                    epi = nib.load(fullpath).get_fdata() # neural data from one run
                    t_epi_mean = epi.mean(axis=3) # mean signal across time at each voxel; shape (128, 100, 42)
                    t_epi_std = epi.std(axis=3) # standard dev signal across time at each voxel; shape (128, 100, 42)
                    
                    epi_mask_path = os.path.join(root, 'epi_masked_binary.nii.gz') # grab binary brain mask for ima
                    epi_mask = nib.load(epi_mask_path).get_fdata()
                    epi_mask = np.nonzero(epi_mask) # generate mask locations
                    
                    # tSNR = mean signal / std signal at each voxel
                    #tSNR_epi = t_epi_mean/t_epi_std
                    tSNR_epi = np.divide(t_epi_mean, t_epi_std, out=np.zeros_like(t_epi_mean), where = t_epi_std != 0)  #avoid division by 0
                    sesh_tSNR.append(tSNR_epi)
                    tSNR_epi_masked = tSNR_epi[epi_mask]
                    tSNR_brain_avg = tSNR_epi_masked.mean()
                    sesh_tSNR_val.append(tSNR_brain_avg)
    
        
                    scanner_snr.append(scan_snr_est)
                    mion_dose.append(amt_mion)
                    num_good_runs.append(num_runs)
                    receive_coils.append(r_coils)
                    
                    if in_roi:
                        
                        for j, mask_path in enumerate(roi_mask_paths): 
                            roi_name = str(mask_path).split(".")[0]
                            roi_names_tSNR.append(roi_name+'_tSNR')
                            roi_dat = nib.load(os.path.join(rois_dir, mask_path)).get_fdata()
                            roi_loc = np.nonzero(roi_dat) # coordinates of roi
                            
                            roi_tSNR = tSNR_epi[roi_loc]
                            roi_tSNR_avg = roi_tSNR.mean()
                            roi_tSNR_avgs.append(roi_tSNR_avg)

                            roi_ima.append(str(os.path.basename(root)))
    
                    ima.append(str(os.path.basename(root)))
                    print('finished ima ', str(os.path.basename(root)))
                    
        # Calculate session tSNR (averaged across imas)
        sesh_tSNR_map = np.array(sesh_tSNR).mean(axis=0)
        sesh_tSNR_map_img = nib.Nifti1Image(sesh_tSNR_map, nib.load(fullpath).affine)
        sesh_tSNR_outpath = os.path.join(session_dir, 'tSNR_map.nii.gz')
        nib.save(sesh_tSNR_map_img, sesh_tSNR_outpath)
        # Calculate average tSNR across brain across voxels
        sesh_tSNR_val_mean = np.array(sesh_tSNR_val).mean()
        sesh_tSNR_val.append(sesh_tSNR_val_mean)
        
        ima.append('Avg')
        scanner_snr.append(scan_snr_est)
        mion_dose.append(amt_mion)
        num_good_runs.append(num_runs)
        receive_coils.append(r_coils)
        
        # create session df
        session_SNR_report = pd.DataFrame({'ima': ima, 'scanner_SNR': scanner_snr, 'MION': mion_dose, 'num_runs': num_good_runs, 'receive_coils': receive_coils, 'whole_brain_tSNR': sesh_tSNR_val})
        #session_SNR_report.to_csv(os.path.join(sesh, 'SNR_report.csv'))
        
        if in_roi:
            roi_df_tSNR = pd.DataFrame({'ima': roi_ima, 'roi': roi_names_tSNR, 'tSNR': roi_tSNR_avgs})
            roi_df_wide_tSNR = roi_df_tSNR.pivot(index='ima', columns='roi', values='tSNR')
            roi_df_wide_tSNR.loc['Avg'] = roi_df_wide_tSNR.mean()
            roi_df_wide_tSNR = roi_df_wide_tSNR.reset_index()

            full_SNR_report = session_SNR_report.merge(roi_df_wide_tSNR, how = 'inner', on = 'ima')
        else:
            full_SNR_report = session_SNR_report
        full_SNR_report.to_csv(os.path.join(session_dir, 'SNR_report.csv'))
        print('SNR report and images can be found at ', str(session_dir))
    
