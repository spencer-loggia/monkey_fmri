# MTurk1 move nifti from functional to anatomical space
# Adapted from / using Spencer pipeline functions such as support_functions.apply_warp() and support_functions.generate_subject_overlays(), which utilize ANTs and Freesurfer commands, by Helen
# Moves contrasts in functional space to inflated hemisphere surface
# via functional->downsampled t1 space (which shares physical space with high res t1)->surface
# i.e., there is no voxel scaling to the high res, etc., once image is in anat space
# Smoothing if wanted is accomplished in functional space.
# To change surface projection parameters, look at --projfrac-max or avg; min, max, delta
# This does NOT support binary files (i.e., roi masks)

import sys

if len(sys.argv) < 3:
    print(
        "Please supply (1) subject name, (2) path to directory containing functional space niftis, (3) the kind of contrasts, choose from [snr, scp, scd, any], and (4) functional img smoothing in mm (0 for none)"
    )
else:
    import os
    import multiprocessing

    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn", force=True)
    import sys

    sys.path.append(os.path.abspath("/home/ssbeast/Projects/SS/monkey_fmri/bin"))
    from support_functions import apply_warp
    from preprocess import _pad_to_cube
    from analysis import _find_scale_factor
    from analysis import _scale_and_standardize
    import subprocess
    import nibabel as nib
    import numpy as np
    from nilearn.image import smooth_img

    if __name__ == "__main__":
        # Set subject
        # subject = 'jeeves'
        subject = str(sys.argv[1])
        # Directory of niftis you want to move to surface
        # nii_dir = '/media/ssbeast/DATA/Users/Helen/temp/six_imas'
        nii_dir = str(sys.argv[2])
        kind = str(sys.argv[3])  # one of SNR, scp, scd, any
        smooth = float(sys.argv[4])  # mm smoothing

        # subjects dir
        sd = "/home/ssbeast/Projects/SS/monkey_fmri/MTurk1_12TB/subjects"
        # Set fMRI working directory
        os.environ["FMRI_WORK_DIR"] = os.path.join(sd, subject)
        # Subject root, which is Jeeves' subject folder
        subject_root = os.path.realpath(os.environ.get("FMRI_WORK_DIR"))
        # Project root, which is MT1
        project_root = os.path.realpath(os.path.join(subject_root, "..", ".."))

        if kind == "snr":
            nii_paths = [
                f
                for f in os.listdir(nii_dir)
                if ".nii.gz" in f and "SNR" in f or "probe" in f and "~" not in f
            ]
        elif kind == "scd":
            nii_paths = [
                f
                for f in os.listdir(nii_dir)
                if ".nii.gz" in f and "minus" in f or "probe" in f and "~" not in f
            ]
        elif kind == "scp":
            nii_paths = [
                f
                for f in os.listdir(nii_dir)
                if ".nii.gz" in f and "minus" in f or "scp" in f and "~" not in f
            ]
        elif kind == "any":
            nii_paths = [
                f
                for f in os.listdir(nii_dir)
                if ".nii.gz" in f
                and "~" not in f
                and "itk" not in f
                and "3d" not in f
                and "Warped" not in f
            ]
        else:
            print("any?")

        for i, path in enumerate(nii_paths):
            path = nii_dir + "/" + path
            if smooth != 0:
                img = smooth_img(path, smooth)
                print(
                    "Smoothing will decrease noise but lose spatial detail. You are using full-width half-maximum smoothing over ",
                    smooth,
                    " mm",
                )
                smooth_name = (
                    str(smooth).split(".")[0] + "_" + str(smooth).split(".")[1] + "mm"
                )
                path = (
                    str(path).split(".")[0]
                    + "_smoothed_"
                    + str(smooth_name)
                    + ".nii.gz"
                )
                # path = str(path).split('.')[0]+'_smoothed_'+str(int(round(smooth)))+'.nii.gz'
                nib.save(img, path)
            else:
                path = path
            nii_paths[i] = path

        # Set anatomical target
        if subject == "jeeves":
            volume_in_target_space = subject_root + "/mri/ds_final_t1.nii.gz"
            orig_high_res_anatomical = subject_root + "/mri/avT1.nii.gz"
            orig_low_res_anatomical = subject_root + "/mri/ds_final_t1.nii.gz"
        else:
            volume_in_target_space = subject_root + "/mri/ds_avT1.nii.gz"
            orig_high_res_anatomical = subject_root + "/mri/masked_avT1.nii.gz"
            orig_low_res_anatomical = subject_root + "/mri/ds_avT1.nii.gz"

        # Set transform text file
        forward_gross_transform_path = subject_root + "/mri/itkManual.txt"
        # Set fine transform
        fine_transform_path = subject_root + "/mri/Composite.h5"

        # Apply functional to anatomical warp using support functions apply_warp function
        func_to_anat = apply_warp(
            nii_paths,
            volume_in_target_space,
            forward_gross_transform_path,
            fine_transform_path,
        )
        hemis = ["lh", "rh"]

        anat_nii_paths = [
            f
            for f in os.listdir(nii_dir)
            if ".nii.gz" in f and "reg" in f and "~" not in f
        ]
        print(anat_nii_paths)
        for i, path in enumerate(anat_nii_paths):
            path = nii_dir + "/" + path
            anat_nii_paths[i] = path

        # Move from anatomical to surface and output .mgh overlay file
        for nifti in anat_nii_paths:
            for hemi in hemis:
                n = os.path.splitext(os.path.splitext(os.path.basename(nifti))[0])[0]
                d = os.path.dirname(nifti)
                surf_out = os.path.join(d, "sigsurface_" + hemi + "_" + n + ".mgh")
                subprocess.run(
                    [
                        "mri_vol2surf",
                        "--mov",
                        nifti,
                        "--regheader",
                        subject,
                        "--projfrac-max",
                        ".05",
                        ".95",
                        ".025",
                        "--interp",
                        "nearest",
                        "--hemi",
                        hemi,
                        "--out",
                        surf_out,
                        "--sd",
                        sd,
                    ]
                )
                print("saved at:", surf_out)
