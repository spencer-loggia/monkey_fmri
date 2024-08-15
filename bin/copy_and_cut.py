import sys
import os
import shutil
import nibabel as nib
import numpy as np

base = str(sys.argv[1])  # session directory
ima = str(sys.argv[2])
cutoff = int(sys.argv[3])  # 48, 96, 144

if os.path.exists(os.path.join(base, ima, "epi_masked_full_.nii.gz")):
    print("full epi already saved, not re-copying")
else:
    shutil.copy(
        os.path.join(base, ima, "epi_masked.nii.gz"),
        os.path.join(base, ima, "epi_masked_full.nii.gz"),
    )
epi = nib.load(os.path.join(base, ima, "epi_masked.nii.gz"))
aff = epi.affine
dat = epi.get_fdata()
new = dat[:, :, :, :cutoff]
img = nib.Nifti1Image(new, aff)
nib.save(img, os.path.join(base, ima, "epi_masked.nii.gz"))

print(
    "img at",
    os.path.join(base, ima, "epi_masked.nii.gz"),
    " now has dim ",
    new.shape,
    " meaning ",
    new.shape[-1],
    " time points",
)
# base = '/home/ssbeast/Projects/SS/monkey_fmri/MTurk1_12TB/subjects/wooster/sessions/congruency_20240723'
# ima = '18'
# cutoff = 96
