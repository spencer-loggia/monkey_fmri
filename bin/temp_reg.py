# fixed is /mnt/mturk1_12tb/Projects/MTurk1/MTurk1/subjects/wooster/mri/masked_ds_avT1.nii.gz, moving is /mnt/mturk1_12tb/Projects/MTurk1/MTurk1/subjects/wooster/mri/masked_itkManual.nii.gz
# outP is /mnt/mturk1_12tb/Projects/MTurk1/MTurk1/subjects/wooster/mri/coreg_3df.nii.gz
# def temp_antsCoreg(fixedP, movingP, outP, radiusi1, radiusi2, updateFV, totalFV, outToken, initialTrsnfrmP=None,
#              across_modalities=False, outPref='antsReg', nonlinear=True,
#              run=True, n_jobs=64, full=False):

if __name__ == "__main__":
    import os
    import subprocess
    import sys

    fixedP = "/mnt/mturk1_12tb/Projects/MTurk1/MTurk1/subjects/wooster/mri/masked_ds_avT1.nii.gz"
    movingP = "/mnt/mturk1_12tb/Projects/MTurk1/MTurk1/subjects/wooster/mri/masked_itkManual.nii.gz"
    outdir = "/mnt/mturk1_12tb/Projects/MTurk1/MTurk1/subjects/wooster/mri"

    radiusi1 = str(sys.argv[1])
    radiusi2 = str(sys.argv[2])
    updateFV = str(sys.argv[3])
    totalFV = str(sys.argv[4])
    outToken = str(sys.argv[5])

    def temp_antsCoreg(
        fixedP,
        movingP,
        outdir,
        radiusi1,
        radiusi2,
        updateFV,
        totalFV,
        outToken,
        nonlinear=True,
    ):
        cur_dir = os.getcwd()
        os.chdir(outdir)
        print(outdir)

        cmd = (
            "antsRegistration"
            " --verbose 1 --dimensionality 3 --float 0 --collapse-output-transforms 1 --write-composite-transform 1"
            " --output [ ./,./Warped_"
            + outToken
            + ".nii.gz,./InverseWarped_"
            + outToken
            + ".nii.gz ] "
            " --interpolation Linear --use-histogram-matching 0 --winsorize-image-intensities [ 0.005,0.995 ]"
            " --initial-moving-transform [" + fixedP + "," + movingP + ",1 ]"
            " --transform Affine[ 0.01 ] --metric MI[ "
            + fixedP
            + ","
            + movingP
            + ",1,32,Regular,0.25 ]"
            " --convergence [ 500,1e-9,10 ] --shrink-factors 1 --smoothing-sigmas 0vox"
        )
        if nonlinear:
            cmd = (
                cmd
                + " --transform SyN[ .01,"
                + updateFV
                + ","
                + totalFV
                + " ] --metric CC[ "
                + fixedP
                + ","
                + movingP
                + ",1,"
                + radiusi1
                + " ]"
                " --convergence [ 300x300,1e-9,25 ] --shrink-factors 2x1 --smoothing-sigmas 1x0vox"
                " --transform SyN[ .01,"
                + updateFV
                + ","
                + totalFV
                + " ] --metric CC[ "
                + fixedP
                + ","
                + movingP
                + ",1,"
                + radiusi2
                + " ]"
                " --convergence [ 300,1e-9,25 ] --shrink-factors 1 --smoothing-sigmas 0vox"
            )

        print(cmd)
        subprocess.call(cmd, shell=True)

        frwdTrnsP = os.path.join(outdir, "Composite_" + str(outToken) + ".h5")
        invTrnsP = os.path.join(outdir, "InverseComposite_" + str(outToken) + ".h5")
        os.chdir(os.path.realpath(cur_dir))
        return frwdTrnsP, invTrnsP

    temp_antsCoreg(
        fixedP=fixedP,
        movingP=movingP,
        outdir=outdir,
        radiusi1=radiusi1,
        radiusi2=radiusi2,
        updateFV=updateFV,
        totalFV=totalFV,
        outToken=outToken,
        nonlinear=True,
    )
