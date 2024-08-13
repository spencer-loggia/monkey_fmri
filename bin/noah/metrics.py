#!/usr/bin/env python3
#
# Given directories containing contrast and ROI images, return the mean
# magnitude (i.e., z-score) of each contrast within each ROI.
#
"""Attempt to assess data quality."""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONTRAST_NAME_PATTERNS = {
    "decode": [
        "probe_chromatic_uncolored_minus_colored_circle",
        "probe_chromatic_uncolored_minus_gray",
        "probe_colored_circle_minus_gray",
    ],
    # TODO: Add this in.
    "passive": [""],
}

# For creating the DataFrame of contrast magnitudes.
CONTRAST_NAME_MAP = {
    "probe_chromatic_uncolored_minus_colored_circle": "uncolored_minus_colored",
    "probe_chromatic_uncolored_minus_gray": "uncolored_minus_grey",
    "probe_colored_circle_minus_gray": "colored_minus_grey",
}

# XXX: Hard-coded to work with ROIs created by Helen.
REGION_NAME_MAP = {
    "ant_IT": "AIT",
    "entorhinal": "ERh",
    "far_IT": "AM",
    "IT": "IT",  # Redundant, but necessary.
    "mid_IT": "CIT",
    "post_IT": "PIT",
    "v1v2v3": "V1-V3",
    "V4": "V4",  # Redundant, but necessary.
}


def _style_plot():
    import matplotlib as mpl


def _get_img_name(img):
    """Return image name from a path."""
    return os.path.basename(img).split(".")[0]


# NOTE: Assumes input is in the form `task_YYYYMMDD`.
def _get_ses_date(img):
    """Return session date from path."""
    from dateutil.parser import isoparse

    _date = os.path.dirname(img).split("_")[-1]
    date = isoparse(_date)
    # Don't want to include time.
    date = date.date().isoformat()
    return date


def load_imgs(input_dir, img_type, task=None, combined=False):
    """Return paths of task-relevant and region images."""
    _contrast_flags = [
        "c",
        "contrast",
    ]
    # Probably overkill.
    _mask_flags = [
        "r",
        "roi",
        "region",
        "m",
        "mask",
    ]

    if img_type in _contrast_flags and task is not None:
        # This looks really complicated, but all it does is return the
        # paths to the contrast images of a particular task. The `[!~]`
        # at the end of the pattern just means that any
        # files that include `~` in their names are excluded.
        imgs = [
            i
            for p in range(len(CONTRAST_NAME_PATTERNS[task]))
            for i in glob.glob(
                f"{input_dir}/*/{CONTRAST_NAME_PATTERNS[task][p]}.nii.*[!~]"
            )
            if os.path.isfile(i)
        ]

    if img_type in _contrast_flags and task is None:
        raise ValueError("")

    if img_type in _mask_flags and combined:
        imgs = [i for i in glob.glob(f"{input_dir}/*.nii.*[!~]") if os.path.isfile(i)]
    elif img_type in _mask_flags and not combined:
        imgs = [
            i for i in glob.glob(f"{input_dir}/*?h*.nii.*[!~]") if os.path.isfile(i)
        ]

    return imgs


def apply_mask(img, mask_img):
    """Extract a region of a whole-brain image."""
    import nibabel as nib

    img_data = nib.load(img).get_fdata()
    mask_data = nib.load(mask_img).get_fdata()

    mask_idxs = np.nonzero(mask_data >= 0.5)
    masked_img_data = img_data[mask_idxs[0], mask_idxs[1], mask_idxs[2]]
    return masked_img_data


def get_mean_magnitude(contrast_img, mask_imgs):
    """Return average magnitude over a set of regions."""
    magnitude = []

    for mask_img in mask_imgs:
        masked_img_data = apply_mask(contrast_img, mask_img)
        magnitude.append(np.mean(masked_img_data))
    return magnitude


def plot_magnitudes(contrasts, region, output_dir=None, show_plots=False):
    """Show average magnitudes across regions."""
    dates = [d for d in contrasts["session"]]
    _contrasts = contrasts.drop("session", axis=1)

    for c, contrast in enumerate(_contrasts):
        _, ax = plt.subplots()
        # XXX: Is this correct? Am I grabbing all contrasts?
        ax.bar(dates, [i[c] for i in _contrasts[contrast]])
        plt.title(contrast)

        if show_plots:
            plt.show()

        if output_dir is not None:
            plt.savefig(
                os.path.join(
                    output_dir, f"desc-{region}_{CONTRAST_NAME_MAP[contrast]}.png"
                )
            )


# TODO: Add SNR plotting from Helen's code.


def main():
    """Main program."""
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-s", "--sessions")
    parser.add_argument("-t", "--task")
    parser.add_argument("-r", "--regions")
    parser.add_argument("-b", "--both-hemispheres")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    contrast_imgs = load_imgs(args.sessions, img_type="c", task=str(args.task))
    mask_imgs = load_imgs(
        args.regions, img_type="r", combined=bool(args.both_hemispheres)
    )

    # XXX: Hard-coded.
    _regions = ["ant_IT", "post_IT", "v1v2v3"]
    mask_imgs_ = [m for m in mask_imgs if _get_img_name(m) in _regions]

    _dates = pd.Series(
        [
            _get_ses_date(i)
            for i in contrast_imgs[:: len(CONTRAST_NAME_PATTERNS[args.task])]
        ]
    )
    contrasts_ = {}

    # TODO: Rework this.
    for contrast_img in contrast_imgs:
        if CONTRAST_NAME_MAP[_get_img_name(contrast_img)] not in contrasts_:
            contrasts_[CONTRAST_NAME_MAP[_get_img_name(contrast_img)]] = [
                get_mean_magnitude(contrast_img, mask_imgs_)
            ]
        else:
            contrasts_[CONTRAST_NAME_MAP[_get_img_name(contrast_img)]].append(
                get_mean_magnitude(contrast_img, mask_imgs_)
            )

    contrasts = pd.DataFrame.from_dict(contrasts_)
    contrasts.insert(loc=0, column="session", value=_dates)
    for r in _regions:
        plot_magnitudes(contrasts, r, show_plots=True)


if __name__ == "__main__":
    main()
