#!/usr/bin/env python3
#
# TODO: Add proper documentation.

import os

import numpy as np
import pandas as pd


def _get_unique(df, col):
    if not isinstance(col, str):
        col = str(col)
    return pd.unique(df[col])


def _load_img_info(img):
    import nibabel as nb

    img_ = nb.load(img)
    img_affine = img_.affine
    img_data = img_.get_fdata()
    return img_affine, img_data


def listify_paths(col):
    paths = []

    for p, _ in enumerate(col):
        for path in eval(col[p]):
            paths.append(path)
    return paths


def gen_region_map(dk):
    if not isinstance(dk, pd.DataFrame):
        dk = pd.read_csv(dk)

    return {
        "".join(eval(dk["img_path"][r])): dk["region"][r]
        for r in range(len(dk["region"]))
    }


def _get_cond_group_imgs(dk, cond_groups):
    from pathlib import Path

    # XXX: Hard-coded.
    _base_dir = Path("/home/bizon/Projects/MTurk1/MTurk1").absolute()
    paths = {}

    for cond_group in cond_groups:
        col = dk[dk["condition_group"] == cond_group]["beta_path"]
        col = col.reset_index(drop=True)
        paths[cond_group] = listify_paths(col)
        # Quick fix for relative paths.
        paths[cond_group] = [os.path.join(_base_dir, p) for p in paths[cond_group]]
    return paths


def get_imgs(dk, is_mask):
    if not isinstance(dk, pd.DataFrame):
        dk = pd.read_csv(dk)

    if is_mask:
        col = dk["img_path"]
        paths = listify_paths(col)
    elif not is_mask:
        cond_groups = _get_unique(dk, "condition_group")
        paths = _get_cond_group_imgs(dk, cond_groups)
    return paths


def apply_mask(img, mask_img):
    img_affine, img_data = _load_img_info(img)
    mask_affine, mask_data = _load_img_info(mask_img)

    if not np.allclose(img_affine, mask_affine):
        raise ValueError("")

    if img_data.shape != mask_data.shape:
        raise ValueError("")

    mask_idxs = np.nonzero(mask_data >= 0.5)
    masked_img_data = img_data[mask_idxs[0], mask_idxs[1], mask_idxs[2]]
    return masked_img_data


def get_mean_beta(imgs, mask_img):
    betas = []

    for img in imgs:
        masked_img_data = apply_mask(img, mask_img)
        betas.append(masked_img_data)
    return np.mean(betas)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-k", "--data-key")
    parser.add_argument("-r", "--region-paths")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    region_map = gen_region_map(args.region_paths)

    imgs = get_imgs(args.data_key, is_mask=False)
    mask_imgs = get_imgs(args.region_paths, is_mask=True)

    betas = {}

    for mask_img in mask_imgs:
        # XXX: Hard-coded for now.
        beta_ = get_mean_beta(imgs["uncolored_shapes"], mask_img)
        betas[region_map[mask_img]] = [beta_]

    betas_ = pd.DataFrame.from_dict(betas)
    betas_.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
