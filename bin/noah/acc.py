#!/usr/bin/env python3
#
# TODO: Add proper documentation.

import math
import multiprocessing as mp
import os
import queue
import random
import warnings

import nibabel as nb
import numpy as np
import pandas as pd
import torch

CONDITION_INTEGER_MAP = {
    1: 2,
    2: 4,
    3: 6,
    4: 8,
    5: 10,
    6: 12,
    7: 14,
    8: 2,
    9: 4,
    10: 6,
    11: 8,
    12: 10,
    13: 12,
    14: 14,
    15: 1,
    16: 3,
    17: 5,
    18: 7,
    19: 9,
    20: 11,
    21: 13,
    22: 1,
    23: 3,
    24: 5,
    25: 7,
    26: 9,
    27: 11,
    28: 13,
}


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


# NOTE: This class is taken from Spencer's decoder code.
class DataLoader:
    def __init__(
        self,
        base_dir,
        dk_path,
        batch_size,
        crop,
        ignore_sessions=(),
        ignore_classes=(),
        seed=4242,
        binary_target=None,
        binary_other=None,
        folds=1,
        use_behavior=True,
        verbose=False,
        device="cuda",
    ):
        self.base_dir = base_dir

        data = pd.read_csv(dk_path)
        for session in ignore_sessions:
            data = data[data["session"] != session]
        self.data = data.sample(frac=1.0, ignore_index=True, random_state=seed)

        self.batch_size = batch_size

        # Image-related properties
        self.full_size = None
        self.affine = None
        self.hdr = None

        self.binary_target = binary_target
        self.binary_other = binary_other
        self.ignore_classes = ignore_classes

        self.use_behavior = use_behavior
        self.verbose = verbose

        self.current_fold = 0
        self.folds = folds

        self.crop = crop
        size = max([c[1] - c[0] for c in crop])
        self.shape = (size, size, size)

        color_set = self._get_cond_set("colored_blobs", correct_only=self.use_behavior)
        self.color_all = color_set
        self.color_train = color_set[: int(80 * len(color_set) / 100)]
        self.color_test = color_set[int(80 * len(color_set) / 100) :]

        shape_set = self._get_cond_set(
            "uncolored_shapes", correct_only=self.use_behavior
        )
        self.shape_all = shape_set
        self.shape_train = shape_set[: int(80 * len(shape_set) / 100)]
        self.shape_test = shape_set[int(80 * len(shape_set) / 100) :]

        self.device = device
        self._processed = 0
        self._max_queue_size = 20
        self.set_mem = {}

    def _get_cond_set(self, cond, correct_only=True):
        data = self.data[self.data["condition_group"] == cond].copy()

        if correct_only:
            data = data[data["correct"] == 1].reset_index(drop=True)
        else:
            data.reset_index(drop=True)

        conds = sorted(pd.unique(data["condition_integer"]))
        data_ = data.copy()

        for _, cond in enumerate(conds):
            # Returns a warning.
            data_["condition_integer"].loc[data["condition_integer"] == cond] = int(
                CONDITION_INTEGER_MAP[cond]
            )
        data = data_

        img = nb.load(os.path.join(self.base_dir, eval(data["beta_path"][0])[0]))
        self.full_size = img.get_fdata().shape
        self.affine = img.affine
        self.hdr = img.header

        return data

    def _make_full(self, data):
        full_size = np.zeros(self.full_size, dtype=float)
        crop_size = self.shape[0]
        crop = []

        for _, c in enumerate(self.crop):
            dim = c[1] - c[0]
            size_diff = crop_size - dim
            pad_left = int(math.floor(size_diff / 2))
            pad_right = int(math.ceil(size_diff / 2))
            crop.append((pad_left, crop_size - pad_right))

        full_size[
            self.crop[0][0] : self.crop[0][1],
            self.crop[1][0] : self.crop[1][1],
            self.crop[2][0] : self.crop[2][1],
        ] = data[
            crop[0][0] : crop[0][1],
            crop[1][0] : crop[1][1],
            crop[2][0] : crop[2][1],
        ]
        return full_size

    @staticmethod
    def _pad_data(data, time_axis=3):
        if time_axis < np.ndim(data):
            size = max(data.shape[:time_axis] + data.shape[time_axis + 1 :])
        else:
            size = max(data.shape)

        pad = [0] * np.ndim(data)

        for a in range(len(pad)):
            if a != time_axis:
                ideal_size = (size - data.shape[a]) / 2
                pad[a] = (int(np.floor(ideal_size)), int(np.ceil(ideal_size)))
            else:
                pad[a] = (0, 0)

        padded_data = np.pad(data, pad, mode="constant", constant_values=(0, 0))
        return padded_data

    def get_stats(self, dtype):
        try:
            data = eval("".join(["self.", dtype.strip()]))
        except AttributeError:
            raise ValueError("")

        betas = []

        for paths_ in data["beta_path"]:
            paths = eval(paths_)
            beta_ = []

            for path in paths[2:]:
                img = nb.load(os.path.join(self.base_dir, path))
                # Horrible variable names. I'm honestly ashamed of
                # myself.
                betas_ = img.get_fdata()[
                    self.crop[0][0] : self.crop[0][1],
                    self.crop[1][0] : self.crop[1][1],
                    self.crop[2][0] : self.crop[2][1],
                ]
                beta_.append(betas_)

            beta_ = np.stack(beta_, axis=0)
            betas.append(beta_)

        betas = np.stack(betas, axis=0)
        mean_beta = self._pad_data(betas.mean(axis=0), time_axis=0)
        std_beta = self._pad_data(betas.std(axis=0), time_axis=0)
        return mean_beta, std_beta

    def crop_img(self, img, make_cube=True):

        cropped_img = img[
            self.crop[0][0] : self.crop[0][1],
            self.crop[1][0] : self.crop[1][1],
            self.crop[2][0] : self.crop[2][1],
        ]

        if make_cube:
            cropped_img = self._pad_data(cropped_img)
        return cropped_img

    @staticmethod
    def translate_img(img, x, y, z):
        _, width, height, depth = img.shape

        translated_img = np.zeros_like(img)

        i = np.arange(width) - x
        j = np.arange(height) - y
        k = np.arange(depth) - z

        if x < 0:
            i = np.arange(-x, width - x)
        if y < 0:
            j = np.arange(-y, height - y)
        if z < 0:
            k = np.arange(-z, depth - z)

        i = np.clip(i, 0, width - 1)
        j = np.clip(j, 0, height - 1)
        k = np.clip(k, 0, depth - 1)

        translated_img[:, i[:, None, None], j[None, :, None], k[None, None, :]] = img
        return translated_img

    def gen_data(
        self, data, noise_var=0.2, translation_var=0.67, max_base_ex=1, make_cube=True
    ):
        options = sorted(list(set(range(1, 15)) - set(self.ignore_classes)))

        if self.binary_target is not None:
            if random.choice([True, False]):
                ex_class = self.binary_target
                target = 0
            else:
                if self.binary_other == "all" or self.binary_other is None:
                    options = set(options) - {self.binary_target}
                    ex_class = int(random.choice(list(options)))
                else:
                    ex_class = self.binary_other
                target = 1
        else:
            ex_class = int(random.choice(options))
            target = options.index(ex_class)

        basis_dim = np.random.randint(1, max_base_ex + 1)
        trajectory = np.random.random((basis_dim, 1, 1, 1, 1))
        trajectory /= np.sum(trajectory)
        class_data = data[data["condition_integer"] == ex_class]
        basis_idxs = np.random.randint(0, len(class_data), size=(basis_dim,))

        data_ = class_data.iloc[basis_idxs]
        betas = []

        for _, paths_ in enumerate(data_["beta_path"]):
            paths = eval(paths_)
            beta_ = []

            for path in paths[2:]:
                if path not in self.set_mem:
                    img = nb.load(os.path.join(self.base_dir, path))
                    betas_ = self.crop_img(img.get_fdata(), make_cube=make_cube)

                    if np.count_nonzero(np.isnan(betas_)) > 0:
                        warnings.warn("")

                    self.set_mem[path] = betas_
                else:
                    betas_ = self.set_mem[path]
                beta_.append(betas_)

            beta_ = np.stack(beta_, axis=0)
            betas.append(beta_)

        betas = np.stack(betas, axis=0)
        betas *= trajectory
        beta = betas.sum(axis=0)

        if translation_var > 0.0:
            translation = np.round(
                np.random.normal(loc=0, scale=translation_var, size=(3,))
            )
            beta = self.translate_img(
                beta, int(translation[0]), int(translation[1]), int(translation[2])
            )

        std_ = np.std(beta)
        noise = np.random.normal(0, std_ * noise_var, beta.shape)
        beta += noise

        return beta, target

    def get_batch(self, batch_size, data, resample=False, make_cube=True):
        betas = []
        targets = []

        for _ in range(int(batch_size)):
            if resample:
                beta, target = self.gen_data(
                    data,
                    noise_var=0.2,
                    translation_var=0.67,
                    max_base_ex=1,
                    make_cube=make_cube,
                )
            else:
                beta, target = self.gen_data(
                    data,
                    noise_var=0.0,
                    translation_var=0.0,
                    max_base_ex=1,
                    make_cube=make_cube,
                )

            betas.append(beta)
            targets.append(target)
        betas = np.stack(betas, axis=0)
        targets = np.array(targets, dtype=int)
        return betas, targets

    def queue_data(
        self,
        data,
        batch_size,
        n_batches,
        resample,
        standardize,
        mean_beta,
        std_beta,
        queuer,
        make_cube=True,
    ):
        while self._processed < n_batches:
            betas, targets = self.get_batch(
                batch_size, data, resample=resample, make_cube=make_cube
            )

            if standardize:
                betas = (betas - mean_beta[None, :, :, :, :]) / std_beta[
                    None, :, :, :, :
                ]
                betas = np.nan_to_num(betas, nan=0.0, neginf=0.0, posinf=0.0)

            try:
                queuer.put((betas, targets), block=True, timeout=120)
            except queue.Full:
                warnings.warn("")
                del betas
                del targets
                # XXX: It might be good to put something else here.
                return None

    def batch_iter(
        self,
        dtype,
        n_train_batches=1000,
        return_all=False,
        standardize=False,
        resample=True,
        n_workers=16,
        make_cube=True,
    ):
        try:
            data = eval("".join(["self.", dtype.strip()]))
        except AttributeError:
            raise ValueError("")

        mean_beta = 0
        std_beta = None

        if standardize:
            mean_beta, std_beta = self.get_stats(dtype)

        data = data.sample(frac=1.0, ignore_index=True)

        if return_all:
            batch_size = len(data)
        else:
            batch_size = self.batch_size

        self._processed = 0
        context = mp.get_context("spawn")
        queuer = context.Queue(maxsize=self._max_queue_size)
        use_mp = n_workers > 1
        workers = []

        if use_mp:
            for w in range(n_workers):
                p = context.Process(
                    target=self.queue_data,
                    args=(
                        data,
                        batch_size,
                        n_train_batches,
                        resample,
                        standardize,
                        mean_beta,
                        std_beta,
                        queuer,
                        make_cube,
                    ),
                )
                p.start()
                workers.append(p)

        for _ in range(n_train_batches):
            if use_mp:
                try:
                    res = queuer.get(block=True, timeout=120)
                except queue.Full:
                    warnings.warn("")
                    break

                self._processed += 1
                betas, targets = res
            else:
                betas, targets = self.get_batch(
                    batch_size, data, resample=resample, make_cube=make_cube
                )
            yield betas, targets

        if use_mp:
            print("")
            queuer.close()

            for w in range(n_workers):
                workers[w].terminate()
                workers[w].kill()


# NOTE: This class is an adaptation of code written by Helen and
# Karthik.
class DecodingAccuracy:
    def __init__(self, dl, model, test_on, mask_img=None):
        self.dl = dl

        self.model = model
        self.test_set = dl.batch_iter(
            test_on, n_train_batches=100, standardize=True, n_workers=2
        )

        self.mask_img = mask_img

    def _get_data(self):
        logits = []
        batch_targets = []

        with torch.no_grad():
            for n, (class_, target) in enumerate(self.test_set):
                target = torch.from_numpy(target)
                targets = (
                    target.long()
                    .to(self.model.device)
                    .reshape([-1] + [1] * self.model.dim)
                )
                targets = torch.tile(targets, [1] + list(self.model.out_spatials[-1]))

                targets_ = []

                for e in range(targets.shape[0]):
                    targets_.append(int(targets[e, 0, 0, 0]))

                batch_targets.append(targets_)

                batch_size = len(target)
                residual = self.model.step(class_).reshape(
                    (
                        [batch_size, self.model.n_classes]
                        + list(self.model.out_spatials[-1])
                    )
                )
                residual_ = residual.cpu()

                for b in range(residual_.shape[0]):
                    for c in range(residual_.shape[1]):
                        logit = [n, b, c, residual_[b, c, :, :, :]]
                        logits.append(logit)

            if self.model.dim == 2:
                method = "bilinear"
            if self.model.dim == 3:
                method = "trilinear"
            else:
                raise ValueError("")

            for l in logits:
                l[3] = (
                    torch.nn.functional.interpolate(
                        l[3].unsqueeze(0).unsqueeze(0),
                        size=(64, 64, 64),
                        mode=method,
                    )
                    .squeeze()
                    .numpy()
                )
                l[3] = self.dl._make_full(l[3])

        return logits, batch_targets

    def _vote(self, logits):
        batch_votes = []

        # XXX: Hard-coded because it's a pain in the ass to get these
        # values from `logits`.
        for batch in range(100):
            batch_ = [l for l in logits if l[0] == batch]

            vote = []

            for ex in range(15):
                ex_ = [e for e in batch_ if e[1] == ex]

                classes = []
                confidence = []

                for class_ in range(6):
                    c_ = [c for c in ex_ if c[2] == class_]

                    data_ = c_[0][3]

                    if self.mask_img is not None:
                        mask = (nb.load(self.mask_img).get_fdata() > 0.5).squeeze()
                        data = data_[mask]
                    else:
                        data = np.array(data_).flatten()

                    classes.append(data)
                    confidence.append(sum(data))

                vote.append(np.argmax(np.array(confidence)))
            batch_votes.append(vote)
        return batch_votes

    def calc_acc(self):
        logits, batch_targets = self._get_data()
        batch_votes = self._vote(logits)

        count = 0
        correct = 0

        for v, t in zip(batch_votes, batch_targets):
            for vote, target in zip(v, t):
                count += 1

                if vote == target:
                    correct += 1

        return correct / count


def main():
    import pickle
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("-k", "--data-key")
    parser.add_argument("-r", "--region-paths")
    parser.add_argument("-m", "--model-path")
    parser.add_argument("-f", "--folds")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    # XXX: Hard-coded for Jeeves.
    base_dir_ = Path("/home/bizon/Projects/MTurk1/MTurk1").absolute()
    dk_path_ = args.data_key
    batch_size_ = 15
    crop_ = ((38, 89), (13, 77), (0, 42))
    ignore_classes_ = (1, 4, 5, 8, 9, 12, 13, 14)
    dl_ = DataLoader(
        base_dir=base_dir_,
        dk_path=dk_path_,
        batch_size=batch_size_,
        crop=crop_,
        ignore_classes=ignore_classes_,
    )

    region_map = gen_region_map(args.region_paths)

    # XXX: Lazy workaround.
    r_ = pd.read_csv(args.region_paths)
    mask_imgs = listify_paths(r_["img_path"])

    with open(args.model_path, "+rb") as f:
        model_ = pickle.load(f)

    # XXX: Hard-coded for within-cue decoding.
    if "color_2_shape" in args.model_path:
        test_on_ = "color_test"
    elif "shape_2_color" in args.model_path:
        test_on_ = "shape_test"

    accs = {}

    for mask_img in mask_imgs:
        for f in range(int(args.folds)):
            acc_ = DecodingAccuracy(
                dl=dl_, model=model_, test_on=test_on_, mask_img=mask_img
            )

            if f == 0:
                accs[region_map[mask_img]] = [acc_.calc_acc()]
            else:
                accs[region_map[mask_img]].append(acc_.calc_acc())

    accs_ = pd.DataFrame.from_dict(accs)
    accs_.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
