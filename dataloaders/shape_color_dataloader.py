import random

import os

from typing import Tuple, List
import torch
from torch import Tensor
import cv2
from dataloaders.base import PsychDataloader


class ShapeColorBasicData(PsychDataloader):

    def __int__(self, data_folder_path, exp_image_size, stim_frames=1):
        super.__init__(data_folder_path, exp_image_size, stim_frames)

    def __repr__(self):
        return "shape_color_basic"

    def get_batch(self, batch_size=10) -> Tuple[Tensor, List[str]]:
        condition_names = ['chromatic_shape_uncolored', 'chromatic_shape_colored', 'achromatic_shape', 'colored_circle']
        stimuli = torch.zeros((len(condition_names), self.stim_frames, batch_size, 3, self.exp_image_size[0],
                               self.exp_image_size[1]))
        for cond_idx, condition in enumerate(condition_names):
            condition_path = os.path.join(self.data_folder, condition)
            file_names = [f for f in os.listdir(condition_path) if ('.png' in f and f[0] != '.')]
            batch_files = random.choices(file_names, k=batch_size)
            for idx, file in enumerate(batch_files):
                img_path = os.path.join(condition_path, file)
                img = cv2.imread(os.path.join(condition_path, img_path))
                img = torch.from_numpy(img)
                if len(img.shape) == 2:
                    img = img[:, :, None]
                if img.shape[2] == 1:
                    img = torch.tile(img, (1, 1, 3))
                img = img.reshape((3, self.exp_image_size[0], self.exp_image_size[1]))
                stimuli[cond_idx, 0, idx, :, :, :] = img
        stim_mean = torch.mean(stimuli.flatten())
        stim_std = torch.std(stimuli.flatten())
        stimuli -= stim_mean
        stimuli /= stim_std
        return stimuli, condition_names