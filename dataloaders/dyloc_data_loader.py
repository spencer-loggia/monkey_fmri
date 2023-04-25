import random
from dataloaders.base import PsychDataloader
import os
import torch
import cv2

def make_downseampled_stim(path_to_dyloc_source, out_dir):
    stim_dirs = os.listdir(path_to_dyloc_source)
    for stim_dir in stim_dirs:
        stim_dir_path = os.path.join(path_to_dyloc_source, stim_dir)
        if not os.path.isdir(stim_dir_path):
            continue
        stim_out_dir = os.path.join(out_dir, stim_dir)
        if not os.path.exists(stim_out_dir):
            os.mkdir(stim_out_dir)
        for i, vid_clip_name in enumerate(os.listdir(stim_dir_path)):
            vid_clip_path = os.path.join(stim_dir_path, vid_clip_name)
            vid_clip_out = os.path.join(stim_out_dir, "exp" + str(i))
            if not os.path.exists(vid_clip_out):
                os.mkdir(vid_clip_out)
            if '.mov' not in vid_clip_name:
                continue
            vid = cv2.VideoCapture(vid_clip_path)
            print("read video ", vid_clip_name)
            frame_num = -1
            im_num = 0
            while vid.isOpened():
                frame_num += 1
                if (frame_num % 3) != 0:
                    ret = vid.grab()
                    if not ret:
                        break
                    continue
                else:
                    ret, frame = vid.read()
                if ret:
                    img_out = os.path.join(vid_clip_out, str(im_num) + "_frame.png")
                    frame = cv2.resize(frame, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(img_out, frame)
                    im_num += 1


class DylocDataloader(PsychDataloader):

    def __int__(self, data_folder_path, exp_image_size, stim_frames):
        super.__init__(data_folder_path, exp_image_size, stim_frames)

    def __repr__(self):
        return "dyloc"

    def get_batch(self, num_vids_in_batch=10):
        condition_names = ['bw_faces', 'bw_bodies', 'bw_scenes', 'bw_scrambled', 'bw_objects', 'c_faces', 'c_bodies', 'c_scenes', 'c_scrambled', 'c_objects']
        stimuli = torch.zeros((len(condition_names), self.stim_frames, num_vids_in_batch, 3, self.exp_image_size[0], self.exp_image_size[1]))
        for cond_idx, condition in enumerate(condition_names):
            condition_path = os.path.join(self.data_folder, condition)
            file_names = [f for f in os.listdir(condition_path) if ('exp' in f and f[0] != '.')]
            batch_files = random.choices(file_names, k=num_vids_in_batch)
            for idx, file in enumerate(batch_files):
                vid_path = os.path.join(condition_path, file)
                imgs = sorted([im for im in os.listdir(vid_path) if ('.png' in im and im[0] != '.')])
                for frame, img_path in enumerate(imgs):
                    if frame >= self.stim_frames:
                        break
                    img = cv2.imread(os.path.join(vid_path, img_path))
                    try:
                        img = torch.from_numpy(img)
                    except Exception:
                        print("hi")
                    if len(img.shape) == 2:
                        img = img[:, :, None]
                    if img.shape[2] == 1:
                        img = torch.tile(img, (1, 1, 3))
                    img = img.reshape((3, self.exp_image_size[0], self.exp_image_size[1]))
                    stimuli[cond_idx, frame, idx, :, :, :] = img
        stim_mean = torch.mean(stimuli.flatten())
        stim_std = torch.std(stimuli.flatten())
        stimuli -= stim_mean
        stimuli /= stim_std
        return stimuli, condition_names


if __name__=='__main__':
    make_downseampled_stim("/Users/loggiasr/Projects/fmri/monkey_fmri/MTurk1/stimuli/dyloc_stimuli",
                           "/Users/loggiasr/Projects/fmri/monkey_fmri/MTurk1/stimuli/dyloc_downsampled")