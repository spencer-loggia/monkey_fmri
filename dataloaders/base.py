from typing import Tuple, List
from torch import Tensor


class PsychDataloader:
    def __init__(self, data_folder, exp_image_size=(64, 64), stim_frames=30):
        """
        data exp
        :param data_folder:
        """
        self.data_folder = data_folder
        self.exp_image_size = exp_image_size
        self.stim_frames = stim_frames

    def get_batch(self, batch_size) -> Tuple[Tensor, List[str]]:
        """
        gets a new batch from the stimuli set.
        :return: tuple[Tensor<conditions, time(frames), batch, channels, spatial1, spatial2>,
                                    List[condition_names]
        """
        pass

