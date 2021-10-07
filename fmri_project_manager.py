import os
import shutil
import datetime


class ProjectManager:
    """
    A class to manage a fmri project for you!
    Keeps track of files, tracks changes and any manipulations, ensures compatibility between different libraries, and
    provides actual useful error messages

    structure:

    root
        tmp : temporary intermediate files
        raw_functional : raw functional data
        preprocessed_functional : preprocessed functional data
            scan_directories_ : yyyy-mm-dd
        anatomical :
            orig.mgz (given)
            low_res.nii (generated)
        surfaces :
            ?h.white
            ?h.inflated
            ?h.pial
            ?h.grey
            ?h.thickness
        contrasts:

    """

    def __init__(self):
        self.subject_id = None  # str
        self.scan_session_directories = []  # List[str]
        self.full_resolution_anatomical = None  # bool
        self.config_file = None  # path
        self.surface_files = {'white': None,
                              'pial': None,
                              'inflated': None,
                              'thickness': None,
                              'grey': None}
        self.project_root = ""  # str
        self.head = None  # str
        self.log = []  # List[str]
        self.contrasts = []

    def new_from_freesurfer(self, path, root_path):
        """
        Imports anatomical data from the default freesurfer file structure, gathers anatomical and surfaces
        :return:
        """
        shutil.copy(os.path.join(path, 'mri', 'orig.mgz'), )


