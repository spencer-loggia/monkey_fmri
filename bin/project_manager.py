import json
import os
import input_control
from process_control_flow import DefaultSubjectControlNet


def _create_dir_if_needed(base: str, name: str):
    out_dir = os.path.join(base, name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return out_dir


class Project:

    def __init__(self, base_dir, project_name):
        project_root = _create_dir_if_needed(base_dir, project_name)
        _create_dir_if_needed(project_root, 'templates')
        _create_dir_if_needed(project_root, 'analysis')
        _create_dir_if_needed(project_root, 'subjects')
        os.chdir(project_root)
        self.abs_base = project_root
        self.project_config = {}
        if not self.load_project_config():
            self.project_config['project_name'] = project_name
            self.project_config['paradigms'] = {}
            self.project_config['subjects'] = []
            self.project_config['data_map'] = {}  # paradigms -> subjects ->  session_ids / run nums
            self.save_project_config()

    def add_subject_interactive(self):
        print("Subject selection")
        choice = input_control.select_option_input(self.project_config['subjects'] + ['Add new subject...'])
        if choice == len(self.project_config['subjects']):
            name = input("Enter subject name: ")
            self.project_config['subjects'].append(name)
            for para in self.project_config['data_map']:
                self.project_config['data_map'][para][name] = {}
        else:
            name = self.project_config['subjects'][choice]
        subject_dir = _create_dir_if_needed(os.path.join(self.abs_base, 'subjects'), name)
        self.save_project_config()
        _create_dir_if_needed(subject_dir, 'mri')
        _create_dir_if_needed(subject_dir, 'analysis')
        _create_dir_if_needed(subject_dir, 'surf')
        _create_dir_if_needed(subject_dir, 'sessions')
        os.environ.setdefault(key='FMRI_WORK_DIR', value=subject_dir)
        subject_pipe = DefaultSubjectControlNet(name)
        subj_config_path = os.path.join(subject_dir, 'subject_net.json')
        if os.path.exists(subj_config_path):
            subject_pipe.load_net(subj_config_path)
        subject_pipe.control_loop(subj_config_path)
        os.chdir(self.abs_base)

    def _sync_paradigms_to_data_map(self):
        """
        Utility function to assist with the v3 addition of data map extra project state tracker
        :return:
        """
        for paradigm in self.project_config['paradigms']:
            if paradigm not in self.project_config['data_map']:
                self.project_config['data_map'][paradigm] = {subj: {}
                                                             for subj in self.project_config['subjects']}

    def save_project_config(self):
        out_path = os.path.join(self.abs_base, 'config.json')
        with open(out_path, 'w') as f:
            json.dump(self.project_config, f, indent=4)
        return out_path

    def load_project_config(self):
        in_path = os.path.join(self.abs_base, 'config.json')
        if os.path.exists(in_path):
            with open(in_path, 'r') as f:
                self.project_config = json.load(f)
            return True
        else:
            print("no existing project configuration. Creating fresh... ")
            return False


if __name__=='__main__':
    print("****************WELCOME TO fMRI PROJECT MANAGER*********************")
    print("Creating new / loading existing project...")
    existing_proj = input_control.bool_input("Load existing project? (otherwise create new)")
    if existing_proj:
        path = input_control.dir_input("Enter path to existing project root... ")
        base_dir = os.path.dirname(path)
        proj_name = os.path.basename(path)
    else:
        base_dir = input_control.dir_input("Enter directory to create project in... ")
        proj_name = input("Enter name for project... ")
    proj = Project(os.path.abspath(base_dir), proj_name)
    proj.add_subject_interactive()
