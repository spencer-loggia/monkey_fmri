import os

import json
import datetime
import networkx as nx
import time

import preprocess
import support_functions
import analysis
from input_control import select_option_input, bool_input


class BaseControlNet:

    def __init__(self):
        self.network = nx.DiGraph()
        self.head = []

    def init_head_states(self):
        self.head = []
        connected = list(nx.connected_components(self.network.to_undirected()))
        if len(connected) > 1:
            print("Error: graph is diconnected. Print non-main components")
            for comp in connected:
                print(comp)
        data_node, process_node = nx.bipartite.sets(self.network)
        for n in process_node:
            pred = list(self.network.predecessors(n))
            good_head = False
            if len(pred) == 0:
                good_head = True
            else:
                for d in pred:
                    if 'complete' not in self.network.nodes[d]:
                        raise ValueError(d + " : " + str(self.network.nodes[d]))
                    if self.network.nodes[d]['complete'] or self.network.nodes[d]['complete'] is None:
                        # TODO: Needs fixin'
                        # if 'modified' in self.network.nodes[n] and 'modified' in self.network.nodes[d] and self.network.nodes[d]['complete'] is not None:
                        #     if datetime.datetime.fromisoformat(self.network.nodes[d]['modified']) > \
                        #        datetime.datetime.fromisoformat(self.network.nodes[n]['modified']) \
                        #             and 'always_show' not in self.network.nodes[n]:
                        #         good_head = False
                        #         break
                        good_head = True
                    else:
                        good_head = False
                        break
            if good_head:
                self.head.append(n)

    def control_loop(self, path):
        while self.interactive_advance():
            self.serialize(path)
        return path

    def display_node_info(self, node_id):
        node_data = self.network.nodes[node_id]
        print('*' * 30)
        print("INFO FOR", node_id)
        print("Embedded Function:", node_data['fxn'])
        print("Dependencies:", list(self.network.predecessors(node_id)))
        print("Children:", list(self.network.successors(node_id)))
        if 'desc' in node_data:
            print("Description:\n", node_data['desc'])
        print("Enjoy your day :) ")
        print('*' * 30)
        bool_input("Done? ")

    def interactive_advance(self):
        print("action selection")
        self.head = sorted(self.head)
        options = [str(n) + '   modified: ' + self.network.nodes[n]['modified']
                   if 'modified' in self.network.nodes[n] else str(n) + '   modified: unknown' for n in self.head]
        choice = select_option_input(options + ['back'])
        need_help = False
        if type(choice) is tuple:
            need_help = choice[1]
            choice = choice[0]
        if choice == len(self.head):
            return False
        action_name = self.head[choice]
        if need_help:
            self.display_node_info(action_name)
            return self.interactive_advance()
        self.network.nodes[action_name]['modified'] = str(datetime.datetime.now())
        pred = list(self.network.predecessors(action_name))
        fxn = eval(self.network.nodes[action_name]['fxn'])
        pred = sorted(pred, key=lambda x: self.network.edges[(x, action_name)]['order'])
        data_params = [self.network.nodes[p]['data'] for p in pred]
        if 'argv' in self.network.nodes[action_name]:
            res = fxn(*data_params, self.network.nodes[action_name]['argv'])
        else:
            res = fxn(*data_params)
        bfs_tree = nx.traversal.bfs_tree(self.network, action_name)
        to_remove = set()
        for node, data in bfs_tree.nodes(data=True):
            if 'complete' in data and data['complete'] is True:
                self.network[node]['complete'] = False
                to_remove.add(node)
        self.head = list(set(self.head) - to_remove)
        suc = list(self.network.successors(action_name))
        for s in suc:
            res_idx = self.network.edges[(action_name, s)]['order']
            if res_idx is not None:
                if type(res) is tuple:
                    self.network.nodes[s]['data'] = res[res_idx]
                else:
                    self.network.nodes[s]['data'] = res
            self.network.nodes[s]['modified'] = str(datetime.datetime.now())
            if 'complete' in self.network.nodes[s] and self.network.nodes[s]['complete'] is not None:
                self.network.nodes[s]['complete'] = True
            try:
                self.head += [n for n in self.network.successors(s) if False not in
                              [self.network.nodes[j]['complete'] for j in
                               self.network.predecessors(n)]]  # add newly available states if their dependencies are met
            except KeyError:
                print(self.network.successors(s))
                exit(-1)
        return True

    def serialize(self, out_path):
        bkp = None
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                bkp = json.load(f)
        try:
            node_link_dict = nx.readwrite.node_link_data(self.network)
            with open(out_path, 'w') as f:
                json.dump(node_link_dict, f, indent=4)
        except Exception:
            print("FAILED TO SAVE CHANGES TO PROJECT STATE. RESTORED TO LAST STABLE STATE.")
            if bkp is None:
                print("No saved stable state. Data lost.")
            else:
                with open(out_path, 'w') as f:
                    json.dump(bkp, f, indent=4)
        return out_path

    def load_net(self, in_path, ignore=tuple()):
        """

        :param in_path: path to json node link data
        :param ignore: nodes to not update data attributes from file
        :return:
        """
        with open(in_path, 'r') as f:
            node_link_dict = json.load(f)
        loaded_net = nx.readwrite.node_link_graph(node_link_dict, directed=True, multigraph=False)
        for g_att in loaded_net.graph.keys():
            self.network.graph[g_att] = loaded_net.graph[g_att]
        for n, data in loaded_net.nodes(data=True):
            self.network.add_node(n, **data)
        for s, t, data in loaded_net.edges(data=True):
            self.network.add_edge(s, t, **data)
        self.init_head_states()


class DefaultSubjectControlNet(BaseControlNet):

    def __init__(self, subject_name):
        super().__init__()
        self.network.graph['subject_name'] = subject_name
        self.initialize_processing_structure()
        self.init_head_states()

    def create_load_session(self, ds_t1, ds_t1_mask, ds_t1_masked, dil_t1_mask, sessions):
        """
        local method
        only creates a new session for now
        :param ds_t1:
        :param ds_t1_mask:
        :param ds_t1_masked:
        :param dil_t1_mask:
        :return: path to session json
        """
        session_id = input("enter date of session")
        subj_root = os.environ.get('FMRI_WORK_DIR')
        path = os.path.join(subj_root, 'sessions', session_id, 'session_net.json')
        session = DefaultSessionControlNet(session_id, self.network.nodes['functional_representative'],
                                           self.network.nodes['functional_representative_mask'],
                                           self.network.nodes['functional_representative_masked'],
                                           self.network.nodes['functional_representative_dil_mask'])
        if path in sessions or os.path.exists(path):
            print("loading previous session " + session_id)
            session.load_net(path, ignore=('ds_t1', 'ds_t1_mask', 'ds_t1_masked', 'dil_t1_mask'))
        if 'is_mion' not in session.network.graph or session.network.graph['is_mion'] is None:
            mion = bool_input("Is this session using MION?")
            session.network.graph['is_mion'] = mion
        if 'scan_pos' not in session.network.graph or session.network.graph['scan_pos'] is None:
            is_hfp = bool_input("Is the scan acquasition position Head-First-Prone (y) or Head-First-Suppine (n)? ")
            if is_hfp:
                session.network.graph['scan_pos'] = 'HFP'
            else:
                session.network.graph['scan_pos'] = 'HFS'
        session.network.nodes['create_beta_matrix']['argv'] = session.network.graph['is_mion']
        session.network.nodes['sphinx_correct']['argv'] = session.network.graph['scan_pos']
        session.network.nodes['sphinx_correct_3d_rep']['argv'] = session.network.graph['scan_pos']
        return list(set(sessions + [os.path.relpath(session.control_loop(path), subj_root)]))

    def add_paradigm_control_set(self):
        para_def_dict, para_path = support_functions._create_paradigm()

        para_name = para_def_dict['name']
        subj_root = os.environ.get('FMRI_WORK_DIR')
        self.network.add_node('paradigm_' + para_name, data=None, complete=True, bipartite=0, type='json')
        self.network.add_node('beta_' + para_name, data=None, complete=False, bipartite=0, type='4d_volume',
                              space='ds_t1_native')
        self.network.add_node('registered_betas_' + para_name, data=None, complete=False, bipartite=0, type='4d_volume')
        self.network.add_node('contrasts_' + para_name, data=[], complete=False, bipartite=0, type='volume',
                              space='ds_t1_native')
        self.network.add_node('sigsurfaces_' + para_name, data=[], complete=False, bipartite=0, type='overlay',
                              space='t1_native')

        self.network.add_node('create_' + para_name + '_betas', bipartite=1,
                              fxn='support_functions.construct_complete_beta_file',
                              desc="Gets the beta coefficient matrix files from sessions that use the " + para_name +
                                   " paradigm and have been promoted to subject level, then combines them "
                                   "(weighted by nuber of runs) intp a single subject level beta coefficient matrix, "
                                   "which becomes the basis for many different subject level analysis. The beta matrix "
                                   "is designed to capture the information we care about from this experiment, and "
                                   "satisfies the equation (DesignMatrix * hrf) @ BetaMatrix ~= EpiTimeSeries "
                                   "where * is the convolution operator and @ is matrix multiplication.")

        self.network.add_node('register_' + para_name + '_betas', bipartite=1,
                              fxn='support_functions.apply_warp_4d',
                              desc="take a complete beta matrix from all promoted sessions of this paradigm from the"
                                   " standard functional space to the anatomical space")

        self.network.add_node('create_' + para_name + '_contrasts', bipartite=1,
                              fxn='support_functions.create_contrast',
                              desc="Generates the contrasts requested in the paradigm definition using the subjet level "
                                   "beta matrix. Contrasts are a useful tool for comparing activation to two "
                                   "conditions or linear combinations of conditions on a per voxel level. Contrast "
                                   "values are interpreted as Z-scores here. ")
        self.network.add_node('create_' + para_name + '_sigsurface_overlays', bipartite=1,
                              fxn='support_functions.generate_subject_overlays')

        self.network.add_node('automatic_' + para_name + '_volume_rois', bipartite=1,
                              fxn='support_functions.automatic_volume_rois')
        self.network.add_node('delete_paradigm_' + para_name, bipartite=1, fxn='self.remove_paradigm_control_set')

        self.network.add_edge('add_new_paradigm', 'paradigm_' + para_name, order=0)  # 10

        self.network.add_edge('paradigm_' + para_name, 'create_' + para_name + '_betas', order=0)  # 01
        self.network.add_edge('create_' + para_name + '_betas', 'beta_' + para_name, order=0)  # 10

        self.network.add_edge('paradigm_' + para_name, 'delete_paradigm_' + para_name, order=0)  # 01

        self.network.add_edge('beta_' + para_name, 'register_' + para_name + '_betas', order=0)
        self.network.add_edge('ds_t1_masked', 'register_' + para_name + '_betas', order=1)
        self.network.add_edge('manual_transform', 'register_' + para_name + '_betas', order=2)
        self.network.add_edge('auto_composite_transform', 'register_' + para_name + '_betas', order=3)
        self.network.add_edge('register_' + para_name + '_betas', 'registered_betas_' + para_name, order=0)

        self.network.add_edge('registered_betas_' + para_name, 'create_' + para_name + '_contrasts', order=0)  # 01
        self.network.add_edge('paradigm_' + para_name, 'create_' + para_name + '_contrasts', order=1)  # 01
        self.network.add_edge('create_' + para_name + '_contrasts', 'contrasts_' + para_name, order=0)  # 10

        self.network.add_edge('paradigm_' + para_name, 'create_' + para_name + '_sigsurface_overlays', order=0)
        self.network.add_edge('contrasts_' + para_name, 'create_' + para_name + '_sigsurface_overlays', order=1)
        self.network.add_edge('white_surfs', 'create_' + para_name + '_sigsurface_overlays', order=2)
        self.network.add_edge('t1', 'create_' + para_name + '_sigsurface_overlays', order=3)
        self.network.add_edge('ds_t1', 'create_' + para_name + '_sigsurface_overlays', order=4)
        self.network.add_edge('create_' + para_name + '_sigsurface_overlays', 'sigsurfaces_' + para_name, order=0)
        self.network.add_edge('create_' + para_name + '_sigsurface_overlays', 'paradigm_complete_checkpoint', order=1)

        self.network.add_edge('paradigm_' + para_name, 'automatic_' + para_name + '_volume_rois', order=0)
        self.network.add_edge('contrasts_' + para_name, 'automatic_' + para_name + '_volume_rois', order=1)
        self.network.add_edge('ds_volume_rois', 'automatic_' + para_name + '_volume_rois', order=2)
        self.network.add_edge('automatic_' + para_name + '_volume_rois', 'ds_volume_rois', order=0)

        self.serialize(os.path.join(subj_root, 'subject_net.json'))
        return para_path

    def remove_paradigm_control_set(self, paradigm_path):
        para_name, remove = support_functions.delete_paradigm(paradigm_path)
        if remove:
            self.network.remove_node('paradigm_' + para_name)
            self.network.remove_node('beta_' + para_name)
            self.network.remove_node('contrasts_' + para_name)
            self.network.remove_node('sigsurfaces_' + para_name)
            self.network.remove_node('delete_paradigm_' + para_name)

            self.network.remove_node('create_' + para_name + '_betas')
            self.network.remove_node('create_' + para_name + '_contrasts')
            self.network.remove_node('create_' + para_name + '_sigsurface_overlays')

            self.network.remove_node('automatic_' + para_name + '_volume_rois')

    def initialize_processing_structure(self):
        """
        Should eventualy be replaced with a full anatomical pipeline.
        :return:
        """
        self.network.add_node('t1', data=None, type='volume', bipartite=0, complete=False, space='t1_native', )
        self.network.add_node('t1_mask', data=None, type='volume', bipartite=0, complete=False, space='t1_native')

        self.network.add_node('ds_t1', data=None, type='volume', bipartite=0, complete=False, space='ds_t1_native')
        self.network.add_node('ds_t1_mask', data=None, type='volume', bipartite=0, complete=False, space='ds_t1_native')
        self.network.add_node('ds_t1_masked', data=None, type='volume', bipartite=0, complete=False,
                              space='ds_t1_native')
        self.network.add_node('dil_ds_t1_mask', data=None, type='volume', bipartite=0,complete=False, space='ds_t1_native')

        self.network.add_node('white_surfs', data=[], type='surface', bipartite=0, complete=False, space='t1_native')

        self.network.add_node('sessions', data=[], type='net_json', complete=None, bipartite=0, always_show=True)

        self.network.add_node('functional_representative', data=None, type='volume', bipartite=0, complete=False, space='epi_std')
        self.network.add_node('functional_representative_mask', data=None, type='volume', bipartite=0, complete=False, space='epi_std')
        self.network.add_node('functional_representative_masked', data=None, type='volume', bipartite=0, complete=False, space='epi_std')
        self.network.add_node('functional_representative_dil_mask', data=None, type='volume', bipartite=0, complete=False, space='epi_std')

        self.network.add_node('manual_reg_epi_rep', data=None, type='volume', bipartite=0, complete=False,
                              space='ds_t1_aprox')
        self.network.add_node('masked_manual_reg_epi_rep', data=None, type='volume', bipartite=0, complete=False,
                              space='ds_t1_aprox')
        self.network.add_node('manual_transform', data=None, type='transform', bipartite=0, complete=False,
                              invertable=True, space=('epi_std', 'ds_t1_aprox'))
        self.network.add_node('auto_composite_transform', data=None, type='transform', bipartite=0, complete=False,
                              invertable=False, space=('ds_t1_aprox', 'ds_t1_native'))
        self.network.add_node('inverse_auto_composite_transform', data=None, type='transform', bipartite=0,
                              complete=False, invertable=False, space=('ds_t1_native', 'ds_t1_aprox'))

        self.network.add_node('paradigm_complete_checkpoint', data=None, complete=None, bipartite=0)

        # three below data is expected to be dict('roi_set_name' : tuple(condition_of_interest_intID, data_path))
        self.network.add_node('surface_defined_rois', data={}, complete=None, type='label', bipartite=0)
        self.network.add_node('volume_rois', data={}, complete=None, type='volume', space='t1_native', bipartite=0)
        self.network.add_node('ds_volume_rois', data={}, complete=None, type='volume', space='ds_t1_native',
                              bipartite=0)
        self.network.add_node('roi_time_courses', data={}, complete=None, type='array', bipartite=0)

        self.network.add_node('load_t1_data', fxn='support_functions.load_t1_data', bipartite=1,
                              desc='Accept user input for paths to t1 and t1 mask anatomical volumes. Used for all subject sessions')
        self.network.add_node('load_white_surfaces', fxn='support_functions.load_white_surfs', bipartite=1,
                              desc='Accept user input for paths to white matter and cortical thickness surfaces')
        self.network.add_node('downsample_t1', fxn='support_functions.downsample_anatomical', bipartite=1,
                              desc='Downsample orginal t1 by a factor of 2 to the working space.')
        self.network.add_node('downsample_t1_mask', fxn='support_functions.downsample_anatomical', bipartite=1,
                              desc='Downsample orginal t1 mask by a factor of 2 to the working space.')
        self.network.add_node('mask_downsampled_t1', fxn='support_functions.apply_binary_mask_vol', bipartite=1,
                              desc='Apply downsampled mask to downsampled t1.')
        self.network.add_node('create_masked_functional_rep', fxn="support_functions.apply_binary_mask_vol", bipartite=1,
                              desc="apply a registered mask to the functional image")

        self.network.add_node('create_load_session', fxn='self.create_load_session', bipartite=1,
                              desc='create preprocessing and basic analysis pipeline for new fmri session.')

        self.network.add_node("get_functional_representative", fxn="support_functions.get_functional_target",
                              bipartite=1, desc='get a masked functional image all sessions will preregister to')

        self.network.add_node('manual_registration', fxn='support_functions.itk_manual', bipartite=1,
                              desc='Use ITK-Snap to preform a course grain manual registration between the epi std rep volume and '
                                   'the downsampled t1, creating a crude linear mapping between the functional and working space.'
                                   ' It is most important that the borders of the cortex closely match, since this map is used for '
                                   ' brain masking the epi rep.')
        self.network.add_node('apply_reg_epi_mask', fxn='support_functions.apply_binary_mask_vol', bipartite=1)

        self.network.add_node('automatic_coregistration', fxn='support_functions.coreg_wrapper', bipartite=1,
                              desc='Use ants coregistration to create a fine grained nonlinear mapping from the skull '
                                   'stripped manually aligned std epi rep to the downsampled t1. The manual registration got'
                                   ' the brains as overlapping as possible, now this attempts to morph the cortex to correct'
                                   ' differences in white matter boundaries due to field warp in epi acquisition.')

        self.network.add_node('create_functional_representative_mask', fxn='support_functions.apply_warp_inverse',
                              bipartite=1, desc="invert the t1 mask to create a masked functional image")

        self.network.add_node('create_dilated_functional_representative_mask', fxn='support_functions.apply_warp_inverse',
                              bipartite=1, desc="invert the t1 mask to create a masked functional image")

        self.network.add_node('dilate_ds_t1_mask', fxn='support_functions.dilate_mask', bipartite=1, desc='dilate the downsampled t1 mask')

        # self.network.add_node('mask_functional_rep', fxn='support_functions.apply_binary_mask_vol',
        #                       desc="Apply the dilated mask to the functional (will probobly need to be corrected manually)")

        self.network.add_node('add_new_paradigm', fxn='self.add_paradigm_control_set', bipartite=1)

        self.network.add_node('manual_surface_rois', fxn='support_functions.define_surface_rois', bipartite=1,
                              desc='Just a placeholder to tell the project manager that you created manual rois, and to'
                                   ' provide instructions. You will need to draw and save the rois in freeview.')
        self.network.add_node('volume_rois_from_surface_rois', fxn='support_functions.surf_labels_to_vol_mask',
                              bipartite=1,
                              desc='project ROIs defined on the surface back into the volume.')
        self.network.add_node('get_vol_rois_time_series', fxn='support_functions.get_vol_rois_time_series', bipartite=1,
                              desc='Combine time series created with different stimuli orderings via block of interest stacking method, then extract the average time course for each volume roi.')
        self.network.add_node('downsample_vol_rois', fxn='support_functions.downsample_vol_rois', bipartite=1,
                              desc='downsample volume rois by a factor of 2 to the working space.')
        self.network.add_node('manual_volume_rois', fxn='support_functions.manual_volume_rois', bipartite=1)

        self.network.add_edge('load_t1_data', 't1', order=0)  # 10
        self.network.add_edge('load_t1_data', 't1_mask', order=1)  # 10

        self.network.add_edge('add_new_paradigm', 'sessions', order=None)  # 10
        self.network.add_edge('add_new_paradigm', 'paradigm_complete_checkpoint', order=None)  # 10

        self.network.add_edge('load_white_surfaces', 'white_surfs', order=0)  # 10

        self.network.add_edge('t1', 'downsample_t1', order=0)  # 01
        self.network.add_edge('downsample_t1', 'ds_t1', order=0)  # 10

        self.network.add_edge('t1_mask', 'downsample_t1_mask', order=0)  # 01
        self.network.add_edge('downsample_t1_mask', 'ds_t1_mask', order=0)  # 10

        self.network.add_edge('ds_t1_mask', 'dilate_ds_t1_mask', order=0)
        self.network.add_edge('dilate_ds_t1_mask', 'dil_ds_t1_mask', order=0)

        self.network.add_edge('ds_t1', 'mask_downsampled_t1', order=0)
        self.network.add_edge('ds_t1_mask', 'mask_downsampled_t1', order=1)
        self.network.add_edge('mask_downsampled_t1', 'ds_t1_masked', order=0)

        self.network.add_edge('ds_t1', 'create_load_session', order=0)  # 01
        self.network.add_edge('functional_representative', 'create_load_session', order=1) # 01
        self.network.add_edge('functional_representative_masked', 'create_load_session', order=2)
        self.network.add_edge('functional_representative_mask', 'create_load_session', order=3)
        self.network.add_edge('sessions', 'create_load_session', order=4)
        self.network.add_edge('create_load_session', 'sessions', order=0)  # 10

        self.network.add_edge('paradigm_complete_checkpoint', 'manual_surface_rois', order=0)
        self.network.add_edge('surface_defined_rois', 'manual_surface_rois', order=1)
        self.network.add_edge('manual_surface_rois', 'surface_defined_rois', order=0)
        #
        self.network.add_edge('surface_defined_rois', 'volume_rois_from_surface_rois', order=0)
        self.network.add_edge('white_surfs', 'volume_rois_from_surface_rois', order=1)
        self.network.add_edge('t1', 'volume_rois_from_surface_rois', order=2)
        self.network.add_edge('ds_t1', 'volume_rois_from_surface_rois', order=3)
        self.network.add_edge('volume_rois', 'volume_rois_from_surface_rois', order=4)
        self.network.add_edge('volume_rois_from_surface_rois', 'volume_rois', order=0)

        self.network.add_edge('volume_rois', 'downsample_vol_rois', order=0)
        self.network.add_edge('ds_volume_rois', 'downsample_vol_rois', order=1)
        self.network.add_edge('downsample_vol_rois', 'ds_volume_rois', order=0)

        self.network.add_edge('get_functional_representative', 'functional_representative', order=0)

        self.network.add_edge('functional_representative', 'manual_registration', order=0)  # 01
        self.network.add_edge('ds_t1', 'manual_registration', order=1)  # 01
        self.network.add_edge('manual_registration', 'manual_transform', order=0)  # 10
        self.network.add_edge('manual_registration', 'manual_reg_epi_rep', order=1)  # 10

        self.network.add_edge('manual_reg_epi_rep', 'apply_reg_epi_mask', order=0)  # 01
        self.network.add_edge('dil_ds_t1_mask', 'apply_reg_epi_mask', order=1)  # 01
        self.network.add_edge('apply_reg_epi_mask', 'masked_manual_reg_epi_rep', order=0)  #

        self.network.add_edge('masked_manual_reg_epi_rep', 'automatic_coregistration', order=0)  # 01
        self.network.add_edge('ds_t1_masked', 'automatic_coregistration', order=1)  # 01
        self.network.add_edge('automatic_coregistration', 'auto_composite_transform', order=0)  # 10
        self.network.add_edge('automatic_coregistration', 'inverse_auto_composite_transform', order=1)  # 10

        self.network.add_edge('ds_t1_mask', 'create_functional_representative_mask', order=0)  # 01
        self.network.add_edge('functional_representative', 'create_functional_representative_mask', order=1)
        self.network.add_edge('manual_transform', 'create_functional_representative_mask', order=2)  # 01
        self.network.add_edge('inverse_auto_composite_transform', 'create_functional_representative_mask', order=3)  # 01
        self.network.add_edge('create_functional_representative_mask', 'functional_representative_mask', order=0)  # 10

        self.network.add_edge('dil_ds_t1_mask', 'create_dilated_functional_representative_mask', order=0)  # 01
        self.network.add_edge('functional_representative', 'create_dilated_functional_representative_mask', order=1)
        self.network.add_edge('manual_transform', 'create_dilated_functional_representative_mask', order=2)  # 01
        self.network.add_edge('inverse_auto_composite_transform', 'create_dilated_functional_representative_mask', order=3)  # 01
        self.network.add_edge('create_dilated_functional_representative_mask', 'functional_representative_dil_mask', order=0)  # 10

        self.network.add_edge('functional_representative', 'create_masked_functional_rep', order=0)
        self.network.add_edge('functional_representative_mask', 'create_masked_functional_rep', order=1)
        self.network.add_edge('create_masked_functional_rep', 'functional_representative_masked', order=0)

        self.network.add_edge('ds_volume_rois', 'get_vol_rois_time_series', order=0)
        self.network.add_edge('roi_time_courses', 'get_vol_rois_time_series', order=1)
        self.network.add_edge('ds_t1', 'get_vol_rois_time_series', order=2)
        self.network.add_edge('get_vol_rois_time_series', 'roi_time_courses', order=0)

        self.network.add_edge('ds_t1', 'manual_volume_rois', order=0)
        self.network.add_edge('paradigm_complete_checkpoint', 'manual_volume_rois', order=1)
        self.network.add_edge('ds_volume_rois', 'manual_volume_rois', order=2)
        self.network.add_edge('manual_volume_rois', 'ds_volume_rois', order=0)


class DefaultSessionControlNet(BaseControlNet):
    """
    In essensce, a wrapper for a wrapper for a directed bipartite networkx (nx) graph that defines the relationship
    between input or intermediate data files (bipartite=0) and data processing nodes (bipartite=1) the node attribute
    "bipartite" is reserved by nx for defining graphs with more than one distict nodes sets without internel edges,
    and is used by algorithms that operate on such datastructures. This data structure describes how a session of fmri
    data must be processed to get different desired products
    """

    def __init__(self, session_id, func_rep_node, func_rep_masked_node, func_rep_mask_node, func_rep_dil_mask_node):
        super().__init__()
        self.initialize_proccessing_structure(session_id, func_rep_node, func_rep_masked_node,
                                              func_rep_mask_node, func_rep_dil_mask_node)
        self.init_head_states()

    def initialize_proccessing_structure(self, session_id, func_rep_node, func_rep_mask_node,
                                         function_rep_masked_node, functional_rep_dil_mask_node):
        # runtime defined. If path attribute is a list, indicates these are multiple files to process in parallel
        self.network.graph['session_id'] = session_id
        self.network.graph['is_mion'] = None

        # Initial preproccessing nodes
        self.network.add_node('other_images', data=[], type='volume', bipartite=0, complete=False, space='epi_native')
        self.network.add_node('raw_epi', data=[], type='time_series', bipartite=0, complete=False, space='epi_native')

        self.network.add_node('nordic_epi', data=None, type='time_series', bipartite=0, complete=False,
                              space='epi_native')

        self.network.add_node('sphinx_epi', data=[], type='time_series', bipartite=0, complete=False,
                              space='epi_native')
        self.network.add_node('moco_epi', data=[], type='time_series', bipartite=0, complete=False, space='epi_native')

        self.network.add_node('3d_epi_rep', data=None, type='volume', bipartite=0, complete=False, space='epi_native')
        self.network.add_node('3d_epi_rep_sphinx', data=None, type='volume', bipartite=0, complete=False,
                              space='epi_native')

        self.network.add_node('functional_std', **func_rep_node)
        self.network.add_node('functional_std_dil_mask', **functional_rep_dil_mask_node)
        self.network.add_node('functional_std_mask', **func_rep_mask_node)
        self.network.add_node('functional_std_masked', **function_rep_masked_node)

        self.network.add_node('manual_reg_epi_rep', data=None, type='volume', bipartite=0, complete=False,
                              space='epi_std_aprox')
        self.network.add_node('masked_manual_reg_epi_rep', data=None, type='volume', bipartite=0, complete=False,
                              space='epi_std_aprox')
        self.network.add_node('manual_transform', data=None, type='transform', bipartite=0, complete=False,
                              invertable=True, space=('epi_native', 'epi_std_aprox'))
        self.network.add_node('auto_composite_transform', data=None, type='transform', bipartite=0, complete=False,
                              invertable=False, space=('epi_std_aprox', 'epi_std'))
        self.network.add_node('inverse_auto_composite_transform', data=None, type='transform', bipartite=0,
                              complete=False, invertable=False, space=('epi_std', 'epi_std_aprox'))

        self.network.add_node('epi_mask', data=None, type='volume', bipartite=0, complete=False, space='epi_native')
        self.network.add_node('epi_masked', data=[], fname=None, type='time_series', bipartite=0, complete=False,
                              space='epi_native')

        self.network.add_node('epi_masked_filtered', data=[], type='time_series', bipartite=0, complete=False,
                              space='epi_native')

        self.network.add_node('paradigm', data=None, type='json', bipartite=0, complete=False)
        self.network.add_node('ima_order_map', type='json', bipartite=0, complete=False)

        self.network.add_node('run_beta_matrices', data=[], type='4D_volume', bipartite=0, complete=False,
                              space='epi_native')
        self.network.add_node('reg_beta_matrices', data=None, type='4D_volume', bipartite=0, complete=False,
                              spece='epi_std_aprox')
        self.network.add_node('reg_beta_matrix', data=None, type='4D_volume', bipartite=0, complete=False,
                              spece='epi_std_aprox')

        # index in path is id of contrast
        self.network.add_node('reg_contrasts', data=[], type='volume', bipartite=0, complete=False,
                              space='epi_std')

        self.network.add_node('reg_3d_epi_rep', data=None, type='volume', bipartite=0, complete=False,
                              space='epi_std_aprox')

        self.network.add_node('slice_contrast_img', data=None, type='std_image', bipartite=0, complete=False, space='')

        self.network.add_node('hrf_estimate', data=None, type='ndarray', bipartite=0, complete=False, space='')

        # Define functional data processing nodes
        self.network.add_node('get_images', argv=session_id, fxn='support_functions.get_images', bipartite=1,
                              desc='Accept user input for path to session directory, load images into a dump folder')
        self.network.add_node('get_epi', argv=session_id, fxn='support_functions.get_epis', bipartite=1,
                              desc='Accept user input for path to session directory containing directories of dicoms or'
                                   ' a nifti, and load the data into project format (freesurfer convention)')
        self.network.add_node('noise_correct', fxn='support_functions.nordic_correction_wrapper', bipartite=1,
                              desc="Apply Steen Moeller of UMN 's very nice thermal noise correction matlab script to "
                                   "our raw epis. Need to provide a thermal noise image path, ideally collected in the "
                                   "same session or otherwise a close of one as possible. Must have cloned the "
                                   "SteenMoeller/NORDIC_raw github and added it to matlab path.")
        self.network.add_node('sphinx_correct', fxn='preprocess.convert_to_sphinx', bipartite=1,
                              desc='Preform Sphinx orientation correction on the 4d raw epi data')
        self.network.add_node('motion_correction', fxn='support_functions.motion_correction_wrapper', bipartite=1,
                              desc='Linear motion correction to target 3D rep image. All frames in whole session are '
                                   'affine aligned to this target, using mutual information cost. Uses either FSL MCFLIRT '
                                   'or ANTs motion correction under the hood depending on package availability. FSL is '
                                   'preferred due to faster runtimes with similar results. If multiple targets are provided,'
                                   'returns motion correction that resulted in lowest displacement')
        self.network.add_node('select_3d_rep', fxn='support_functions.get_3d_rep', bipartite=1,
                              desc='Selects a frame from the middle of the run to use as the functional 3d '
                                   'representative volume. This volume defines the functional space. '
                                   'Each session has a unique functional space, though they likely are very similar')
        self.network.add_node('sphinx_correct_3d_rep', fxn='support_functions.convert_to_sphinx_vol_wrap', bipartite=1,
                              desc='Preform sphinx orientaion correction on the representative epi volume.')
        self.network.add_node('manual_registration', fxn='support_functions.itk_manual', bipartite=1,
                              desc='Use ITK-Snap to preform a course grain manual registration between the epi rep volume and '
                                   'the downsampled t1, creating a crude linear mapping between the functional and working space.'
                                   ' It is most important that the borders of the cortex closely match, since this map is used for '
                                   ' brain masking the epi rep.')
        self.network.add_node('automatic_coregistration', fxn='support_functions.coreg_wrapper', bipartite=1,
                              desc='Use ants coregistration to create a fine grained nonlinear mapping from the skull '
                                   'stripped manually aligned epi rep to the downsampled t1. The manual registration got'
                                   ' the brains as overlapping as possible, now this attempts to morph the cortex to correct'
                                   ' differences in white matter boundaries due to field warp in epi acquisition.')
        self.network.add_node('create_functional_mask', fxn='support_functions.apply_warp_inverse', bipartite=1,
                              desc='Uses the manual and nonlinear coregistration in inverse to create a brainmask in '
                                   'the functional space.')
        self.network.add_node('apply_reg_epi_mask', fxn='support_functions.apply_binary_mask_vol', biprtite=1,
                              desc='Applys the downsampled t1 mask to the epi rep thats been manually registered to the '
                                   'ds t1, so that the epi can be nonlinear coregistered to the t1 without intereference from the skull')
        self.network.add_node('apply_functional_mask', fxn='support_functions.apply_binary_mask_functional',
                              bipartite=1,
                              desc='Apply the brainmask in the functional space to all epi time series.')
        self.network.add_node('bandpass_filter', fxn='support_functions.bandpass_wrapper', bipartite=1,
                              desc="Removes high frequency (less than 2 trs) and low frequency "
                                   "(greater than 2.25 blocks) signal")
        self.network.add_node('load_paradigm', fxn='support_functions.create_load_paradigm', bipartite=1,
                              desc='Either load an existing stimuli paradigm json config file, or use the prompt to '
                                   'create a new one.')
        self.network.add_node('create_load_ima_order_map', fxn='support_functions.create_load_ima_order_map',
                              bipartite=1,
                              desc='Map each epi ima number to a corresponding stimuli paradigm order number.')

        self.network.add_node('create_beta_matrix', fxn='support_functions.get_beta_matrix',
                              bipartite=1, argv=self.network.graph['is_mion'],
                              desc='Create the size voxels x stimulus_conditions beta matrix from paradigm def and '
                                   'session masked epis. The beta matrix is designed to capture the information we care'
                                   'about from this experiment, and satisfies the equation (DesignMatrix * hrf) @ BetaMatrix ~= EpiTimeSeries'
                                   ' where * is the convolution operator and @ is matrix multiplication.')

        self.network.add_node('create_session_averaged_beta_matrix', fxn='analysis.create_averaged_beta',
                              bipartite=1, desc='Average all the individual run betas from this session to construct '
                                                'session averaged beta matrix.')

        self.network.add_node('register_beta_matrices', fxn='support_functions.apply_warp_4d', bipartite=1,
                              desc='Use the functional -> working space transform to take the run level beta coefficients to '
                                   'the working space. ')
        self.network.add_node('create_session_contrast', fxn='support_functions.create_contrast', bipartite=1,
                              desc='Create contrast volumes from the beta matrix.')

        self.network.add_node('promote_session', argv=session_id,
                              fxn='support_functions.promote_session', bipartite=1,
                              desc='This session is marked as complete-ish and will be included by future subject level analysis, and for construction of'
                                   ' the subject level beta matrix. Basically its added to the subject config data index.')
        self.network.add_node('demote_session', argv=session_id, fxn='support_functions.demote_session',
                              desc='Removes this session from subject config data index. It will no longer be included '
                                   'in future subject level analysis.')
        self.network.add_node('register_3d_rep', fxn='support_functions.apply_warp', bipartite=1)
        self.network.add_node('contrast_slice_overlay', fxn='support_functions.create_slice_overlays', bipartite=1)

        # define edges (parameter / return values)

        self.network.add_edge('raw_epi', 'get_images', order=0)
        self.network.add_edge('get_images', 'other_images', order=0)

        self.network.add_edge('get_epi', 'raw_epi', order=0)  # 10

        self.network.add_edge('raw_epi', 'noise_correct', order=0)
        self.network.add_edge('noise_correct', 'nordic_epi', order=0)

        self.network.add_edge('nordic_epi', 'sphinx_correct', order=0)  # 01
        self.network.add_edge('sphinx_correct', 'sphinx_epi', order=0)  # 10

        self.network.add_edge('nordic_epi', 'select_3d_rep', order=0)  # 01
        self.network.add_edge('select_3d_rep', '3d_epi_rep', order=0)  # 10

        self.network.add_edge('3d_epi_rep', 'sphinx_correct_3d_rep', order=0)
        self.network.add_edge('sphinx_correct_3d_rep', '3d_epi_rep_sphinx', order=0)

        self.network.add_edge('sphinx_epi', 'motion_correction', order=0)  # 01
        self.network.add_edge('3d_epi_rep_sphinx', 'motion_correction', order=1)  # 01
        self.network.add_edge('motion_correction', 'moco_epi', order=0)  # 10
        self.network.add_edge('motion_correction', '3d_epi_rep_sphinx', order=1)  # 10

        self.network.add_edge('3d_epi_rep_sphinx', 'manual_registration', order=0)  # 01
        self.network.add_edge('functional_std', 'manual_registration', order=1)  # 01
        self.network.add_edge('manual_registration', 'manual_transform', order=0)  # 10
        self.network.add_edge('manual_registration', 'manual_reg_epi_rep', order=1)  # 10

        self.network.add_edge('manual_reg_epi_rep', 'apply_reg_epi_mask', order=0)  # 01
        self.network.add_edge('functional_std_dil_mask', 'apply_reg_epi_mask', order=1)  # 01
        self.network.add_edge('apply_reg_epi_mask', 'masked_manual_reg_epi_rep', order=0)  # 10

        self.network.add_edge('masked_manual_reg_epi_rep', 'automatic_coregistration', order=0)  # 01
        self.network.add_edge('functional_std_masked', 'automatic_coregistration', order=1)  # 01
        self.network.add_edge('automatic_coregistration', 'auto_composite_transform', order=0)  # 10
        self.network.add_edge('automatic_coregistration', 'inverse_auto_composite_transform', order=1)  # 10

        self.network.add_edge('functional_std_mask', 'create_functional_mask', order=0)  # 01
        self.network.add_edge('3d_epi_rep_sphinx', 'create_functional_mask', order=1)
        self.network.add_edge('manual_transform', 'create_functional_mask', order=2)  # 01
        self.network.add_edge('inverse_auto_composite_transform', 'create_functional_mask', order=3)  # 01
        self.network.add_edge('create_functional_mask', 'epi_mask', order=0)  # 10

        self.network.add_edge('moco_epi', 'apply_functional_mask', order=0)  # 01
        self.network.add_edge('epi_mask', 'apply_functional_mask', order=1)  # 01
        self.network.add_edge('apply_functional_mask', 'epi_masked', order=0)  # 10

        self.network.add_edge('load_paradigm', 'paradigm', order=0)  # 10

        self.network.add_edge('epi_masked', 'create_load_ima_order_map', order=0)  # 01
        self.network.add_edge('create_load_ima_order_map', 'ima_order_map', order=0)  # 10

        self.network.add_edge('epi_masked', 'bandpass_filter', order=0)
        self.network.add_edge('paradigm', 'bandpass_filter', order=1)
        self.network.add_edge('bandpass_filter', 'epi_masked_filtered', order=0)

        self.network.add_edge('epi_masked_filtered', 'create_beta_matrix', order=0)  # 01
        self.network.add_edge('paradigm', 'create_beta_matrix', order=1)  # 01
        self.network.add_edge('ima_order_map', 'create_beta_matrix', order=2)  # 01
        self.network.add_edge('create_beta_matrix', 'run_beta_matrices', order=0)  # 10
        self.network.add_edge('create_beta_matrix', 'hrf_estimate', order=1)  # 10

        self.network.add_edge('run_beta_matrices', 'register_beta_matrices', order=0)  # 01
        self.network.add_edge('functional_std_masked', 'register_beta_matrices', order=1)
        self.network.add_edge('manual_transform', 'register_beta_matrices', order=2)  # 01
        self.network.add_edge('auto_composite_transform', 'register_beta_matrices', order=3)  # 01
        self.network.add_edge('register_beta_matrices', 'reg_beta_matrices', order=0)  # 10

        self.network.add_edge('reg_beta_matrices', 'create_session_averaged_beta_matrix', order=0)
        self.network.add_edge('create_session_averaged_beta_matrix', 'reg_beta_matrix', order=0)

        self.network.add_edge('reg_beta_matrix', 'promote_session', order=0)  # 01
        self.network.add_edge('paradigm', 'promote_session', order=1)  # 01
        self.network.add_edge('epi_masked', 'promote_session', order=2)  # 01
        self.network.add_edge('ima_order_map', 'promote_session', order=3)  # 01

        self.network.add_edge('paradigm', 'demote_session', order=0)  # 01

        self.network.add_edge('reg_beta_matrix', 'create_session_contrast', order=0)  # 01
        self.network.add_edge('paradigm', 'create_session_contrast', order=1)  # 01
        self.network.add_edge('create_session_contrast', 'reg_contrasts', order=0)  # 10

        self.network.add_edge('3d_epi_rep_sphinx', 'register_3d_rep', order=0)
        self.network.add_edge('functional_std_masked', 'register_3d_rep', order=1)
        self.network.add_edge('manual_transform', 'register_3d_rep', order=2)
        self.network.add_edge('auto_composite_transform', 'register_3d_rep', order=3)
        self.network.add_edge('register_3d_rep', 'reg_3d_epi_rep', order=0)

        self.network.add_edge('reg_3d_epi_rep', 'contrast_slice_overlay', order=0)
        self.network.add_edge('functional_std_masked', 'contrast_slice_overlay', order=1)
        self.network.add_edge('reg_contrasts', 'contrast_slice_overlay', order=2)
        self.network.add_edge('contrast_slice_overlay', 'slice_contrast_img', order=0)

        self.network.add_edge('reg_3d_epi_rep', 'contrast_slice_overlay', order=0)
        self.network.add_edge('functional_std_masked', 'contrast_slice_overlay', order=1)
        self.network.add_edge('reg_contrasts', 'contrast_slice_overlay', order=2)
        self.network.add_edge('contrast_slice_overlay', 'slice_contrast_img', order=0)

