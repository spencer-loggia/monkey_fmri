import os

import json
import datetime
import networkx as nx
import preprocess
import support_functions
from input_control import select_option_input, bool_input


class BaseControlNet:

    def __init__(self):
        self.network = nx.DiGraph()
        self.head = []

    def init_head_states(self):
        self.head = []
        data_node, process_node = nx.bipartite.sets(self.network)
        for n in process_node:
            pred = list(self.network.predecessors(n))
            if len(pred) == 0 or False not in [self.network.nodes[d]['complete'] for d in pred]:
                self.head.append(n)

    def control_loop(self, path):
        while self.interactive_advance():
            self.serialize(path)
        return path

    def interactive_advance(self):
        print("action selection")
        options = [str(n) + '   modified: ' + self.network.nodes[n]['modified']
                   if 'modified' in self.network.nodes[n] else str(n) + '   modified: unknown' for n in self.head]
        choice = select_option_input(options + ['back'])
        if choice == len(self.head):
            return False
        action_name = self.head.pop(choice)
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
            if type(res) is tuple:
                res_idx = self.network.edges[(action_name, s)]['order']
                self.network.nodes[s]['data'] = res[res_idx]
            else:
                self.network.nodes[s]['data'] = res
            if 'complete' in self.network.nodes[s]:
                self.network.nodes[s]['complete'] = True
            self.head += [n for n in self.network.successors(s) if False not in
                          [self.network.nodes[j]['complete'] for j in self.network.predecessors(n)]] # add newly available states if their dependencies are met
        return True

    def serialize(self, out_path):
        node_link_dict = nx.readwrite.node_link_data(self.network)
        with open(out_path, 'w') as f:
            json.dump(node_link_dict, f, indent=4)
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
        cur_nodes = self.network.nodes()
        for n, data in loaded_net.nodes(data=True):
            if n not in ignore:
                if 'bipartite' in data and data['bipartite'] == 1 and n in cur_nodes:
                    data['fxn'] = self.network.nodes[n]['fxn']
                self.network.add_node(n, **data)
        self.init_head_states()


class DefaultSubjectControlNet(BaseControlNet):

    def __init__(self, subject_name):
        super().__init__()
        self.network.graph['subject_name'] = subject_name
        self.initialize_processing_structure()
        self.init_head_states()

    def create_load_session(self, ds_t1, ds_t1_mask, ds_t1_masked, sessions):
        """
        local method
        only creates a new session for now
        :param ds_t1:
        :param ds_t1_mask:
        :param ds_t1_masked:
        :return: path to session json
        """
        session_id = input("enter date of session")
        subj_root = os.environ.get('FMRI_WORK_DIR')
        path = os.path.join(subj_root, 'sessions', session_id, 'session_net.json')
        session = DefaultSessionControlNet(session_id, self.network.nodes['ds_t1'],
                                                       self.network.nodes['ds_t1_mask'],
                                                       self.network.nodes['ds_t1_masked'])
        if path in sessions or os.path.exists(path):
            print("loading previous session " + session_id)
            session.load_net(path, ignore=('ds_t1', 'ds_t1_mask', 'ds_t1_masked'))
        if 'is_mion' not in session.network.graph or session.network.graph['is_mion'] is None:
            mion = bool_input("Is this session using MION?")
            session.network.graph['is_mion'] = mion
            session.network.nodes['create_contrast']['argv'] = mion
        return list(set(sessions + [session.control_loop(path)]))

    def initialize_processing_structure(self):
        """
        Should eventualy be replaced with a full anatomical pipeline.
        :return:
        """
        self.network.add_node('t1', data=None, type='volume', bipartite=0, complete=False, space='t1_native')
        self.network.add_node('t1_mask', data=None, type='volume', bipartite=0, complete=False, space='t1_native')

        self.network.add_node('ds_t1', data=None, type='volume', bipartite=0, complete=False, space='ds_t1_native')
        self.network.add_node('ds_t1_mask', data=None, type='volume', bipartite=0, complete=False, space='ds_t1_native')
        self.network.add_node('ds_t1_masked', data=None, type='volume', bipartite=0, complete=False, space='ds_t1_native')

        self.network.add_node('white_surfs', data=[], type='surface', bipartite=0, complete=False, space='t1_native')

        self.network.add_node('sessions', data=[], type='net_json', complete=None,  bipartite=0)

        self.network.add_node('subject_surface_overlays', data=[], complete=None, type='overlay', bipartite=0)

        self.network.add_node('load_t1_data', fxn='support_functions.load_t1_data', bipartite=1)
        self.network.add_node('load_white_surfaces', fxn='support_functions.load_white_surfs', bipartite=1)
        self.network.add_node('downsample_t1', fxn='support_functions.downsample_anatomical', bipartite=1)
        self.network.add_node('downsample_t1_mask', fxn='support_functions.downsample_anatomical', bipartite=1)
        self.network.add_node('mask_downsampled_t1', fxn='support_functions.apply_binary_mask_vol', bipartite=1)
        self.network.add_node('create_load_session', fxn='self.create_load_session', bipartite=1)
        self.network.add_node('generate_sigsurface_overlays', fxn='support_functions.generate_subject_overlays', bipartite=1)


        self.network.add_edge('load_t1_data', 't1', order=0) #10
        self.network.add_edge('load_t1_data', 't1_mask', order=1) #10

        self.network.add_edge('load_white_surfaces', 'white_surfs', order=0) #10

        self.network.add_edge('t1', 'downsample_t1', order=0) #01
        self.network.add_edge('downsample_t1', 'ds_t1', order=0) #10

        self.network.add_edge('t1_mask', 'downsample_t1_mask', order=0) #01
        self.network.add_edge('downsample_t1_mask', 'ds_t1_mask', order=0) #10

        self.network.add_edge('ds_t1', 'mask_downsampled_t1', order=0)
        self.network.add_edge('ds_t1_mask', 'mask_downsampled_t1', order=1)
        self.network.add_edge('mask_downsampled_t1', 'ds_t1_masked', order=0)

        self.network.add_edge('ds_t1', 'create_load_session', order=0) #01
        self.network.add_edge('ds_t1_mask', 'create_load_session', order=1) #01
        self.network.add_edge('ds_t1_masked', 'create_load_session', order=2)
        self.network.add_edge('sessions', 'create_load_session', order=3)
        self.network.add_edge('create_load_session', 'sessions', order=0) #10

        self.network.add_edge('sessions', 'generate_sigsurface_overlays', order=0)
        self.network.add_edge('white_surfs', 'generate_sigsurface_overlays', order=1)
        self.network.add_edge('t1', 'generate_sigsurface_overlays', order=2)
        self.network.add_edge('ds_t1', 'generate_sigsurface_overlays', order=3)
        self.network.add_edge('generate_sigsurface_overlays', 'subject_surface_overlays', order=0)


class DefaultSessionControlNet(BaseControlNet):
    """
    In essensce, a wrapper for a wrapper for a directed bipartite networkx (nx) graph that defines the relationship
    between input or intermediate data files (bipartite=0) and data processing nodes (bipartite=1) the node attribute
    "bipartite" is reserved by nx for defining graphs with more than one distict nodes sets without internel edges,
    and is used by algorithms that operate on such datastructures. This data structure describes how a session of fmri
    data must be processed to get different desired products
    """
    def __init__(self, session_id, ds_t1_path, ds_t1_mask_path, ds_t1_masked_path):
        super().__init__()
        self.initialize_proccessing_structure(session_id, ds_t1_path, ds_t1_mask_path, ds_t1_masked_path)
        self.init_head_states()

    def initialize_proccessing_structure(self, session_id, ds_t1_node, ds_t1_mask_node, ds_t1_masked_node):
        # runtime defined. If path attribute is a list, indicates these are multiple files to process in parallel
        self.network.graph['session_id'] = session_id
        self.network.graph['is_mion'] = None

        # Initial preproccessing nodes
        self.network.add_node('raw_epi', data=[], type='time_series', bipartite=0, complete=False, space='epi_native')
        self.network.add_node('sphinx_epi', data=[], type='time_series', bipartite=0, complete=False,
                              space='epi_native')
        self.network.add_node('moco_epi', data=[], type='time_series', bipartite=0, complete=False, space='epi_native')

        self.network.add_node('3d_epi_rep', data=None, type='volume', bipartite=0, complete=False, space='epi_native')
        self.network.add_node('3d_epi_rep_sphinx', data=None, type='volume', bipartite=0, complete=False, space='epi_native')

        self.network.add_node('ds_t1', **ds_t1_node)
        self.network.add_node('ds_t1_mask', **ds_t1_mask_node)
        self.network.add_node('ds_t1_masked', **ds_t1_masked_node)

        self.network.add_node('manual_reg_epi_rep', data=None, type='volume', bipartite=0, complete=False,
                              space='ds_t1_aprox')
        self.network.add_node('masked_manual_reg_epi_rep', data=None, type='volume', bipartite=0, complete=False,
                              space='ds_t1_aprox')
        self.network.add_node('manual_transform', data=None, type='transform', bipartite=0, complete=False,
                              invertable=True, space=('epi_native', 'ds_t1_aprox'))
        self.network.add_node('auto_composite_transform', data=None, type='transform', bipartite=0, complete=False,
                              invertable=False, space=('ds_t1_aprox', 'ds_t1_native'))
        self.network.add_node('inverse_auto_composite_transform', data=None, type='transform', bipartite=0,
                              complete=False, invertable=False, space=('ds_t1_native', 'ds_t1_aprox'))

        self.network.add_node('epi_mask', data=None, type='volume', bipartite=0, complete=False, space='epi_native')
        self.network.add_node('epi_masked', data=[], fname=None, type='time_series', bipartite=0, complete=False,
                              space='epi_native')

        self.network.add_node('paradigm', data=None, type='json', bipartite=0, complete=False)
        self.network.add_node('ima_order_map', type='json', bipartite=0, complete=False)

        # index in path is id of contrast
        self.network.add_node('contrasts', data=[], type='volume', bipartite=0, complete=False,
                              space='epi_native')

        self.network.add_node('reg_contrast', data=[], type='volume', bipartite=0, complete=False,
                              space='ds_t1_native')

        self.network.add_node('reg_3d_epi_rep', data=None, type='volume', bipartite=0, complete=False, space='ds_t1_approx')

        self.network.add_node('slice_contrast_img', data=None, type='std_image', bipartite=0, complete=False, space='')

        # Define functional data processing nodes

        self.network.add_node('get_epi', argv=session_id, fxn='support_functions.get_epis', bipartite=1)
        self.network.add_node('sphinx_correct', fxn='preprocess.convert_to_sphinx', bipartite=1)
        self.network.add_node('motion_correction', fxn='preprocess.motion_correction', bipartite=1)
        self.network.add_node('select_3d_rep', fxn='support_functions.get_3d_rep', bipartite=1)
        self.network.add_node('sphinx_correct_3d_rep', fxn='support_functions.convert_to_sphinx_vol_wrap', bipartite=1)
        self.network.add_node('manual_registration', fxn='support_functions.itk_manual', bipartite=1)
        self.network.add_node('automatic_coregistration', fxn='support_functions.coreg_wrapper', bipartite=1)
        self.network.add_node('create_functional_mask', fxn='support_functions.apply_warp_inverse', bipartite=1)
        self.network.add_node('apply_reg_epi_mask', fxn='support_functions.apply_binary_mask_vol', biprtite=1)
        self.network.add_node('apply_functional_mask', fxn='support_functions.apply_binary_mask_functional', bipartite=1)
        self.network.add_node('create_load_paradigm', fxn='support_functions.create_load_paradigm', bipartite=1)
        self.network.add_node('create_load_ima_order_map', fxn='support_functions.create_load_ima_order_map', bipartite=1)
        self.network.add_node('create_contrast', fxn='support_functions.create_contrast',
                              bipartite=1, argv=self.network.graph['is_mion'])
        self.network.add_node('register_contrast', fxn='support_functions.apply_warp', bipartite=1)
        self.network.add_node('add_to_subject_average_contrast', argv=session_id,
                              fxn='support_functions.add_to_subject_contrast', bipartite=1)
        self.network.add_node('register_3d_rep', fxn='support_functions.apply_warp', bipartite=1)
        self.network.add_node('contrast_slice_overlay', fxn='support_functions.create_slice_overlays', bipartite=1)

        # define edges (parameter / return values)

        self.network.add_edge('get_epi', 'raw_epi', order=0) #10

        self.network.add_edge('raw_epi', 'sphinx_correct', order=0)  # 01
        self.network.add_edge('sphinx_correct', 'sphinx_epi', order=0)  # 10

        self.network.add_edge('raw_epi', 'select_3d_rep', order=0)  # 01
        self.network.add_edge('select_3d_rep', '3d_epi_rep', order=0)  # 10

        self.network.add_edge('3d_epi_rep', 'sphinx_correct_3d_rep', order=0)
        self.network.add_edge('sphinx_correct_3d_rep', '3d_epi_rep_sphinx', order=0)

        self.network.add_edge('sphinx_epi', 'motion_correction', order=0)  # 01
        self.network.add_edge('3d_epi_rep_sphinx', 'motion_correction', order=1)  # 01
        self.network.add_edge('motion_correction', 'moco_epi', order=0)  # 10

        self.network.add_edge('3d_epi_rep_sphinx', 'manual_registration', order=0)  # 01
        self.network.add_edge('ds_t1', 'manual_registration', order=1)  # 01
        self.network.add_edge('manual_registration', 'manual_transform', order=0)  # 10
        self.network.add_edge('manual_registration', 'manual_reg_epi_rep', order=1)  # 10

        self.network.add_edge('manual_reg_epi_rep', 'apply_reg_epi_mask', order=0)  # 01
        self.network.add_edge('ds_t1_mask', 'apply_reg_epi_mask', order=1)  # 01
        self.network.add_edge('apply_reg_epi_mask', 'masked_manual_reg_epi_rep', order=0)  # 10

        self.network.add_edge('masked_manual_reg_epi_rep', 'automatic_coregistration', order=0)  # 01
        self.network.add_edge('ds_t1_masked', 'automatic_coregistration', order=1)  # 01
        self.network.add_edge('automatic_coregistration', 'auto_composite_transform', order=0)  # 10
        self.network.add_edge('automatic_coregistration', 'inverse_auto_composite_transform', order=1)  # 10

        self.network.add_edge('ds_t1_mask', 'create_functional_mask', order=0)  # 01
        self.network.add_edge('3d_epi_rep_sphinx', 'create_functional_mask', order=1)
        self.network.add_edge('manual_transform', 'create_functional_mask', order=2)  # 01
        self.network.add_edge('inverse_auto_composite_transform', 'create_functional_mask', order=3)  # 01
        self.network.add_edge('create_functional_mask', 'epi_mask', order=0)  # 10

        self.network.add_edge('moco_epi', 'apply_functional_mask', order=0)  # 01
        self.network.add_edge('epi_mask', 'apply_functional_mask', order=1)  # 01
        self.network.add_edge('apply_functional_mask', 'epi_masked', order=0)  # 10

        self.network.add_edge('create_load_paradigm', 'paradigm', order=0)  # 10

        self.network.add_edge('epi_masked', 'create_load_ima_order_map', order=0)  # 01
        self.network.add_edge('create_load_ima_order_map', 'ima_order_map', order=0)  # 10

        self.network.add_edge('epi_masked', 'create_contrast', order=0)  # 01
        self.network.add_edge('paradigm', 'create_contrast', order=1)  # 01
        self.network.add_edge('ima_order_map', 'create_contrast', order=2)  # 01
        self.network.add_edge('create_contrast', 'contrasts', order=0)  # 10

        self.network.add_edge('contrasts', 'register_contrast', order=0)  # 01
        self.network.add_edge('ds_t1_masked', 'register_contrast', order=1)
        self.network.add_edge('manual_transform', 'register_contrast', order=2)  # 01
        self.network.add_edge('auto_composite_transform', 'register_contrast', order=3)  # 01
        self.network.add_edge('register_contrast', 'reg_contrast', order=0)  # 10

        self.network.add_edge('reg_contrast', 'add_to_subject_average_contrast', order=0) # 01
        self.network.add_edge('paradigm', 'add_to_subject_average_contrast', order=1) # 01

        self.network.add_edge('3d_epi_rep_sphinx', 'register_3d_rep', order=0)
        self.network.add_edge('ds_t1_masked', 'register_3d_rep', order=1)
        self.network.add_edge('manual_transform', 'register_3d_rep', order=2)
        self.network.add_edge('auto_composite_transform', 'register_3d_rep', order=3)
        self.network.add_edge('register_3d_rep', 'reg_3d_epi_rep', order=0)

        self.network.add_edge('reg_3d_epi_rep', 'contrast_slice_overlay', order=0)
        self.network.add_edge('ds_t1_masked', 'contrast_slice_overlay', order=1)
        self.network.add_edge('reg_contrast', 'contrast_slice_overlay', order=2)
        self.network.add_edge('contrast_slice_overlay', 'slice_contrast_img', order=0)

        connected = list(nx.connected_components(self.network.to_undirected()))
        if len(connected) > 1:
            print("Error: graph is diconnected. Print non-main components")
            for comp in connected[1:]:
                print(comp)



