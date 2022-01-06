import json
from typing import List

import networkx as nx
import analysis
import preprocess
import support_functions
from input_control import select_option_input


class BaseControlNet:

    def __init__(self):
        self.network = nx.DiGraph()
        self.head = []

    def _init_head_states(self):
        data_node, process_node = nx.bipartite.sets(self.head)
        for n in process_node:
            pred = self.network.predecessors(n)
            if len(pred) > 0 and False not in [self.network.nodes[d]['complete'] for d in pred]:
                self.head.append(n)

    def interactive_advance(self):
        print("action selection")
        action_name = self.head.pop(select_option_input(self.head))
        pred = self.network.predecessors(action_name)
        not_complete = [p for p in pred if not p['complete']]
        if len(not_complete) > 0:
            print("dependency(s) " + str(not_complete) + " have not been satisfied")
            return not_complete
        else:
            suc = self.network.successors(action_name)
            fxn = self.network.nodes[action_name]['fxn']
            pred = sorted(pred, key=lambda x: x['order'])
            data_params = [self.network.nodes[p]['data'] for p in pred]
            if 'argv' in self.network.nodes[action_name]:
                res = fxn(*data_params, self.network.nodes[action_name]['argv'])
            else:
                res = fxn(*data_params)
            for s in suc:
                res_idx = s['order']
                self.network.nodes[s]['data'] = res[res_idx]
                if 'complete' in self.network.nodes[s]:
                    self.network.nodes[s]['complete'] = True
                self.head += [n for n in self.network.successors(s) if False not in
                              [self.network.nodes[j]['complete'] for j in self.network.predecessors(n)]] # add newly available states if their dependencies are met

    def serialize(self, out_path):
        node_link_dict = nx.readwrite.node_link_data(self.network)
        with open(out_path, 'w') as f:
            json.dump(node_link_dict, f, indent=4)
        return out_path

    def load_net(self, in_path):
        with open(in_path, 'w') as f:
            node_link_dict = json.load(f)
        self.network = nx.readwrite.node_link_graph(node_link_dict, directed=True, multigraph=False)


class DefaultSubjectControlNet(BaseControlNet):

    def __init__(self, subject_name):
        super().__init__()
        self.network.graph['subject_name'] = subject_name
        self.initialize_processing_structure()

    def initialize_processing_structure(self):
        """
        Should eventualy be replaced with a full anatomical pipeline.
        :return:
        """
        self.network.add_node('t1', data=None, type='volume', bipartite=0, complete=False, space='t1_native')
        self.network.add_node('t1_mask', data=None, type='volume', bipartite=0, complete=False, space='t1_native')
        self.network.add_node('t1_masked', data=None, type='volume', bipartite=0, complete=False, space='t1_native')

        self.network.add_node('ds_t1', data=None, type='volume', bipartite=0, complete=False, space='ds_t1_native')
        self.network.add_node('ds_t1_mask', data=None, type='volume', bipartite=0, complete=False, space='ds_t1_native')
        self.network.add_node('ds_t1_masked', data=None, type='volume', bipartite=0, complete=False, space='ds_t1_native')

        self.network.add_node('white_surfs', data=[], type='surface', bipartite=0, complete=False, space='t1_native')
        self.network.add_node('inflated_surfs', data=[], type='surface', bipartite=0, complete=False, space='t1_native')

        self.network.add_node('sessions', data=[], type='net_json', bipartite=0)

        self.network.add_node('subject_surface_overlays', data=[], type='overlay', bipartite=0)

        self.network.add_node('downsample_t1', fxn=support_functions.downsample_anatomical, bipartite=1)
        self.network.add_node('downsample_t1_mask', fxn=support_functions.downsample_anatomical, bipartite=1)
        self.network.add_node('mask_downsampled_t1', fxn=support_functions.apply_binary_mask_vol, bipartite=1)
        self.network.add_node('create_load_session', fxn=support_functions.create_load_session, bipartite=1)
        self.network.add_node('generate_sigsurface_overlays', fxn=support_functions.generate_subject_overlays, bipartite=1)

        self.network.add_edge('t1', 'downsample_t1', order=0) #01
        self.network.add_edge('downsample_t1', 'ds_t1', order=1) #10

        self.network.add_edge('t1_mask', 'downsample_t1_mask', order=0) #01
        self.network.add_edge('downsample_t1_mask', 'ds_t1_mask', order=0) #10

        self.network.add_edge('ds_t1', 'mask_downsampled_t1', order=0)
        self.network.add_edge('ds_t1_mask', 'mask_downsampled_t1', order=1)
        self.network.add_edge('mask_downsampled_t1', 'ds_t1_masked', order=0)

        self.network.add_edge('t1', 'create_load_session', order=0) #01
        self.network.add_edge('t1_mask', 'create_load_session', order=1) #01
        self.network.add_edge('t1_masked', 'create_load_session', order=2)
        self.network.add_edge('create_load_session', 'sessions', order=0) #10

        self.network.add_edge('sessions', 'generate_sigsurface_overlays', order=0)
        self.network.add_edge('white_surfs', 'generate_sigsurface_overlays', order=1)
        self.network.add_edge('ds_t1', 'generate_sigsurface_overlays', order=2)
        self.network.add_edge('t1', 'generate_sigsurface_overlays', order=2)
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
        self.initialize_proccessing_structure(ds_t1_path, ds_t1_mask_path, ds_t1_masked_path)
        self.network.graph['session_id'] = session_id
        self._init_head_states()

    def initialize_proccessing_structure(self, t1_path, t1_mask_path, ds_t1_masked_path):
        # runtime defined. If path attribute is a list, indicates these are multiple files to process in parallel

        # Initial preproccessing nodes
        self.network.add_node('raw_epi', data=[], type='time_series', bipartite=0, complete=False, space='epi_native')
        self.network.add_node('sphinx_epi', data=[], type='time_series', bipartite=0, complete=False,
                              space='epi_native')
        self.network.add_node('moco_epi', data=[], type='time_series', bipartite=0, complete=False, space='epi_native')

        self.network.add_node('3d_epi_rep', data=None, type='volume', bipartite=0, complete=False, space='epi_native')

        self.network.add_node('ds_t1', data=t1_path, type='volume', bipartite=0, complete=True, space='ds_t1_native')
        self.network.add_node('ds_t1_mask', data=t1_mask_path, type='volume', bipartite=0, complete=True, space='ds_t1_native')
        self.network.add_node('ds_t1_masked', data=ds_t1_masked_path, type='volume', bipartite=0, complete=True,
                              space='ds_t1_native')

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

        self.network.add_node('paradigm', data=None, type='json', bipartite=0)
        self.network.add_node('ima_order_map', type='json', bipartite=0)
        self.network.add_node('contrast_matrix', value=None, desc=[])

        # index in path is id of contrast
        self.network.add_node('contrasts', data=[], type='volume', bipartite=0, complete=False,
                              space='epi_native')

        self.network.add_node('reg_contrast', data=[], type='volume', bipartite=0, complete=False,
                              space='ds_t1_native')

        # Define functional data processing nodes

        self.network.add_node('get_epi', argv=self.network.graph['session_id'], fxn=support_functions.get_epis, bipartite=1)
        self.network.add_node('sphinx_correct', fxn=preprocess.convert_to_sphinx, bipartite=1)
        self.network.add_node('motion_correction', fxn=preprocess.motion_correction, bipartite=1)
        self.network.add_node('select_3d_rep', fxn=preprocess.get_middle_frame, bipartite=1)
        self.network.add_node('manual_registration', fxn=preprocess.manual_itksnap_registration, bipartite=1)
        self.network.add_node('automatic_coregistration', fxn=support_functions.coreg_wrapper, bipartite=1)
        self.network.add_node('create_functional_mask', fxn=support_functions.apply_warp_inverse, bipartite=1)
        self.network.add_node('apply_reg_epi_mask', fxn=support_functions.apply_binary_mask_vol, biprtite=1)
        self.network.add_node('apply_functional_mask', fxn=support_functions.apply_binary_mask_functional, bipartite=1)
        self.network.add_node('create_load_paradigm', fxn=support_functions.create_load_paradigm, bipartite=1)
        self.network.add_node('create_load_ima_order_map', fxn=support_functions.create_load_ima_order_map, bipartite=1)
        self.network.add_node('create_contrast', fxn=support_functions.create_contrast, bipartite=1)
        self.network.add_node('register_contrast', fxn=support_functions.apply_warp, bipartite=1)
        self.network.add_node('add_to_subject_average_contrast', argv=self.network.graph['session_id'], fxn=support_functions.add_to_subject_contrast, bipartite=1)

        # define edges (parameter / return values)

        self.network.add_edge('get_epi', 'raw_epi', order=0) #10

        self.network.add_edge('raw_epi', 'sphinx_correction', order=0)  # 01
        self.network.add_edge('sphinx_correction', 'sphinx_epi', order=0)  # 10

        self.network.add_edge('sphinx_epi', 'select_3d_rep', order=0)  # 01
        self.network.add_edge('select_3d_rep', '3d_epi_rep', order=0)  # 10

        self.network.add_edge('sphinx_epi', 'motion_correction', order=0)  # 01
        self.network.add_edge('3d_epi_rep', 'motion_correction', order=1)  # 01
        self.network.add_edge('motion_correction', 'moco_epi', order=0)  # 10

        self.network.add_edge('3d_epi_rep', 'manual_registration', order=0)  # 01
        self.network.add_edge('ds_t1', 'manual_registration', order=1)  # 01
        self.network.add_edge('manual_registration', 'manual_transform', order=0)  # 10
        self.network.add_edge('manual_registration', 'manual_reg_epi_rep', order=1)  # 10

        self.network.add_edge('manual_reg_epi_rep', 'apply_reg_epi_mask', order=0)  # 01
        self.network.add_edge('apply_reg_epi_mask', 'masked_manual_reg_epi_rep', order=0)  # 10

        self.network.add_edge('masked_manual_reg_epi_rep', 'automatic_coregistration', order=0)  # 01
        self.network.add_edge('ds_t1_masked', 'automatic_coregistration', order=1)  # 01
        self.network.add_edge('automatic_coregistration', 'auto_composite_transform', order=0)  # 10
        self.network.add_edge('automatic_coregistration', 'inverse_auto_composite_transform', order=1)  # 10

        self.network.add_edge('ds_t1_mask', 'create_functional_mask', order=0)  # 01
        self.network.add_edge('3d_epi_rep', 'create_functional_mask', order=1)
        self.network.add_edge('manual_transform', 'create_functional_mask', order=2)  # 01
        self.network.add_edge('inverse_auto_composite_transform', 'create_functional_mask', order=3)  # 01
        self.network.add_edge('create_functional_mask', 'epi_mask', order=0)  # 10

        self.network.add_edge('epi_mask', 'apply_functional_mask', order=0)  # 01
        self.network.add_edge('apply_functional_mask', 'epi_masked', order=0)  # 10

        self.network.add_edge('create_load_paradigm', 'paradigm', order=0)  # 10

        self.network.add_edge('epi_mask', 'create_load_ima_order_map', order=0)  # 01
        self.network.add_edge('paradigm', 'create_load_ima_order_map', order=1)  # 01
        self.network.add_edge('create_load_ima_order_map', 'ima_order_map', order=0)  # 10

        self.network.add_edge('epi_masked', 'create_contrast', order=0)  # 01
        self.network.add_edge('paradigm', 'create_contrast', order=1)  # 01
        self.network.add_edge('ima_order_map', 'create_contrast', order=2)  # 01
        self.network.add_edge('create_contrast', 'contrasts', order=0)  # 10

        self.network.add_edge('contrasts', 'register_contrast', order=0)  # 01
        self.network.add_edge('ds_t1_masked', 'register_contrast', order=1)
        self.network.add_edge('manual_transform', 'register_contrast', order=2)  # 01
        self.network.add_edge('auto_composite_transform', 'register_contrast', order=3)  # 01
        self.network.add_edge('register_contrast', 'reg_contrast', order=0) #10

