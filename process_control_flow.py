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
            if len(pred) > 0 and False not in [self.network[d]['complete'] for d in pred]:
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
            fxn = self.network[action_name]['fxn']
            pred = sorted(pred, key=lambda x: x['order'])
            path_params = [self.network[p]['path'] for p in pred]
            res = fxn(*path_params)
            for s in suc:
                res_idx = s['order']
                self.network[s]['path'] = res[res_idx]
                self.network[s]['complete'] = True
                self.head += [n for n in self.network.successors(s) if False not in
                              [self.network[j]['complete'] for j in self.network.predecessors(n)]] # add newly available states if their dependencies are met


class DefaultSessionControlNet(BaseControlNet):
    """
    In essensce, a wrapper for a wrapper for a directed bipartite networkx (nx) graph that defines the relationship
    between input or intermediate data files (bipartite=0) and data processing nodes (bipartite=1) the node attribute
    "bipartite" is reserved by nx for defining graphs with more than one distict nodes sets without internel edges,
    and is used by algorithms that operate on such datastructures. This data structure describes how a session of fmri
    data must be processed to get different desired products
    """
    def __init__(self, t1_path, t1_mask_path):
        super().__init__()
        self.network = nx.DiGraph()
        self.initialize_session_proccessing_structure(t1_path, t1_mask_path)

    def initialize_session_proccessing_structure(self, t1_path, t1_mask_path):
        # runtime defined. If path attribute is a list, indicates these are multiple files to process in parallel

        # Initial preproccessing nodes
        self.network.add_node('raw_epi', path=[], type='time_series', bipartite=0, complete=False, space='epi_native')
        self.network.add_node('sphinx_epi', path=[], type='time_series', bipartite=0, complete=False,
                              space='epi_native')
        self.network.add_node('moco_epi', path=[], type='time_series', bipartite=0, complete=False, space='epi_native')

        self.network.add_node('3d_epi_rep', path=None, type='volume', bipartite=0, complete=False, space='epi_native')

        self.network.add_node('ds_t1', path=None, type='volume', bipartite=0, complete=True, space='ds_t1_native')
        self.network.add_node('ds_t1_mask', path=None, type='volume', bipartite=0, complete=True, space='ds_t1_native')
        self.network.add_node('ds_t1_masked', path=None, type='volume', bipartite=0, complete=True,
                              space='ds_t1_native')

        self.network.add_node('manual_reg_epi_rep', path=None, type='volume', bipartite=0, complete=False,
                              space='ds_t1_aprox')
        self.network.add_node('masked_manual_reg_epi_rep', path=None, type='volume', bipartite=0, complete=False,
                              space='ds_t1_aprox')
        self.network.add_node('manual_transform', path=None, type='transform', bipartite=0, complete=False,
                              invertable=True, space=('epi_native', 'ds_t1_aprox'))
        self.network.add_node('auto_composite_transform', path=None, type='transform', bipartite=0, complete=False,
                              invertable=False, space=('ds_t1_aprox', 'ds_t1_native'))
        self.network.add_node('inverse_auto_composite_transform', path=None, type='transform', bipartite=0,
                              complete=False, invertable=False, space=('ds_t1_native', 'ds_t1_aprox'))

        self.network.add_node('epi_mask', path=None, type='volume', bipartite=0, complete=False, space='epi_native')
        self.network.add_node('epi_masked', path=[], fname=None, type='time_series', bipartite=0, complete=False,
                              space='epi_native')

        self.network.add_node('paradigm', path=None, type='json', bipartite=0)
        self.network.add_node('ima_order_map', type='json', bipartite=0)
        self.network.add_node('contrast_matrix', value=None, desc=[])

        # index in path is id of contrast
        self.network.add_node('contrasts', path=[], type='volume', bipartite=0, complete=False,
                              space='epi_native')

        self.network.add_node('reg_contrast', path=[], type='volume', bipartite=0, complete=False,
                              space='ds_t1_native')

        # Define functional data processing nodes

        self.network.add_node('get_epi', fxn=support_functions.get_epis, bipartite=1)
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

