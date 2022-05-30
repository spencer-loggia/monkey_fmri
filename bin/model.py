import numpy as np
from typing import List

import torch
import math
from torch.utils.data import DataLoader
import networkx as nx

from bin import analysis, representation_geometry


def _compute_roi_dim(num, dist_from_root, root_spatial_dim):
    """
    All node spatial dims should differ from input image spatial dims by powers of 2 only. channel dims can take
    arbitrary positive integer values.
    nodes are assigned spatial dimensionality based on number of units and distance in graph from root, and put
    remaining neurons on channel dim
    :param num:
    :return: a tuple defining the dimensionality of a computation roi, (channels, spatial1, spatial2)
    """
    max_possible_dim = math.ceil(num ** .5)
    allowed_dim = math.inf
    while allowed_dim > max_possible_dim:
        allowed_dim = max(root_spatial_dim * 2 ** (-1 * dist_from_root), 1)
        dist_from_root += 1
    channel_dim = math.ceil(num / (allowed_dim ** 2))
    return 1, channel_dim, int(allowed_dim), int(allowed_dim)


def _compute_convolutional_sequence(in_spatial, out_spatial, in_channels, out_channels):
    """
    Return a convolution, spatial rescaling, and hyperbolic tangent sequence that maps the units of one node to the
    units of another.
    :param in_spatial:
    :param out_spatial:
    :param in_channels:
    :param out_channels:
    :return:
    """
    stride = 1
    pad = .1
    kernel = min(4, in_spatial)
    while int(pad) != pad:
        # compute padding that will maintain spatial dims during actual conv
        pad = (((out_spatial - 1) * stride) - in_spatial + kernel) / 2
        kernel = max(kernel - 1, 1)
    # now compute pool size needed to match spatial dims
    if in_spatial >= out_spatial:
        rescale_kernel = int(in_spatial / out_spatial)
        rescale = torch.nn.MaxPool2d(rescale_kernel)
        assert in_spatial / rescale_kernel == out_spatial
    else:
        rescale_kernel = int(out_spatial / in_spatial)
        rescale = torch.nn.Upsample(scale_factor=rescale_kernel, mode='nearest')
        assert in_spatial * rescale_kernel == out_spatial
    conv = torch.nn.Conv2d(kernel_size=int(kernel), padding=int(pad), stride=1, in_channels=int(in_channels),
                           out_channels=int(out_channels))
    activation = torch.nn.Tanh()
    return conv, rescale, activation


class BrainMimic:

    def __init__(self, structure: nx.Graph, input_node,
                units_per_voxel=10, stimuli_shape=(1, 3, 64, 64)):
        self.brain = nx.DiGraph()
        self.rdm_loss_fxn = torch.nn.MSELoss()
        self.structure = structure
        self.head_node = input_node  # index of first node in data stream (e.g. corresponds to roi v1 for vision)
        self.params = []
        path_lengths = nx.single_source_shortest_path_length(structure, self.head_node)
        for n, data in structure.nodes(data=True):
            desired_neurons = data['num_dims'] * units_per_voxel
            roi_shape = _compute_roi_dim(desired_neurons, path_lengths[n], stimuli_shape[2])
            self.brain.add_node(n,
                                roi_name=data['roi_name'],
                                shape=roi_shape,
                                neurons=torch.zeros(roi_shape),
                                rdm=None,
                                computed=False)
            print('added computational node', n, data['roi_name'], 'with size', roi_shape)
        total_edges = len(structure.edges)
        count = 1
        for u, v, data in structure.edges(data=True):
            _, u_channels, u_spatial, _ = self.brain.nodes[u]['shape']  # must be square in spatial dims
            _, v_channels, v_spatial, _ = self.brain.nodes[v]['shape']
            u_v_sequence = _compute_convolutional_sequence(u_spatial, v_spatial, u_channels, v_channels)
            v_u_sequence = _compute_convolutional_sequence(v_spatial, u_spatial, v_channels, u_channels)
            self.params.append(u_v_sequence[0].weight)
            self.params.append(v_u_sequence[0].weight)
            self.brain.add_edge(u, v,
                                sequence=u_v_sequence)
            self.brain.add_edge(v, u,
                                sequence=v_u_sequence)
            print('created edges (', u, ',', v, ') -- ', count, 'of', total_edges)
            count += 1

        # add stimulus input node
        self.brain.add_node(-1, roi_name='stimulus', neurons=torch.zeros(stimuli_shape), shape=stimuli_shape)
        _, u_channels, u_spatial, _ = stimuli_shape
        _, v_channels, v_spatial, _ = self.brain.nodes[self.head_node]['shape']
        initial_sequence = _compute_convolutional_sequence(u_spatial, v_spatial, u_channels, v_channels)
        self.params.append(initial_sequence[0].weight)  # append the convolution
        self.brain.add_edge(-1, self.head_node,
                            sequence=initial_sequence)
        self.optimizer = torch.optim.SGD(lr=.01, params=self.params)

    def fit_rdms(self, stimuli: List[List[torch.Tensor]], epochs=1000):
        """
        Attempt to find  structure for the brain network that matches observed geometry.

        :param brain_data: a nx.Graph that has all nodes in self.brain. Each Node should have an attribute 'rdm' which
                           is a (k, k) dissimilarity matrix where k is the number of conditions in stimuli
        :param stimuli: a 2D list where axis 0 is length k stimuli conditions, and axis 1 is stimuli (as tensors). It's
                        critical that the condition order here matches that in the brain data rdms.
        :return: None
        """
        # optimize over all incoming edges with source node thats been computed.
        for epoch in range(epochs):
            batch_stim = [stim_cond[np.random.randint(0, len(stim_cond))] for stim_cond in stimuli]
            batch_stim = torch.stack(batch_stim, dim=0)
            cond_presentation_order = torch.randperm(len(batch_stim))
            stimulus_frames = 50  # dts to present stimulus
            run_list = []
            # show the network the stimuli from head
            self.optimizer.zero_grad()  # this actually isn't theoretically necessary here :o but it will free up ram.
            activation_states = {n: [] for n in self.brain.nodes if n != -1}
            for cond in cond_presentation_order:
                stim = batch_stim[cond]
                self.brain.nodes[-1]['neurons'][:] = stim
                for i in range(stimulus_frames):
                    run_list.append(cond)
                    # find inputs (i.e. the set of edges from predecessor nodes marked as computed)
                    for node in self.brain.nodes():
                        if node == -1:
                            # if initial stimulus node continue
                            continue
                        for pred in self.brain.predecessors(node):
                            state = self.brain.nodes[pred]['neurons']
                            connection_sequence = self.brain.edges[(pred, node)]['sequence']
                            for step in connection_sequence:
                                state = step(state)
                            try:
                                self.brain.nodes[node]['neurons'] += state
                            except Exception:
                                print('bad')
                        activation_states[node].append(self.brain.nodes['neurons'])
            design_matrix = analysis.design_matrix_from_run_list(run_list, len(cond_presentation_order),
                                                                 base_condition_idxs=[])  # t x k
            loss = torch.Tensor([0.])
            for node in activation_states.keys():
                time_course = torch.stack(activation_states[node].flatten(), dim=0)  # n x t
                betas = torch.inverse(design_matrix.T @ design_matrix) @ design_matrix.T @ time_course
                rdm = representation_geometry.dissimilarity(betas, metric='dot')
                self.brain.nodes[node]['rdm'] = rdm
                target_brain_rdm = self.structure[node]['rdm']
                local_loss = self.rdm_loss_fxn(rdm, target_brain_rdm)
                loss = loss + local_loss
            # finalize loss term
            l1_reg = sum([torch.abs(w).sum() for w in self.params])  # enforce sparsity & keep weights smallish
            loss = loss + .1 * l1_reg
            loss.backward()
            self.optimizer.step()

    def fit_task(self):
        # A method that should optimize our brain to actually preform the task
        raise NotImplementedError
