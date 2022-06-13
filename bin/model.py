import matplotlib.pyplot as plt
import numpy as np
from typing import List

import torch
import math
import os
import random
from torch.utils.data import DataLoader
import networkx as nx

from bin import analysis, representation_geometry
from dataloaders.base import PsychDataloader


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
    dist_from_root = float(dist_from_root)
    root_spatial_dim = float(root_spatial_dim)
    while allowed_dim > max_possible_dim:
        allowed_dim = max(root_spatial_dim * 2 ** (-1 * dist_from_root), 1)
        dist_from_root += 1
    channel_dim = math.ceil(num / (allowed_dim ** 2))
    return 1, int(channel_dim), int(allowed_dim), int(allowed_dim)


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
    kernel = min(6, in_spatial)
    while round(pad) != pad or pad >= kernel:
        # compute padding that will maintain spatial dims during actual conv
        kernel = max(kernel - 1, 1)
        pad = (((in_spatial - 1) * stride) - in_spatial + kernel) / 2

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


def exp_decay(cur_x, start_y, stop_y, stop_x):
    a = start_y
    b = -1 * (stop_y / start_y) ** (1 / stop_x) + 1
    y = a * (1 - b) ** cur_x
    return y


class BrainMimic:

    def __init__(self, structure: nx.Graph, input_node,
                units_per_voxel=10, stimuli_shape=(1, 3, 64, 64)):
        self.brain = nx.DiGraph()
        self.rdm_loss_fxn = torch.nn.MSELoss()
        self.stim_shape = stimuli_shape
        self.structure = structure
        self.head_node = input_node  # index of first node in data stream (e.g. corresponds to roi v1 for vision)
        self.params = []

        # we want to find highest magnitude path with fewest connections
        max_weight = max([np.abs(data[2]['weight']) for data in structure.edges(data=True)])
        for s, t, data in structure.edges(data=True):
            structure.edges[s, t]['weight'] = max_weight - np.abs(data['weight'])

        spatial_bins = int(np.log2(self.stim_shape[2]))
        path_lengths = nx.single_source_dijkstra_path_length(structure, self.head_node)
        nodes = np.array(list(path_lengths.keys()))
        nodes = sorted(nodes)
        path_lengths = np.array([path_lengths[n] for n in nodes])
        bin_edges = np.histogram_bin_edges(path_lengths)
        binned = np.digitize(path_lengths, bins=bin_edges)
        binned[binned > spatial_bins] = spatial_bins
        for n in nodes:
            if n == self.head_node:
                spatial_bin = 0
            else:
                spatial_bin = binned[n]
            data = self.structure.nodes[n]
            desired_neurons = data['num_dims'] * units_per_voxel
            roi_shape = _compute_roi_dim(desired_neurons, spatial_bin, stimuli_shape[2])
            self.brain.add_node(n,
                                roi_name=data['roi_name'],
                                shape=roi_shape,
                                neurons=None,
                                rdm=[],
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
        self.brain.add_node(-1, roi_name='stimulus', neurons=None, shape=stimuli_shape)
        _, u_channels, u_spatial, _ = stimuli_shape
        _, v_channels, v_spatial, _ = self.brain.nodes[self.head_node]['shape']
        initial_sequence = _compute_convolutional_sequence(u_spatial, v_spatial, u_channels, v_channels)
        self.params.append(initial_sequence[0].weight)  # append the convolution
        self.brain.add_edge(-1, self.head_node,
                            sequence=initial_sequence)

    def weight_regularization(self, verbose=True):
        """
        Want all individual weights to be go to zero, and low variance within a given edge
        :return:
        """
        l1_cost = torch.Tensor([0.0])
        var_cost = torch.Tensor([0.0])
        for param in self.params:
            flat_param = param.flatten()
            l1 = torch.sum(torch.abs(flat_param))
            var = torch.std(flat_param)
            l1_cost = l1_cost + l1
            var_cost = var_cost + var * len(flat_param)
        l1_cost = l1_cost / len(self.params)
        var_cost = var_cost / len(self.params)
        if verbose:
            print("L1 cost:", l1_cost.detach().item())
            print("Var Cost: ", var_cost.detach().item())
        cost = l1_cost + var_cost
        return cost

    def prune_graph(self, prune_factor=.05, verbose=False):
        """
        Deletes low ranked node edges.
        :param prune_factor:
        :return:
        """
        print("Pruning...")
        total_weights = torch.stack([torch.mean(torch.abs(w.flatten())) for w in self.params], dim=0)
        torch.sort(total_weights)
        cutoff_idx = math.ceil(len(total_weights) * prune_factor)
        cutoff = total_weights[cutoff_idx]
        for s, t, data in self.brain.edges(data=True):
            link_weight = torch.mean(torch.abs(data['sequence'][0].weight.flatten()))
            if link_weight < cutoff:
                self.brain.remove_edge(s, t)
                if verbose:
                    print("Pruned edge ", self.brain.nodes[s]['roi_name'], '->', self.brain.nodes[t]['roi_name'],
                          "with weight", link_weight.detach().item())

    def initialize_optimizer(self, lr):
        """
        update currently existing param set and create an optimizer object.
        :param lr: the learning rate to use.
        :return:
        """
        self.params = []
        for u, v, data in self.brain.edges(data=True):
            self.params.append(data['sequence'][0].weight)
        optimizer = torch.optim.SGD(lr=lr, params=self.params)
        return optimizer

    @torch.utils.hooks.unserializable_hook
    def fit_rdms(self, stim_gen: List[PsychDataloader], epochs=1000, stimulus_frames = 20, batch_size=10,
                 start_lr=.00001, final_lr=.00000001, prune_start=.1, prune_stop=.01, verbose=True, snapshot_out='./', prune=True):
        """
        Attempt to find  structure for the brain network that matches observed geometry.

        :param brain_data: a nx.Graph that has all nodes in self.brain. Each Node should have an attribute 'rdm' which
                           is a (k, k) dissimilarity matrix where k is the number of conditions in stimuli
        :param stim_gen: a list of PyschDataloader child objects
        """
        # optimize over all incoming edges with source node thats been computed.
        loss_history = []

        for node in self.brain.nodes():
            self.brain.nodes[node]['rdm'] = [None for _ in stim_gen]

        for epoch in range(epochs):
            lr = exp_decay(epoch, start_lr, final_lr, epochs)
            optimizer = self.initialize_optimizer(lr)

            for para_idx, paradigm in enumerate(stim_gen):
                print("********************")
                print("PRESENTING paradigm", paradigm)
                print("********************")
                # generate batch of this paradigm
                stimuli, condition_names = paradigm.get_batch(batch_size)

                cond_presentation_order = torch.randperm(len(stimuli))
                run_list = []

                optimizer.zero_grad()
                activation_states = {n: [] for n in self.brain.nodes if n != -1}

                for cond in cond_presentation_order:
                    stim_cond = stimuli[cond]
                    batch_size = stim_cond.shape[1]
                    actual_stim_frames = stim_cond.shape[0]
                    assert len(stim_cond.shape) == 5

                    # set batch size
                    # allocating memory in network for stimulus shapes
                    for node, data in self.brain.nodes(data=True):
                        in_shape = list(data['shape'])
                        in_shape[0] = batch_size
                        self.brain.nodes[node]['neurons'] = torch.ones(in_shape) * -.1

                    for i in range(stimulus_frames):
                        if verbose:
                            print("\nPRESENTING cond", condition_names[cond.item()], "frame", i)
                        stim = stim_cond[i % actual_stim_frames]  # cycle over stimframes if not enough in dataset
                        # set internal state of stimulus input pseudo-node to equal this stimulus batch
                        self.brain.nodes[-1]['neurons'] = stim
                        run_list.append(cond)
                        nodes = list(self.brain.nodes())
                        random.shuffle(nodes)
                        # find inputs (i.e. the set of edges from predecessor nodes marked as computed)

                        for node in nodes:
                            if verbose:
                                print("computing inputs to node", self.brain.nodes[node]['roi_name'], " on epoch", epoch)
                            if node == -1:
                                # if initial stimulus node continue
                                continue
                            activation_states[node].append(torch.zeros_like(self.brain.nodes[node]['neurons']))
                            for pred in self.brain.predecessors(node):
                                state = self.brain.nodes[pred]['neurons']
                                # assert len(state.shape) == 4
                                connection_sequence = self.brain.edges[(pred, node)]['sequence']
                                for step in connection_sequence:
                                    state = step(state.clone())
                                activation_states[node][-1] = activation_states[node][-1] + state
                        for node in nodes:
                            if node == -1:
                                continue
                            self.brain.nodes[node]['neurons'] = activation_states[node][-1]
                            activation_states[node][-1] = activation_states[node][-1].reshape(batch_size, -1)

                # this design matrix holds for all parallel stimuli in batch this epoch
                design_matrix = torch.from_numpy(analysis.design_matrix_from_run_list(run_list,
                                                                                      len(cond_presentation_order),
                                                                                      base_condition_idxs=[])).float()# t x k
                design_matrix = design_matrix[:, :-1]
                loss = torch.Tensor([0.])

                # compute beta matrix from network activity, compute rdms on matrix, compare to brain rdms
                for node in activation_states.keys():
                    time_course = torch.stack(activation_states[node], dim=1)  # batch x t x n
                    betas = torch.transpose(torch.inverse(design_matrix.T @ design_matrix) @ design_matrix.T @ time_course, 1, 2) # batch x n x k
                    betas = torch.mean(betas, dim=0)  # average out batch dimmension
                    rdm = representation_geometry.dissimilarity(betas, metric='dot')
                    # rdm.register_hook(lambda x: print(node, 'gradient:', x))

                    self.brain.nodes[node]['rdm'][para_idx] = rdm.detach().clone()
                    target_brain_rdm = torch.Tensor(self.structure.nodes[node]['rdm'][para_idx])
                    local_loss = self.rdm_loss_fxn(rdm, target_brain_rdm)
                    loss = loss + local_loss
                if verbose:
                    print("computed beta coefficients")
                # finalize loss term
                reg_weight = .05
                reg = self.weight_regularization(verbose=verbose)
                reg_loss = loss + reg_weight * reg
                # save states
                nx.write_gpickle(self.brain, os.path.join(snapshot_out, 'brain_mimic_epoch_' + str(epoch)))
                reg_loss.backward()
                optimizer.step()
                loss_history.append(reg_loss.detach().item())
                if verbose:
                    print("completed", str(paradigm), "optimization subroutine: LOSS = ", loss.detach().item(), "REGULARIZATION = ", reg_weight * reg.detach().item())

            # remove unimportant node edges
            prune_factor = exp_decay(epoch, prune_start, prune_stop, epochs)
            self.prune_graph(prune_factor, verbose=verbose)
        plt.plot(np.arange(epochs), np.array(loss_history))
        return loss_history

    def fit_task(self):
        # A method that should optimize our brain to actually preform the task
        raise NotImplementedError

    def load_network_state(self, path):
        self.brain = nx.read_gpickle(path)
