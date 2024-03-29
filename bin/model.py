import matplotlib.pyplot as plt

import numpy as np
from typing import List

import torch
import math
import os
import random
from torch.utils.data import DataLoader
import networkx as nx

import analysis, representation_geometry
from dataloaders.base import PsychDataloader

from neurotools.models import ReverbNetwork

plt.ion()


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


def _compute_convolutional_sequence(in_spatial, out_spatial, in_channels, out_channels, mean=0., std=.01):
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
                           out_channels=int(out_channels), bias=False)
    weights = torch.nn.Parameter(torch.normal(mean, std, size=conv.weight.shape))
    conv.weight = weights
    activation = torch.nn.Tanh()
    return conv, rescale, activation


def exp_decay(cur_x, start_y, stop_y, stop_x):
    a = start_y
    b = -1 * (stop_y / start_y) ** (1 / stop_x) + 1
    y = a * (1 - b) ** cur_x
    return y


class BrainMimic:

    def __init__(self, structure: nx.Graph, input_node,
                 units_per_voxel=10, stimuli_shape=(1, 3, 64, 64), start_lr=.0001):
        structure = nx.to_directed(structure)
        # add intra-node connectivity
        for n in structure.nodes:
            structure.add_edge(n, n)
        self.brain = ReverbNetwork(structure, node_shape=stimuli_shape, input_node=input_node)
        self.rdm_loss_fxn = torch.nn.MSELoss()
        self.stim_shape = stimuli_shape
        self.structure = structure

    def prune_graph(self, prune_factor=.05, min_edges=30, verbose=True):
        """
        TODO: FIX
        Deletes low ranked node edges.
        :param prune_factor:
        :return:
        """
        print("Pruning, factor=", prune_factor)
        print("current number edges:", len(self.brain.edges()))
        if len(self.brain.edges()) <= min_edges:
            return
        total_weights = torch.stack([torch.mean(torch.abs(res[2]['sequence'][0].weight))
                                     for res in self.brain.edges(data=True)], dim=0)
        total_weights, _ = torch.sort(total_weights)
        mean_weight = torch.mean(total_weights).detach().item()
        print("average weight :", mean_weight)
        cutoff_idx = math.ceil(len(total_weights) * prune_factor)
        cutoff = total_weights[cutoff_idx]
        for s, t, data in list(self.brain.edges(data=True)):
            if s == -1:
                # cant disconnect exp
                continue
            link_weight = torch.mean(torch.abs(data['sequence'][0].weight).flatten())
            if abs(link_weight) <= cutoff:
                self.brain.remove_edge(s, t)
                if verbose:
                    print("Pruned edge ", self.brain.nodes[s]['roi_name'], '->', self.brain.nodes[t]['roi_name'],
                          "with weight", link_weight.detach().item())
        # remove any edges that become disconnected from stimulus
        reachable = nx.descendants(self.brain, self.head_node) | {self.head_node,
                                                                  -1}  # all nodes reachable from head, plus head
        # itself and stimuli pseudo-node
        for node in list(self.brain.nodes()):
            if node not in reachable:
                if verbose:
                    print("Removed disconnected node ", self.brain.nodes[node]['roi_name'])
                self.brain.remove_node(node)

    def beta_dissimilarity_loss(self, activation_states, run_list, num_conditions, paradigm_index, verbose=True):
        """

        :param activation_states: Dictionary keyed on nodes with list of state tensors of same length as run list. All
                                  node keys must exist in `self.structure`.
        :param run_list: order in which conditions were presented to network (num_frames x 1)
        :param num_conditions:
        :param paradigm_index: index of this paradigm in graph rdm list attribute
        :return: loss scalar, the total dissimilarity between representations at each node in `self.brain` with
                 representations at corresponding nodes in `self.structure`.
        """
        # this design matrix holds for all parallel stimuli in batch this epoch
        design_matrix = torch.from_numpy(analysis.design_matrix_from_run_list(run_list,
                                                                              num_conditions,
                                                                              base_condition_idxs=[])).float()  # t x k
        design_matrix = design_matrix[:, :-1]
        loss = torch.Tensor([0.])

        # compute beta matrix from network activity, compute rdms on matrix, compare to brain rdms
        for node in activation_states.keys():
            time_course = torch.stack(activation_states[node], dim=1)  # batch x t x n
            betas = torch.transpose(torch.inverse(design_matrix.T @ design_matrix) @ design_matrix.T @ time_course, 1,
                                    2)  # batch x n x k
            betas = torch.mean(betas, dim=0)  # average out batch dimmension
            rdm = representation_geometry.dissimilarity(betas, metric='dot')
            # rdm.register_hook(lambda x: print(node, 'gradient:', x))

            self.brain.nodes[node]['rdm'][paradigm_index] = rdm.detach().clone()
            target_brain_rdm = torch.Tensor(self.structure.nodes[node]['rdm'][paradigm_index])
            local_loss = self.rdm_loss_fxn(rdm, target_brain_rdm)
            loss = loss + local_loss
        loss = loss / len(self.brain.nodes)
        if verbose:
            print("computed beta coefficients")
        return loss

    def linear_decode_classification_loss(self, decode_nodes: List[int], target_classes):
        """
        We should be able to linearly decode some target from the decode nodes.
        :param decode_nodes: the node to attempt to decode stimuli from.
        :param target_classes: tensor of length runlist, with associated targets for each frame.
        :return:
        """
        raise NotImplementedError

    def fit_rdms(self, stim_gen: List[PsychDataloader], super_epochs=10, epochs=30, stimulus_frames=20, batch_size=10,
                 start_lr=.00001, final_lr=.00000001, prune_start=.1, prune_stop=.01, verbose=True, snapshot_out='./',
                 reg_weight=.2):
        """
        Attempt to find  structure for the brain network that matches observed geometry.

        :param brain_data: a nx.Graph that has all nodes in self.brain. Each Node should have an attribute 'rdm' which
                           is a (k, k) dissimilarity matrix where k is the number of conditions in stimuli
        :param stim_gen: a list of PyschDataloader child objects
        """
        # optimize over all incoming edges with source node thats been computed.

        for node in self.brain.nodes():
            self.brain.nodes[node]['rdm'] = [None for _ in stim_gen]

        super_loss_history = []
        for super_epoch in range(super_epochs):
            print("\n********************")
            print("META: SUPER Epoch #", super_epoch)
            print("********************")
            loss_history = []

            for epoch in range(epochs):
                loss_history.append(0.)
                self.optimizer.zero_grad()
                epoch_loss = torch.Tensor([0.])
                print("\n********************")
                print("META: LOCAL Epoch # ", epoch, "(auper epoch", super_epoch, ")")
                print("********************")
                for para_idx, paradigm in enumerate(stim_gen):
                    print("\n********************")
                    print("PRESENTING paradigm", paradigm)
                    print("********************")
                    # generate batch of this paradigm
                    stimuli, condition_names = paradigm.get_batch(batch_size)

                    cond_presentation_order = torch.randperm(len(stimuli))
                    run_list = []

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
                            self.brain.nodes[node]['neurons'] = torch.ones(in_shape) * self.default_state

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
                                if node == -1:
                                    # if initial stimulus node continue
                                    continue
                                node_state = self.compute_node_update(node)
                                activation_states[node].append(node_state)
                            for node in nodes:
                                if node == -1:
                                    continue
                                self.brain.nodes[node]['neurons'] = activation_states[node][-1]
                                activation_states[node][-1] = activation_states[node][-1].reshape(batch_size, -1)

                    paradigm_dissimilarity_loss = self.beta_dissimilarity_loss(activation_states, run_list,
                                                                               len(cond_presentation_order), para_idx,
                                                                               verbose=verbose)
                    epoch_loss = epoch_loss + paradigm_dissimilarity_loss  # this could cause numerical instability ...

                    # save states
                    nx.write_gpickle(self.brain, os.path.join(snapshot_out, 'brain_mimic_epoch_' + str(epoch)))
                    if verbose:
                        print("completed", str(paradigm), "LOSS = ", paradigm_dissimilarity_loss.detach().item())

                # finalize loss term
                reg = self.weight_regularization(verbose=verbose)
                reg_loss = epoch_loss + reg_weight * reg
                epoch_loss = epoch_loss + reg_loss
                epoch_loss.backward()
                self.optimizer.step()
                loss_history[-1] += epoch_loss.detach().item()
                if verbose:
                    print("COMPLETED super epoch", super_epoch, "epoch", epoch,
                          "optimization subroutine: TOTAL LOSS = ", loss_history[-1], "REGULARIZATION = ",
                          reg_weight * reg.detach().item())
            loss_history = np.array(loss_history)
            super_loss_history.append(loss_history)
            plt.plot(loss_history)
            plt.title("Super Epoch #" + str(super_epoch) + " epoch loss history")
            plt.show(block=False)
            plt.pause(.001)
            # remove unimportant node edges
            prune_factor = exp_decay(super_epoch, prune_start, prune_stop, super_epochs)
            self.prune_graph(prune_factor, verbose=verbose)
        super_mean_loss = np.array([np.mean(loss) for loss in super_loss_history])
        plt.plot(super_mean_loss)
        plt.title("RDM Fit Super Epoch Loss History")
        plt.show()
        return super_loss_history

    def fit_task(self):
        # A method that should optimize our brain to actually preform the task
        raise NotImplementedError

    def load_network_state(self, path):
        self.brain = nx.read_gpickle(path)
