import torch
import networkx as nx


class BrainMimic(torch.nn.Module):

    def __int__(self, structure: nx.Graph, atlas: torch.Tensor):
        raise NotImplementedError

