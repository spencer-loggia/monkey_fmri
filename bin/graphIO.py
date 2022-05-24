import itertools
import json
import sys
from typing import List, Tuple, Union, Dict, Hashable

import networkx as nx
import pandas as pd


class GraphIO:
    """
    Class for IO with networkx `Graph`,  `DiGraph`, `MultiGraph`, or `MultiDiGraph` objects. Provides functions for
    writing to / loading from JSON file, as well as for converting to other graph representations. The JSON file format
    used by this Module is compliant with the node-link json format, and is described in the `file_spec.pdf` file.
    """

    @staticmethod
    def infer_edge_attributes(graph: Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]):
        if type(graph) in [nx.MultiDiGraph, nx.MultiGraph]:
            edge_att_names = set(itertools.chain(*[list(graph.edges[n].keys()) for n in graph.edges(keys=True)]))
        else:
            edge_att_names = set(itertools.chain(*[list(graph.edges[n].keys()) for n in graph.edges()]))
        return edge_att_names

    @staticmethod
    def infer_node_attributes(graph: Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]):
        # find edge attributes names if not provided.
        node_att_names = set(itertools.chain(*[list(graph.nodes[n].keys()) for n in graph.nodes()]))
        return node_att_names

    @staticmethod
    def infer_graph_attributes(graph: Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]):
        graph_att_names = set(graph.graph.keys())
        return graph_att_names

    @staticmethod
    def multigraph_to_graphs(mg: Union[nx.MultiDiGraph, nx.MultiGraph]) -> Dict[Hashable, Union[nx.Graph, nx.DiGraph]]:
        """
        Convert a networkx multigraph to a dictionary of graphs.
        :param mg: The input MultiGraph or MultiDiGraph to convert,
        :return: A dictionary keyed on the MultiGraphs edges keys, with DiGraph or Graph values
        """
        if type(mg) is nx.MultiGraph:
            g_class = nx.Graph
        elif type(mg) is nx.MultiDiGraph:
            g_class = nx.DiGraph
        else:
            raise TypeError("Must give a MultiGraph or MultiDiGraph to convert_multigraph")
        graphs = {}
        for edge in mg.edges(data=True, keys=True):
            link = tuple(edge[:2])
            key = edge[2]
            data = edge[3]
            if key not in graphs:
                graphs[key] = g_class()
                graphs[key].add_nodes_from(mg.nodes(data=True))
            graphs[key].add_edge(link[0], link[1])
            for att_key in data:
                graphs[key].edges[link][att_key] = data[att_key]
        return graphs

    @classmethod
    def flatten_multigraph(cls, mg: Union[nx.MultiDiGraph, nx.MultiGraph], sum_weight=True) -> Union[nx.Graph, nx.DiGraph]:
        """
        flattens a multigraph into a single graph
        :param sum_weight: whether to sum weights of duplicate edges
        :param mg:
        :return:
        """
        if sum_weight:
            if type(mg) is nx.MultiGraph:
                G = nx.Graph()
            elif type(mg) is nx.MultiDiGraph:
                G = nx.DiGraph()
            else:
                raise TypeError("Must give a MultiGraph or MultiDiGraph to convert_multigraph")
            G.add_nodes_from(mg.nodes(data=True))
            for u, v, data in mg.edges(data=True):
                w = data['weight'] if 'weight' in data else 1.0
                if G.has_edge(u, v):
                    G[u][v]['weight'] += w
                else:
                    G.add_edge(u, v, weight=w)
                for att_key in data:
                    if 'weight' not in att_key:
                        G[u][v][att_key] = data[att_key]
            for key in mg.graph.keys():
                G.graph[key] = mg.graph[key]
        else:
            if type(mg) is nx.MultiGraph:
                G = nx.Graph(mg)
            elif type(mg) is nx.MultiDiGraph:
                G = nx.DiGraph(mg)
            else:
                raise TypeError("Must give a MultiGraph or MultiDiGraph to convert_multigraph")
        return G

    @staticmethod
    def graphs_to_multigraph(graphs: Union[List[Union[nx.Graph, nx.DiGraph]],
                                           Dict[Hashable, Union[nx.Graph, nx.DiGraph]]]
                             ) -> Union[nx.MultiGraph, nx.MultiDiGraph]:
        """
        Get a nx MultiGraph or MultiDiGraph from list of nx Graphs or DiGraphs.
        :param graphs: The list or dictionary of input graphs to convert. If list, indexes are used as edge keys. If
                       Dictionary, keys are used as edge keys.
        :return: The networkx MultiGraph or MultiDiGraph Representation of the data
        """
        if type(graphs) is list or type(graphs) is tuple:
            itr = list(enumerate(graphs))
        elif type(graphs) is dict:
            itr = list(graphs.items())
        else:
            raise TypeError("Must pass list or dictionary of Graphs")
        if False not in [type(g) is nx.Graph for _, g in itr]:
            multi_graph = nx.MultiGraph()
        elif False not in [type(g) is nx.DiGraph for _, g in itr]:
            multi_graph = nx.MultiDiGraph()
        else:
            raise TypeError("Iterable must contain networkx Graph or DiGraph.")
        for i, g in itr:
            nodes = g.nodes(data=True)
            for source, target, data in g.edges(data=True):
                multi_graph.add_nodes_from([(source, nodes[source])])
                multi_graph.add_nodes_from([(target, nodes[target])])
                multi_graph.add_edges_from([(source, target, i, data)])
        return multi_graph

    @classmethod
    def get_adjacency_representation(cls, graph: Union[nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph]):
        """
        Get the graph as an adjacency matrix, a list of node attribute DataFrames, and a list of edge attribute
        DataFrames. If a multigraph is passed, it will be converted to a dictionary of graphs keyed on edges keys via
        `GraphIO.multigraph_to_graphs'.
        Node labels are added to the node attribute DataFrame with attribute name `original_node_label`
        :param graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph: The input networkx graph
        :return: If a Graph or Digraph is given:
                    return Tuple[adjacency: ndarray,
                                node_attributes: DataFrame,
                                edge_attributes: DataFrame]
                 If a MultiGraph or MultiDiGraph is given:
                    return Dict[Hashable: Tuple[adjacency: ndarray,
                                          node_attributes: DataFrame,
                                          edge_attributes: DataFrame]
                                ]
                    where the keys of the dictionary are the edge keys from the multigraph.
        """
        out = {}
        if type(graph) in [nx.MultiGraph, nx.MultiDiGraph]:
            graph = cls.multigraph_to_graphs(graph)
        else:
            graph = {0: graph}

        for key in graph.keys():
            g = graph[key]
            np_adj = nx.to_numpy_matrix(g, weight='weight')
            ids = g.nodes()
            g = nx.convert_node_labels_to_integers(g)
            edge_data = g.edges(data=True)
            edge_data = {tuple(e[:2]): e[2] for e in edge_data}
            node_data = g.nodes(data=True)
            node_data = {n[0]: n[1] for n in node_data}
            for i, n in enumerate(ids):
                node_data[i]['original_node_label'] = n
            out[key] = (np_adj,
                        pd.DataFrame.from_dict(node_data, orient='index'),
                        pd.DataFrame.from_dict(edge_data, orient='index'))

        if len(out) == 1:
            return out[0]
        else:
            return out

    @classmethod
    def dump(cls,
             graph: Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph],
             path: str,
             create_using=nx.readwrite.node_link_data):
        """
        Write graph to disk at specified path, using the json file spec.
        :param create_using: function to use for serialization
        :param graph: the graph or list of graphs to dump
        :param path: path to write to, without extension
        :return: None
        """

        node_att_names = cls.infer_node_attributes(graph)
        edge_att_names = cls.infer_edge_attributes(graph)
        graph_att_names = cls.infer_graph_attributes(graph)
        all_att = node_att_names | edge_att_names | graph_att_names
        reserved = {'id', 'source', 'target', 'key'}

        if len(reserved.intersection(all_att)) > 0:
            raise KeyError("Keywords id, source, target, and key are reserved in this format. Any attributes using "
                           "these keywords must be renamed")

        if False in [type(n) == str for n in all_att]:
            print("WARNING: Non-string attribute names detected. "
                  "These will be converted to strings for json compliance", sys.stderr)
        try:
            node_link = create_using(graph)
        except nx.NetworkXError:
            raise KeyError("Node link map is corrupted")

        with open(path, 'w') as f:
            json.dump(node_link, f)

    @classmethod
    def load(cls, path: str, load_using=nx.readwrite.node_link_graph) -> Tuple[Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph], set, set, set]:
        """
        load a json graph from disk.
        :param path: location of graph file.
        :return (graph object, edge attributes, node attributes, graph attributes)
        """
        with open(path, 'r') as f:
            data_dict = json.load(f)

        try:
            graph = nx.readwrite.node_link_graph(data_dict)
        except nx.NetworkXError:
            raise IOError("Unable to graph. Make sure the file is not corrupted, and"
                          "uses the standard source, target, id, and key field names")

        e_att_names = cls.infer_edge_attributes(graph)
        n_att_names = cls.infer_node_attributes(graph)
        g_att_names = cls.infer_graph_attributes(graph)

        return graph, e_att_names, n_att_names, g_att_names