import torch.nn as nn
import dgl
import dgl.function as fn


class LinkScorePredictor(nn.Module):
    """
    a single layer Edge Score Predictor
    """
    def __init__(self, hid_dim):
        super(LinkScorePredictor, self).__init__()

        self.projection_layer = nn.Linear(hid_dim, hid_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge_subgraph: dgl.DGLHeteroGraph, nodes_representation: dict, etype: str):
        """

        :param edge_subgraph: sampled subgraph
        :param nodes_representation: input node features, dict
        :param etype: predict edge type, str
        :return:
        """
        edge_subgraph = edge_subgraph.local_var()
        edge_type_subgraph = edge_subgraph[etype]
        for ntype in nodes_representation:
            edge_type_subgraph.nodes[ntype].data['h'] = self.projection_layer(nodes_representation[ntype])
        edge_type_subgraph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)

        return self.sigmoid(edge_type_subgraph.edata['score'])
