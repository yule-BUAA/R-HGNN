import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationCrossing(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, num_heads: int, dropout: float = 0.0, negative_slope: float = 0.2):
        """
        Relation crossing layer
        Parameters
        ----------
        in_feats : pair of ints, input feature size
        out_feats : int, output feature size
        num_heads : int, number of heads in Multi-Head Attention
        dropout : float, optional, dropout rate, defaults: 0.0
        negative_slope : float, optional, negative slope rate, defaults: 0.2
        """
        super(RelationCrossing, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dsttype_node_features: torch.Tensor, relations_crossing_attention_weight: nn.Parameter):
        """
        :param dsttype_node_features: a tensor of (dsttype_node_relations_num, num_dst_nodes, n_heads * hidden_dim)
        :param relations_crossing_attention_weight: Parameter the shape is (n_heads, hidden_dim)
        :return: output_features: a Tensor
        """
        if len(dsttype_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dsttype_node_features = dsttype_node_features.squeeze(dim=0)
        else:
            # (dsttype_node_relations_num, num_dst_nodes, n_heads, hidden_dim)
            dsttype_node_features = dsttype_node_features.reshape(dsttype_node_features.shape[0], -1, self._num_heads, self._out_feats)
            # shape -> (dsttype_node_relations_num, dst_nodes_num, n_heads, 1),  (dsttype_node_relations_num, dst_nodes_num, n_heads, hidden_dim) * (n_heads, hidden_dim)
            dsttype_node_relation_attention = (dsttype_node_features * relations_crossing_attention_weight).sum(dim=-1, keepdim=True)
            dsttype_node_relation_attention = F.softmax(self.leaky_relu(dsttype_node_relation_attention), dim=0)
            # shape -> (dst_nodes_num, n_heads, hidden_dim),  (dsttype_node_relations_num, dst_nodes_num, n_heads, hidden_dim) * (dsttype_node_relations_num, dst_nodes_num, n_heads, 1)
            dsttype_node_features = (dsttype_node_features * dsttype_node_relation_attention).sum(dim=0)
            dsttype_node_features = self.dropout(dsttype_node_features)
            # shape -> (dst_nodes_num, n_heads * hidden_dim)
            dsttype_node_features = dsttype_node_features.reshape(-1, self._num_heads * self._out_feats)

        return dsttype_node_features
