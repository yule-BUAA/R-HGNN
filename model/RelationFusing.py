import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationFusing(nn.Module):

    def __init__(self, node_hidden_dim: int, relation_hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        """

        :param node_hidden_dim: int, node hidden feature size
        :param relation_hidden_dim: int,relation hidden feature size
        :param num_heads: int, number of heads in Multi-Head Attention
        :param dropout: float, dropout rate, defaults: 0.0
        :param negative_slope: float, negative slope, defaults: 0.2
        """
        super(RelationFusing, self).__init__()
        self.node_hidden_dim = node_hidden_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, dst_node_features: list, dst_relation_embeddings: list,
                dst_node_feature_transformation_weight: list,
                dst_relation_embedding_transformation_weight: list):
        """
        :param dst_node_features: list, [each shape is (num_dst_nodes, n_heads * node_hidden_dim)]
        :param dst_relation_embeddings: list, [each shape is (n_heads * relation_hidden_dim)]
        :param dst_node_feature_transformation_weight: list, [each shape is (n_heads, node_hidden_dim, node_hidden_dim)]
        :param dst_relation_embedding_transformation_weight:  list, [each shape is (n_heads, relation_hidden_dim, relation_hidden_dim)]
        :return: dst_node_relation_fusion_feature: Tensor of the target node representation after relation-aware representations fusion
        """
        if len(dst_node_features) == 1:
            # (num_dst_nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_features[0]
        else:
            # (num_dst_relations, nodes, n_heads, node_hidden_dim)
            dst_node_features = torch.stack(dst_node_features, dim=0).reshape(len(dst_node_features), -1,
                                                                              self.num_heads, self.node_hidden_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim)
            dst_relation_embeddings = torch.stack(dst_relation_embeddings, dim=0).reshape(len(dst_node_features),
                                                                                          self.num_heads,
                                                                                          self.relation_hidden_dim)
            # (num_dst_relations, n_heads, node_hidden_dim, node_hidden_dim)
            dst_node_feature_transformation_weight = torch.stack(dst_node_feature_transformation_weight, dim=0).reshape(
                len(dst_node_features), self.num_heads,
                self.node_hidden_dim, self.node_hidden_dim)
            # (num_dst_relations, n_heads, relation_hidden_dim, relation_hidden_dim)
            dst_relation_embedding_transformation_weight = torch.stack(dst_relation_embedding_transformation_weight,
                                                                       dim=0).reshape(len(dst_node_features),
                                                                                      self.num_heads,
                                                                                      self.relation_hidden_dim,
                                                                                      self.node_hidden_dim)
            # shape (num_dst_relations, nodes, n_heads, hidden_dim)
            dst_node_features = torch.einsum('abcd,acde->abce', dst_node_features,
                                             dst_node_feature_transformation_weight)

            # shape (num_dst_relations, n_heads, hidden_dim)
            dst_relation_embeddings = torch.einsum('abc,abcd->abd', dst_relation_embeddings,
                                                   dst_relation_embedding_transformation_weight)

            # shape (num_dst_relations, nodes, n_heads, 1)
            attention_scores = (dst_node_features * dst_relation_embeddings.unsqueeze(dim=1)).sum(dim=-1, keepdim=True)
            attention_scores = F.softmax(self.leaky_relu(attention_scores), dim=0)
            # (nodes, n_heads, hidden_dim)
            dst_node_relation_fusion_feature = (dst_node_features * attention_scores).sum(dim=0)
            dst_node_relation_fusion_feature = self.dropout(dst_node_relation_fusion_feature)
            # (nodes, n_heads * hidden_dim)
            dst_node_relation_fusion_feature = dst_node_relation_fusion_feature.reshape(-1,
                                                                                        self.num_heads * self.node_hidden_dim)

        return dst_node_relation_fusion_feature
