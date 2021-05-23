import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from model.RelationGraphConv import RelationGraphConv
from model.HeteroConv import HeteroGraphConv
from model.RelationCrossing import RelationCrossing
from model.RelationFusing import RelationFusing


class R_HGNN_Layer(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim: int, hidden_dim: int, relation_input_dim: int,
                 relation_hidden_dim: int, n_heads: int = 8, dropout: float = 0.2, negative_slope: float = 0.2,
                 residual: bool = True):
        """

        :param graph: a heterogeneous graph
        :param input_dim: int, node input dimension
        :param hidden_dim: int, node hidden dimension
        :param relation_input_dim: int, relation input dimension
        :param relation_hidden_dim: int, relation hidden dimension
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param negative_slope: float, negative slope
        :param residual: boolean, residual connections or not
        """
        super(R_HGNN_Layer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual

        # node transformation parameters of each type
        self.node_transformation_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(input_dim, n_heads * hidden_dim))
            for ntype in graph.ntypes
        })

        # relation transformation parameters of each type, used as attention queries
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(relation_input_dim, n_heads * 2 * hidden_dim))
            for etype in graph.etypes
        })

        # relation propagation layer of each relation
        self.relation_propagation_layer = nn.ModuleDict({
            etype: nn.Linear(relation_input_dim, n_heads * relation_hidden_dim)
            for etype in graph.etypes
        })

        # hetero conv modules, each RelationGraphConv deals with a single type of relation
        self.hetero_conv = HeteroGraphConv({
            etype: RelationGraphConv(in_feats=(input_dim, input_dim), out_feats=hidden_dim,
                                     num_heads=n_heads, dropout=dropout, negative_slope=negative_slope)
            for etype in graph.etypes
        })

        if self.residual:
            # residual connection
            self.res_fc = nn.ModuleDict()
            self.residual_weight = nn.ParameterDict()
            for ntype in graph.ntypes:
                self.res_fc[ntype] = nn.Linear(input_dim, n_heads * hidden_dim)
                self.residual_weight[ntype] = nn.Parameter(torch.randn(1))

        # relation type crossing attention trainable parameters
        self.relations_crossing_attention_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, hidden_dim))
            for etype in graph.etypes
        })
        # different relations crossing layer
        self.relations_crossing_layer = RelationCrossing(in_feats=n_heads * hidden_dim,
                                                         out_feats=hidden_dim,
                                                         num_heads=n_heads,
                                                         dropout=dropout,
                                                         negative_slope=negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[weight], gain=gain)
        for etype in self.relation_propagation_layer:
            nn.init.xavier_normal_(self.relation_propagation_layer[etype].weight, gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)
        for weight in self.relations_crossing_attention_weight:
            nn.init.xavier_normal_(self.relations_crossing_attention_weight[weight], gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, relation_target_node_features: dict, relation_embedding: dict):
        """

        :param graph: dgl.DGLHeteroGraph
        :param relation_target_node_features: dict, {relation_type: target_node_features shape (N_nodes, input_dim)},
               each value in relation_target_node_features represents the representation of target node features
        :param relation_embedding: embedding for each relation, dict, {etype: feature}
        :return: output_features: dict, {relation_type: target_node_features}
        """
        # in each relation, target type of nodes has an embedding
        # dictionary of {(srctype, etypye, dsttype): target_node_features}
        input_src = relation_target_node_features

        if graph.is_block:
            input_dst = {}
            for srctype, etypye, dsttype in relation_target_node_features:
                input_dst[(srctype, etypye, dsttype)] = relation_target_node_features[(srctype, etypye, dsttype)][
                                                        :graph.number_of_dst_nodes(dsttype)]
        else:
            input_dst = relation_target_node_features

        # output_features, dict {(srctype, etypye, dsttype): target_node_features}
        output_features = self.hetero_conv(graph, input_src, input_dst, relation_embedding,
                                           self.node_transformation_weight, self.relation_transformation_weight)

        # residual connection for the target node
        if self.residual:
            for srctype, etype, dsttype in output_features:
                alpha = F.sigmoid(self.residual_weight[dsttype])
                output_features[(srctype, etype, dsttype)] = output_features[(srctype, etype, dsttype)] * alpha + \
                                                             self.res_fc[dsttype](
                                                                 input_dst[(srctype, etype, dsttype)]) * (1 - alpha)

        output_features_dict = {}
        # different relations crossing layer
        for srctype, etype, dsttype in output_features:
            # (dsttype_node_relations_num, dst_nodes_num, n_heads * hidden_dim)
            dst_node_relations_features = torch.stack([output_features[(stype, reltype, dtype)]
                                                   for stype, reltype, dtype in output_features if dtype == dsttype], dim=0)

            output_features_dict[(srctype, etype, dsttype)] = self.relations_crossing_layer(dst_node_relations_features,
                                                                                            self.relations_crossing_attention_weight[etype])

        relation_embedding_dict = {}
        for etype in relation_embedding:
            relation_embedding_dict[etype] = self.relation_propagation_layer[etype](relation_embedding[etype])

        # relation features after relation crossing layer, {(srctype, etype, dsttype): target_node_features}
        # relation embeddings after relation update, {etype: relation_embedding}
        return output_features_dict, relation_embedding_dict


class R_HGNN(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim_dict: dict, hidden_dim: int, relation_input_dim: int,
                 relation_hidden_dim: int, num_layers: int, n_heads: int = 4,
                 dropout: float = 0.2, negative_slope: float = 0.2, residual: bool = True):
        """

        :param graph: a heterogeneous graph
        :param input_dim_dict: node input dimension dictionary
        :param hidden_dim: int, node hidden dimension
        :param relation_input_dim: int, relation input dimension
        :param relation_hidden_dim: int, relation hidden dimension
        :param num_layers: int, number of stacked layers
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param negative_slope: float, negative slope
        :param residual: boolean, residual connections or not
        """
        super(R_HGNN, self).__init__()

        self.input_dim_dict = input_dim_dict
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.relation_input_dim = relation_input_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual

        # relation embedding dictionary
        self.relation_embedding = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(relation_input_dim, 1)) for etype in graph.etypes
        })

        # align the dimension of different types of nodes
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_dim * n_heads) for ntype in input_dim_dict
        })

        # each layer takes in the heterogeneous graph as input
        self.layers = nn.ModuleList()

        # for each relation_layer
        self.layers.append(
            R_HGNN_Layer(graph, hidden_dim * n_heads, hidden_dim, relation_input_dim, relation_hidden_dim, n_heads,
                         dropout, negative_slope, residual))
        for _ in range(1, self.num_layers):
            self.layers.append(R_HGNN_Layer(graph, hidden_dim * n_heads, hidden_dim, relation_hidden_dim * n_heads,
                                            relation_hidden_dim, n_heads, dropout, negative_slope, residual))

        # transformation matrix for target node representation under each relation
        self.node_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, hidden_dim, hidden_dim)) for etype in graph.etypes
        })

        # transformation matrix for relation representation
        self.relation_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads, relation_hidden_dim, hidden_dim)) for etype in graph.etypes
        })

        # different relations fusing module
        self.relation_fusing = RelationFusing(node_hidden_dim=hidden_dim,
                                              relation_hidden_dim=relation_hidden_dim,
                                              num_heads=n_heads,
                                              dropout=dropout, negative_slope=negative_slope)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')

        for etype in self.relation_embedding:
            nn.init.xavier_normal_(self.relation_embedding[etype], gain=gain)
        for ntype in self.projection_layer:
            nn.init.xavier_normal_(self.projection_layer[ntype].weight, gain=gain)
        for etype in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[etype], gain=gain)
        for etype in self.relation_transformation_weight:
            nn.init.xavier_normal_(self.relation_transformation_weight[etype], gain=gain)

    def forward(self, blocks: list, relation_target_node_features: dict, relation_embedding: dict = None):
        """

        :param blocks: list of sampled dgl.DGLHeteroGraph
        :param relation_target_node_features: target node features under each relation, dict, {(srctype, etype, dsttype): features}
        :param relation_embedding: embedding for each relation, dict, {etype: feature} or None
        :return:
        """
        # target relation feature projection
        for stype, reltype, dtype in relation_target_node_features:
            relation_target_node_features[(stype, reltype, dtype)] = self.projection_layer[dtype](
                relation_target_node_features[(stype, reltype, dtype)])

        # each relation is associated with a specific type, if no semantic information is given,
        # then the one-hot representation of each relation is assign with trainable hidden representation
        if relation_embedding is None:
            relation_embedding = {}
            for etype in self.relation_embedding:
                relation_embedding[etype] = self.relation_embedding[etype].flatten()

        # graph convolution
        for block, layer in zip(blocks, self.layers):
            relation_target_node_features, relation_embedding = layer(block, relation_target_node_features,
                                                                      relation_embedding)

        relation_fusion_embedding_dict = {}
        # relation_target_node_features -> {(srctype, etype, dsttype): target_node_features}
        for dsttype in set([dtype for _, _, dtype in relation_target_node_features]):
            relation_target_node_features_dict = {etype: relation_target_node_features[(stype, etype, dtype)]
                                                  for stype, etype, dtype in relation_target_node_features}
            etypes = [etype for stype, etype, dtype in relation_target_node_features if dtype == dsttype]
            dst_node_features = [relation_target_node_features_dict[etype] for etype in etypes]
            dst_relation_embeddings = [relation_embedding[etype] for etype in etypes]
            dst_node_feature_transformation_weight = [self.node_transformation_weight[etype] for etype in etypes]
            dst_relation_embedding_transformation_weight = [self.relation_transformation_weight[etype] for etype in etypes]

            # Tensor, shape (heads_num * hidden_dim)
            dst_node_relation_fusion_feature = self.relation_fusing(dst_node_features,
                                                                    dst_relation_embeddings,
                                                                    dst_node_feature_transformation_weight,
                                                                    dst_relation_embedding_transformation_weight)

            relation_fusion_embedding_dict[dsttype] = dst_node_relation_fusion_feature

        # relation_fusion_embedding_dict, {ntype: tensor -> (nodes, n_heads * hidden_dim)}
        # relation_target_node_features, {(srctype, etype, dsttype): (dst_nodes, n_heads * hidden_dim)}
        return relation_fusion_embedding_dict, relation_target_node_features

    def inference(self, graph: dgl.DGLHeteroGraph, relation_target_node_features: dict, relation_embedding: dict = None,
                  device: str = 'cuda:0'):
        """
        mini-batch inference of final representation over all node types. Outer loop: Interate the layers, Inner loop: Interate the batches

        :param graph: The whole relational graphs
        :param relation_target_node_features: target node features under each relation, dict, {(srctype, etype, dsttype): features}
        :param relation_embedding: embedding for each relation, dict, {etype: feature} or None
        :param device: device str
        """
        with torch.no_grad():

            if relation_embedding is None:
                relation_embedding = {}
                for etype in self.relation_embedding:
                    relation_embedding[etype] = self.relation_embedding[etype].flatten()

            # interate over each layer
            for index, layer in enumerate(self.layers):
                # Tensor, features of all relation embeddings of the target nodes, store on cpu
                y = {
                    (stype, etype, dtype): torch.zeros(graph.number_of_nodes(dtype), self.hidden_dim * self.n_heads) for
                    stype, etype, dtype in graph.canonical_etypes}

                # full sample for each type of nodes
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    graph,
                    {ntype: torch.arange(graph.number_of_nodes(ntype)) for ntype in graph.ntypes},
                    sampler,
                    batch_size=1280,
                    shuffle=True,
                    drop_last=False,
                    num_workers=4)

                tqdm_dataloader = tqdm(dataloader, ncols=120)
                for batch, (input_nodes, output_nodes, blocks) in enumerate(tqdm_dataloader):
                    block = blocks[0].to(device)

                    # for relational graphs that only contain a single type of nodes, construct the input and output node dictionary
                    if len(set(blocks[0].ntypes)) == 1:
                        input_nodes = {blocks[0].ntypes[0]: input_nodes}
                        output_nodes = {blocks[0].ntypes[0]: output_nodes}

                    input_features = {(stype, etype, dtype): relation_target_node_features[(stype, etype, dtype)][
                        input_nodes[dtype]].to(device)
                                      for stype, etype, dtype in relation_target_node_features.keys()}

                    input_relation_features = relation_embedding

                    if index == 0:
                        # target relation feature projection for the first layer in the full batch inference
                        for stype, reltype, dtype in input_features:
                            input_features[(stype, reltype, dtype)] = self.projection_layer[dtype](
                                input_features[(stype, reltype, dtype)])
                    h, input_relation_features = layer(block, input_features, input_relation_features)
                    for stype, reltype, dtype in h.keys():
                        y[(stype, reltype, dtype)][output_nodes[dtype]] = h[(stype, reltype, dtype)].cpu()

                    tqdm_dataloader.set_description(f'inference for the {batch}-th batch in model {index}-th layer')

                # update the features of all the nodes (after the graph convolution) in the whole graph
                relation_target_node_features = y
                # relation embedding is updated after each layer
                relation_embedding = input_relation_features

            for stype, etype, dtype in relation_target_node_features:
                relation_target_node_features[(stype, etype, dtype)] = relation_target_node_features[
                    (stype, etype, dtype)].to(device)

            relation_fusion_embedding_dict = {}
            # relation_target_node_features -> {(srctype, etype, dsttype): target_node_features}
            for dsttype in set([dtype for _, _, dtype in relation_target_node_features]):

                relation_target_node_features_dict = {etype: relation_target_node_features[(stype, etype, dtype)]
                                                      for stype, etype, dtype in relation_target_node_features}
                etypes = [etype for stype, etype, dtype in relation_target_node_features if dtype == dsttype]
                dst_node_features = [relation_target_node_features_dict[etype] for etype in etypes]
                dst_relation_embeddings = [relation_embedding[etype] for etype in etypes]
                dst_node_feature_transformation_weight = [self.node_transformation_weight[etype] for etype in etypes]
                dst_relation_embedding_transformation_weight = [self.relation_transformation_weight[etype] for etype in etypes]

                # use mini-batch to avoid out of memory in inference
                relation_fusion_embedding = []
                index = 0
                batch_size = 2560
                while index < dst_node_features[0].shape[0]:
                    # Tensor, shape (heads_num * hidden_dim)
                    relation_fusion_embedding.append(self.relation_fusing(
                        [dst_node_feature[index: index + batch_size, :] for dst_node_feature in dst_node_features],
                        dst_relation_embeddings,
                        dst_node_feature_transformation_weight,
                        dst_relation_embedding_transformation_weight))
                    index += batch_size
                relation_fusion_embedding_dict[dsttype] = torch.cat(relation_fusion_embedding, dim=0)

            # relation_fusion_embedding_dict, {ntype: tensor -> (nodes, n_heads * hidden_dim)}
            # relation_target_node_features, {ntype: tensor -> (num_relations, nodes, n_heads * hidden_dim)}
            return relation_fusion_embedding_dict, relation_target_node_features
