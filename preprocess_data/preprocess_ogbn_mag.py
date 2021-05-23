import torch
import dgl
from torch_sparse import transpose
from torch_geometric.nn import MetaPath2Vec
from dgl.data.utils import save_graphs
import os

from ogb.nodeproppred import DglNodePropPredDataset, PygNodePropPredDataset

"""
    First generate features of nodes that are not associated features in the original graphs
    Then add reverse relations in the original graph, assign the node features
    Finally, store the processed graph
"""

if __name__ == "__main__":
    # process ogbn-mag dataset and use metapath2vec model to generate features for other types of nodes
    args = {
        'cuda': 1,
        'embedding_dim': 128,
        'walk_length': 64,
        'context_size': 7,
        'walks_per_node': 5,
        'num_negative_samples': 5,
        'batch_size': 128,
        'learning_rate': 0.01,
        'epochs': 5,
        'log_steps': 100  # print step
    }

    args['cuda'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'

    os.makedirs('../dataset/OGB_MAG', exist_ok=True)

    nodes_embedding_path = '../dataset/OGB_MAG/nodes_embedding.pkl'
    if os.path.exists(nodes_embedding_path):
        # dictionary of node embeddings, dict
        embedding_dict = torch.load(nodes_embedding_path, map_location='cpu')
        print(f"{nodes_embedding_path} is loaded successfully.")

    else:
        pyg_dataset = PygNodePropPredDataset(name='ogbn-mag', root='../dataset')
        pyg_data = pyg_dataset[0]

        # ('author', 'affiliated_with', 'institution'),
        # ('author', 'writes', 'paper'),
        # ('paper', 'cites', 'paper'),
        # ('paper', 'has_topic', 'field_of_study')
        # reverse edges (relations) in the heterogeneous graph.
        pyg_data.edge_index_dict[('institution', 'rev_affiliated_with', 'author')] = transpose(
            pyg_data.edge_index_dict[('author', 'affiliated_with', 'institution')],
            None, m=pyg_data.num_nodes_dict['author'],
            n=pyg_data.num_nodes_dict['institution'])[0]

        pyg_data.edge_index_dict[('paper', 'rev_writes', 'author')] = transpose(
            pyg_data.edge_index_dict[('author', 'writes', 'paper')], None,
            m=pyg_data.num_nodes_dict['author'], n=pyg_data.num_nodes_dict['paper'])[0]

        pyg_data.edge_index_dict[('paper', 'rev_cites', 'paper')] = transpose(
            pyg_data.edge_index_dict[('paper', 'cites', 'paper')], None,
            m=pyg_data.num_nodes_dict['paper'],
            n=pyg_data.num_nodes_dict['paper'])[0]

        pyg_data.edge_index_dict[('field_of_study', 'rev_has_topic', 'paper')] = transpose(
            pyg_data.edge_index_dict[('paper', 'has_topic', 'field_of_study')], None,
            m=pyg_data.num_nodes_dict['paper'],
            n=pyg_data.num_nodes_dict['field_of_study'])[0]

        print(pyg_data)

        metapath = [
            ('author', 'writes', 'paper'),
            ('paper', 'has_topic', 'field_of_study'),
            ('field_of_study', 'rev_has_topic', 'paper'),
            ('paper', 'rev_cites', 'paper'),
            ('paper', 'rev_writes', 'author'),
            ('author', 'affiliated_with', 'institution'),
            ('institution', 'rev_affiliated_with', 'author'),
            ('author', 'writes', 'paper'),
            ('paper', 'cites', 'paper'),
            ('paper', 'rev_writes', 'author')
        ]

        metapath2vec_model = MetaPath2Vec(pyg_data.edge_index_dict, embedding_dim=args['embedding_dim'],
                                          metapath=metapath, walk_length=args['walk_length'],
                                          context_size=args['context_size'],
                                          walks_per_node=args['walks_per_node'],
                                          num_negative_samples=args['num_negative_samples']).to(args['cuda'])

        loader = metapath2vec_model.loader(batch_size=args['batch_size'], shuffle=True, num_workers=4)
        optimizer = torch.optim.Adam(metapath2vec_model.parameters(), lr=0.01)

        metapath2vec_model.train()
        for epoch in range(1, args['epochs'] + 1):
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = metapath2vec_model.loss(pos_rw.to(args['cuda']), neg_rw.to(args['cuda']))
                loss.backward()
                optimizer.step()

                if (i + 1) % args['log_steps'] == 0:
                    print(f'Epoch: {epoch:02d}, Step: {i + 1:03d}/{len(loader)}, '
                          f'Loss: {loss:.4f}')

        embedding_dict = {}
        for node_type in metapath2vec_model.num_nodes_dict:
            # get embedding of node with specific type
            embedding_dict[node_type] = metapath2vec_model(node_type).detach().cpu()

        torch.save(embedding_dict, nodes_embedding_path)

    dataset = DglNodePropPredDataset(name='ogbn-mag', root='../dataset')

    original_graph, labels = dataset[0]

    # add reverse relations
    data_dict = {}
    for src_type, etype, dst_type in original_graph.canonical_etypes:
        src, dst = original_graph.edges(etype=etype)
        data_dict[(src_type, etype, dst_type)] = (src, dst)
        data_dict[(dst_type, f'rev_{etype}', src_type)] = (dst, src)

    graph = dgl.heterograph(data_dict=data_dict, num_nodes_dict={ntype: original_graph.number_of_nodes(ntype) for ntype in original_graph.ntypes})

    graph.nodes['paper'].data['year'] = original_graph.nodes['paper'].data['year']
    # concat the content and structural feature of paper
    graph.nodes['paper'].data['feat'] = torch.cat([original_graph.nodes['paper'].data['feat'], embedding_dict['paper']], dim=1)
    # add feature of nodes that are not associated with original features in the graph
    graph.nodes['author'].data['feat'] = embedding_dict['author']
    graph.nodes['institution'].data['feat'] = embedding_dict['institution']
    graph.nodes['field_of_study'].data['feat'] = embedding_dict['field_of_study']

    # add types of edges
    for src_type, etype, dst_type in original_graph.canonical_etypes:
        graph.edges[(src_type, etype, dst_type)].data['reltype'] = \
        original_graph.edges[(src_type, etype, dst_type)].data['reltype']
        graph.edges[(dst_type, f'rev_{etype}', src_type)].data['reltype'] = \
        original_graph.edges[(src_type, etype, dst_type)].data['reltype'] + len(original_graph.etypes)

    graph_output_path = '../dataset/OGB_MAG/OGB_MAG.pkl'

    save_graphs(graph_output_path, graph, labels)

    print(f"{graph_output_path} writes successfully.")

    split_idx = dataset.get_idx_split()

    split_idx = {
        'train': {'paper': split_idx['train']['paper']},
        'valid': {'paper': split_idx['valid']['paper']},
        'test': {'paper': split_idx['test']['paper']}
    }
    split_idx_output_path = '../dataset/OGB_MAG/OGB_MAG_split_idx.pkl'
    torch.save(split_idx, split_idx_output_path)
    print(f"{split_idx_output_path} writes successfully.")
