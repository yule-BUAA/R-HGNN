import torch
import torch.nn as nn
import warnings
import copy
import os
import json
import sys
import shutil
from tqdm import tqdm
import dgl

from utils.utils import set_random_seed, convert_to_gpu, load_dataset
from utils.EarlyStopping import EarlyStopping
from utils.utils import get_n_params, get_edge_data_loader, get_predict_edge_index, get_optimizer_and_lr_scheduler, \
    evaluate_link_prediction
from model.R_HGNN import R_HGNN
from utils.LinkScorePredictor import LinkScorePredictor

"""
in link prediction, the training process takes one positive sample with negative_sample_edge_num samples
the evaluation process (validation and test) takes in one positive sample with one negative sample
"""
args = {
    'dataset': 'OGB_MAG',
    'model_name': 'R_HGNN_lr0.001_dropout0.3_seed_0_link_prediction',
    'predict_category': 'paper',
    'seed': 0,
    'cuda': 1,
    'learning_rate': 0.001,
    'num_heads': 8,
    'hidden_units': 32,
    'relation_hidden_units': 8,
    'dropout': 0.3,
    'n_layers': 2,
    'residual': True,
    'norm': True,
    'batch_size': 760,  # the number of edges to train in each batch
    'node_neighbors_min_num': 10,  # number of sampled edges for each type for each GNN layer
    'negative_sample_edge_num': 5,
    'sample_edge_rate': 0.01,  # train: validate: test = 3 : 1 : 1
    'sampled_edge_type': 'affiliated_with',  # writes or affiliated_with, two kinds of predicted relations
    'optimizer': 'adam',
    'weight_decay': 0.0,
    'epochs': 200,
    'patience': 50
}
args['data_path'] = f'../dataset/{args["dataset"]}/{args["dataset"]}.pkl'
args['data_split_idx_path'] = f'../dataset/{args["dataset"]}/{args["dataset"]}_split_idx.pkl'
args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'


def evaluate(model: nn.Module, loader: dgl.dataloading.NodeDataLoader, loss_func: nn.Module,
             sampled_edge_type: str, device: str, mode: str):
    """

    :param model: model
    :param loader: data loader (validate or test)
    :param loss_func: loss function
    :param sampled_edge_type: str
    :param device: device str
    :param mode: str, evaluation mode, validate or test
    :return:
    """
    model.eval()
    with torch.no_grad():
        y_trues = []
        y_predicts = []
        total_loss = 0.0
        loader_tqdm = tqdm(loader, ncols=120)
        for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(loader_tqdm):
            blocks = [convert_to_gpu(b, device=device) for b in blocks]
            positive_graph, negative_graph = convert_to_gpu(positive_graph, negative_graph, device=device)
            # target node relation representation in the heterogeneous graph
            input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                              blocks[0].canonical_etypes}

            nodes_representation, _ = model[0](blocks, copy.deepcopy(input_features))

            positive_score = model[1](positive_graph, nodes_representation, sampled_edge_type).squeeze(dim=-1)
            negative_score = model[1](negative_graph, nodes_representation, sampled_edge_type).squeeze(dim=-1)

            y_predict = torch.cat([positive_score, negative_score], dim=0)
            y_true = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)], dim=0)

            loss = loss_func(y_predict, y_true)

            total_loss += loss.item()
            y_trues.append(y_true.detach().cpu())
            y_predicts.append(y_predict.detach().cpu())

            loader_tqdm.set_description(f'{mode} for the {batch}-th batch, {mode} loss: {loss.item()}')

        total_loss /= (batch + 1)
        y_trues = torch.cat(y_trues, dim=0)
        y_predicts = torch.cat(y_predicts, dim=0)

    return total_loss, y_trues, y_predicts


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    set_random_seed(args['seed'])

    print(f'loading dataset {args["dataset"]}...')

    graph, _, _, _, _, _ = load_dataset(data_path=args['data_path'],
                                        predict_category=args['predict_category'],
                                        data_split_idx_path=args['data_split_idx_path'])

    reverse_etypes = dict()
    for stype, etype, dtype in graph.canonical_etypes:
        for srctype, reltype, dsttype in graph.canonical_etypes:
            if srctype == dtype and dsttype == stype and reltype != etype:
                reverse_etypes[etype] = reltype
                break

    print(f'generating edge idx...')
    train_edge_idx, valid_edge_idx, test_edge_idx = get_predict_edge_index(graph,
                                                                           sample_edge_rate=args['sample_edge_rate'],
                                                                           sampled_edge_type=args['sampled_edge_type'],
                                                                           seed=args['seed'])

    print(
        f'train edge num: {len(train_edge_idx)}, valid edge num: {len(valid_edge_idx)}, test edge num: {len(test_edge_idx)}')

    print(f'get edge data loader...')

    train_loader, val_loader, test_loader = get_edge_data_loader(args['node_neighbors_min_num'], args['n_layers'],
                                                                 graph,
                                                                 batch_size=args['batch_size'],
                                                                 sampled_edge_type=args['sampled_edge_type'],
                                                                 negative_sample_edge_num=args[
                                                                     'negative_sample_edge_num'],
                                                                 train_edge_idx=train_edge_idx,
                                                                 valid_edge_idx=valid_edge_idx,
                                                                 test_edge_idx=test_edge_idx,
                                                                 reverse_etypes=reverse_etypes)

    r_hgnn = R_HGNN(graph=graph,
                    input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes},
                    hidden_dim=args['hidden_units'], relation_input_dim=args['relation_hidden_units'],
                    relation_hidden_dim=args['relation_hidden_units'],
                    num_layers=args['n_layers'], n_heads=args['num_heads'], dropout=args['dropout'],
                    residual=args['residual'], norm=args['norm'])

    link_score_predictor = LinkScorePredictor(hid_dim=args['hidden_units'] * args['num_heads'])

    model = nn.Sequential(r_hgnn, link_score_predictor)

    model = convert_to_gpu(model, device=args['device'])
    print(model)

    print(f'Model #Params: {get_n_params(model)}.')

    print(f'configuration is {args}')

    optimizer, scheduler = get_optimizer_and_lr_scheduler(model, args['optimizer'], args['learning_rate'],
                                                          args['weight_decay'],
                                                          steps_per_epoch=len(train_loader), epochs=args['epochs'])

    save_model_folder = f"../save_model/{args['dataset']}/{args['model_name']}"

    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=args['patience'], save_model_folder=save_model_folder,
                                   save_model_name=args['model_name'])

    loss_func = nn.BCELoss()

    train_steps = 0

    best_validate_RMSE, final_result = None, None

    for epoch in range(args['epochs']):
        model.train()

        train_y_trues = []
        train_y_predicts = []
        train_total_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, ncols=120)
        for batch, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(train_loader_tqdm):
            blocks = [convert_to_gpu(b, device=args['device']) for b in blocks]
            positive_graph, negative_graph = convert_to_gpu(positive_graph, negative_graph, device=args['device'])
            # target node relation representation in the heterogeneous graph
            input_features = {(stype, etype, dtype): blocks[0].srcnodes[dtype].data['feat'] for stype, etype, dtype in
                              blocks[0].canonical_etypes}

            nodes_representation, _ = model[0](blocks, copy.deepcopy(input_features))

            positive_score = model[1](positive_graph, nodes_representation, args['sampled_edge_type']).squeeze(dim=-1)
            negative_score = model[1](negative_graph, nodes_representation, args['sampled_edge_type']).squeeze(dim=-1)

            train_y_predict = torch.cat([positive_score, negative_score], dim=0)
            train_y_true = torch.cat([torch.ones_like(positive_score), torch.zeros_like(negative_score)], dim=0)
            loss = loss_func(train_y_predict, train_y_true)

            train_total_loss += loss.item()
            train_y_trues.append(train_y_true.detach().cpu())
            train_y_predicts.append(train_y_predict.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loader_tqdm.set_description(f'training for the {batch}-th batch, train loss: {loss.item()}')

            # step should be called after a batch has been used for training.
            train_steps += 1
            scheduler.step(train_steps)

        train_total_loss /= (batch + 1)
        train_y_trues = torch.cat(train_y_trues, dim=0)
        train_y_predicts = torch.cat(train_y_predicts, dim=0)

        train_RMSE, train_MAE = evaluate_link_prediction(predict_scores=train_y_predicts, true_scores=train_y_trues)

        model.eval()

        val_total_loss, val_y_trues, val_y_predicts = evaluate(model, loader=val_loader, loss_func=loss_func,
                                                               sampled_edge_type=args['sampled_edge_type'],
                                                               device=args['device'], mode='validate')

        val_RMSE, val_MAE = evaluate_link_prediction(predict_scores=val_y_predicts,
                                                     true_scores=val_y_trues)

        test_total_loss, test_y_trues, test_y_predicts = evaluate(model, loader=test_loader, loss_func=loss_func,
                                                                  sampled_edge_type=args['sampled_edge_type'],
                                                                  device=args['device'], mode='test')

        test_RMSE, test_MAE = evaluate_link_prediction(predict_scores=test_y_predicts,
                                                       true_scores=test_y_trues)

        if best_validate_RMSE is None or val_RMSE < best_validate_RMSE:
            best_validate_RMSE = val_RMSE
            scores = {"RMSE": float(f"{test_RMSE:.4f}"), "MAE": float(f"{test_MAE:.4f}")}
            final_result = json.dumps(scores, indent=4)

        print(
            f'Epoch: {epoch}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_total_loss:.4f}, RMSE {train_RMSE:.4f}, MAE {train_MAE:.4f}, \n'
            f'validate loss: {val_total_loss:.4f}, RMSE {val_RMSE:.4f}, MAE {val_MAE:.4f}, \n'
            f'test loss: {test_total_loss:.4f}, RMSE {test_RMSE:.4f}, MAE {test_MAE:.4f}')

        early_stop = early_stopping.step([('RMSE', val_RMSE, False), ('MAE', val_MAE, False)], model)

        if early_stop:
            break

    # save model result
    save_result_folder = f"../results/{args['dataset']}"
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args['model_name']}.json")

    with open(save_result_path, 'w') as file:
        file.write(final_result)
        file.close()

    print(f'save as {save_result_path}')
    print(f"predicted relation: {args['sampled_edge_type']}")
    print(f'result: {final_result}')

    # sys.exit()
