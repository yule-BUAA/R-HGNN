import torch
import torch.nn as nn
import warnings
import copy
import os
import json
import sys

from utils.utils import set_random_seed, convert_to_gpu, load_dataset, get_n_params, evaluate_node_classification
from model.R_HGNN import R_HGNN
from utils.Classifier import Classifier


args = {
    'dataset': 'OGB_MAG',
    'model_name': 'R_HGNN_lr0.001_dropout0.5_seed_0',
    'predict_category': 'paper',
    'seed': 0,
    'cuda': -1,   # if the GPU device is out of memory during the evaluation, set 'cuda' to a negative number to use CPU
    'learning_rate': 0.001,
    'num_heads': 8,
    'hidden_units': 64,
    'relation_hidden_units': 8,
    'dropout': 0.5,
    'n_layers': 2,
    'residual': True
}
args['data_path'] = f'../dataset/{args["dataset"]}/{args["dataset"]}.pkl'
args['data_split_idx_path'] = f'../dataset/{args["dataset"]}/{args["dataset"]}_split_idx.pkl'
args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    set_random_seed(args['seed'])

    print(f'loading dataset {args["dataset"]}...')

    graph, labels, num_classes, train_idx, valid_idx, test_idx = load_dataset(data_path=args['data_path'],
                                                 predict_category=args['predict_category'],
                                                 data_split_idx_path=args['data_split_idx_path'])

    r_hgnn = R_HGNN(graph=graph,
                    input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes},
                    hidden_dim=args['hidden_units'], relation_input_dim=args['relation_hidden_units'],
                    relation_hidden_dim=args['relation_hidden_units'],
                    num_layers=args['n_layers'], n_heads=args['num_heads'], dropout=args['dropout'],
                    residual=args['residual'])

    classifier = Classifier(n_hid=args['hidden_units'] * args['num_heads'], n_out=num_classes)

    model = nn.Sequential(r_hgnn, classifier)

    model = convert_to_gpu(model, device=args['device'])
    print(model)

    print(f'Model #Params: {get_n_params(model)}.')

    print(f'configuration is {args}')

    save_model_path = f"../save_model/{args['dataset']}/{args['model_name']}/{args['model_name']}.pkl"

    # load model parameter
    model.load_state_dict(torch.load(save_model_path, map_location='cpu'))

    # evaluate the best model
    model.eval()

    nodes_representation, _ = model[0].inference(graph, copy.deepcopy({(stype, etype, dtype): graph.nodes[dtype].data['feat'] for stype, etype, dtype in
                      graph.canonical_etypes}), device=args['device'])

    train_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[train_idx]
    train_y_trues = convert_to_gpu(labels[train_idx], device=args['device'])
    train_accuracy, train_macro_f1 = evaluate_node_classification(predicts=train_y_predicts.argmax(dim=1),
                                                                                  labels=train_y_trues)
    print(f'final train accuracy: {train_accuracy:.4f}, macro f1 {train_macro_f1:.4f}')

    val_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[valid_idx]
    val_y_trues = convert_to_gpu(labels[valid_idx], device=args['device'])
    val_accuracy, val_macro_f1 = evaluate_node_classification(predicts=val_y_predicts.argmax(dim=1),
                                                                            labels=val_y_trues)

    print(f'final valid accuracy {val_accuracy:.4f}, macro f1 {val_macro_f1:.4f}')

    test_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[test_idx]
    test_y_trues = convert_to_gpu(labels[test_idx], device=args['device'])
    test_accuracy, test_macro_f1 = evaluate_node_classification(predicts=test_y_predicts.argmax(dim=1),
                                                                               labels=test_y_trues)
    print(f'final test accuracy {test_accuracy:.4f}, macro f1 {test_macro_f1:.4f}')

    # save model result
    result_json = {
        "train accuracy": float(f"{train_accuracy:.4f}"), "train macro f1": float(f"{train_macro_f1:.4f}"),
        "validate accuracy": float(f"{val_accuracy:.4f}"), "validate macro f1": float(f"{val_macro_f1:.4f}"),
        "test accuracy": float(f"{test_accuracy:.4f}"), "test macro f1": float(f"{test_macro_f1:.4f}")
    }
    result_json = json.dumps(result_json, indent=4)

    print(result_json)

    save_result_folder = f"../results/{args['dataset']}"
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args['model_name']}.json")

    with open(save_result_path, 'w') as file:
        file.write(result_json)

    # sys.exit()
