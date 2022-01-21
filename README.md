# Heterogeneous Graph Representation Learning with Relation Awareness

This experiment is based on stanford OGB (1.3.1) benchmark. 
The description of "Heterogeneous Graph Representation Learning with Relation Awareness" is [available here](https://arxiv.org/abs/2105.11122). 

### To run the node classification task, the steps are:

  1. run ```python preprocess_ogbn_mag.py``` to preprocess the original ogbn_mag dataset. 
  As the OGB-MAG dataset only has input features for paper nodes, for all the other types of nodes (author, affiliation, field), we use the metapath2vec model to generate their structural features. 

  2. run ```python train_R_HGNN_ogbn_mag_node_classification.py``` to train the model.

  3. run ```python eval_R_HGNN_ogbn_mag_node_classification.py``` to evaluate the model.
  
### To run the link prediction task, the steps are:

  1. run ```python preprocess_ogbn_mag.py``` to preprocess the original ogbn_mag dataset. 
  
2. run ```python R_HGNN_ogbn_mag_link_prediction.py``` to train the model and get final performance.


## Environments:
- [PyTorch 1.7.1](https://pytorch.org/)
- [DGL 0.5.3](https://www.dgl.ai/)
- [PyTorch Geometric 1.6.3](https://pytorch-geometric.readthedocs.io/en/latest/)
- [OGB 1.3.1](https://ogb.stanford.edu/docs/home/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://github.com/numpy/numpy)

## Selected hyperparameters:

```
  num_heads               INT      Number of attention heads                          8
  hidden_units            INT      Dimension of node hidden units for each head       64
  relation_hidden_units   INT      Dimension of relation units for each head          8
  n_layers                INT      Number of GNN layers                               2
  learning_rate           FLOAT    Learning rate                                      0.001
  dropout                 FLOAT    Dropout rate                                       0.5
  residual                BOOL     Whether to use the residual connection             True
```

Hyperparameters could be found in the ```args``` variable in ```train_R_HGNN_ogbn_mag.py``` file and you can adjust them when training the model.
When evaluating the model, please make sure the ```args``` in ```eval_R_HGNN_ogbn_mag.py``` keep the same to those in the training process.

## Reference performance for the OGB-MAG dataset:

We run R-HGNN for 10 times with the random seed from 0 to 9 and report the averaged performance.

| Model        | Test Accuracy   | Valid Accuracy  | # Parameter     | Hardware         |
| ---------    | --------------- | --------------  | --------------  |--------------    |
| R-HGNN  | 0.5204 ± 0.0026   | 0.5361 ± 0.0022  |    5,638,053      | NVIDIA Tesla T4 (15 GB) |

## Citation
Please consider citing our paper when using the code.

```bibtex
@article{yu2021heterogeneous,
  title={Heterogeneous Graph Representation Learning with Relation Awareness},
  author={Yu, Le and Sun, Leilei and Du, Bowen and Liu, Chuanren and Lv, Weifeng and Xiong, Hui},
  journal={arXiv preprint arXiv:2105.11122},
  year={2021}
}
```
