# Hypergraph-MLP

This is the repo for our ICASSP 2024 paper: [Hypergraph-MLP: Learning on Hypergraphs without Message Passing](https://arxiv.org/pdf/2312.09778).

## Overview

A quick summary of different folders:

- 'baselines_hypergnn' contains the source code for our baseline hypergraph neural networks.

- 'ours' contains the source code for our Hypergraph-MLP.

## Recommend Environment:
```
conda create -n "hgmlp" python=3.7
conda activate hgmlp
```
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
```
```
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-sparse==0.6.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-cluster==1.5.2 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-geometric==1.6.3 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install ipdb
pip install tqdm
pip install scipy
pip install matplotlib
```
## Data Preparation:

To generate a dataset for training using PyG or DGL, please set up the following three directories:
```
p2root: './data/pyg_data/hypergraph_dataset_updated/'
p2raw: './data/AllSet_all_raw_data/'
p2dgl_data: './data/dgl_data_raw/'
```

Next, unzip the raw data zip file into `p2raw`. The raw data zip file can be found in this [link](https://github.com/jianhao2016/AllSet/tree/main/data/raw_data).

## Acknowledgement

This code is based on the official code of AllSet ([Paper](https://openreview.net/forum?id=hpBTIv2uy_E); [Github](https://github.com/jianhao2016/AllSet)). Sincere appreciation is extended for their valuable contributions.

## Citation

If you use this code, please cite our paper:

```
@inproceedings{tang2024hypergraph,
  title={Hypergraph-MLP: Learning on Hypergraphs without Message Passing},
  author={Tang, Bohan and Chen, Siheng and Dong, Xiaowen},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```


