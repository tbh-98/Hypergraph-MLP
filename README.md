# Hypergraph-MLP

This is the repo for our paper: [Hypergraph-MLP: Learning on Hypergraphs without Message Passing](https://github.com/tbh-98/Hypergraph-MLP).

## Enviroment requirement:

conda create -n "hgmlp" python=3.7
conda activate hgmlp
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-sparse==0.6.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-cluster==1.5.2 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-geometric==1.6.3 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install ipdb
pip install tqdm
pip install scipy
pip install matplotlib

## Generate dataset from raw data.

To generate a dataset for training using PyG or DGL, please set up the following three directories:
```
p2root: './data/pyg_data/hypergraph_dataset_updated/'
p2raw: './data/AllSet_all_raw_data/'
p2dgl_data: './data/dgl_data_raw/'
```

Next, unzip the raw data zip file into `p2raw`. The raw data zip file can be found in this [link](https://github.com/jianhao2016/AllSet/tree/main/data/raw_data).

## Overview

This code is based on the official code of LaneGCN ([Paper](https://arxiv.org/pdf/2007.13732.pdf); [Github](https://github.com/uber-research/LaneGCN)). 

A quick summary of different folders:

- Single Modal contains the source code for the model with proposed collaborative uncertainty framework in single-modal trajectory forecasting.

- Multi Modal contains the source code for the model with proposed collaborative uncertainty framework in multi-modal trajectory forecasting.


## Acknowledgement

This code is based on the official code of AllSet ([Paper](https://openreview.net/forum?id=hpBTIv2uy_E); [Github](https://github.com/jianhao2016/AllSet)). Sincere appreciation is extended for their valuable contributions.

## Citation

If you use this code, please cite our paper:

```

```


