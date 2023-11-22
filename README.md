# Hypergraph-MLP

This is the repo for our paper: [Hypergraph-MLP: Learning on Hypergraphs without Message Passing](https://github.com/tbh-98/Hypergraph-MLP). This code is based on the official code of AllSet ([Paper](https://openreview.net/forum?id=hpBTIv2uy_E); [Github](https://github.com/jianhao2016/AllSet)).



## Enviroment requirement:
This repo is tested with the following enviroment, higher version of torch PyG may also be compatible. 

First let's setup a conda enviroment
```
conda create -n "AllSet" python=3.7
conda activate AllSet
```

Then install pytorch and PyG packages with specific version.
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-sparse==0.6.0 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-cluster==1.5.2 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-geometric==1.6.3 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
```
Finally, install some relative packages

```
pip install ipdb
pip install tqdm
pip install scipy
pip install matplotlib
```

## Generate dataset from raw data.

To generate PyG or DGL dataset for training, please create the following three folders:
```
p2root: './data/pyg_data/hypergraph_dataset_updated/'
p2raw: './data/AllSet_all_raw_data/'
p2dgl_data: './data/dgl_data_raw/'
```

And then unzip the raw data zip file into `p2raw`.
