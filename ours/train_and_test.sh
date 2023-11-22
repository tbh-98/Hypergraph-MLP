# Hypergraph-MLP
python train.py --method MLP --dname 20newsW100 --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0.05
python train.py --method MLP --dname cora --All_num_layers 5 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0.01
python train.py --method MLP --dname citeseer --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0.5
python train.py --method MLP --dname NTU2012 --All_num_layers 4 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0.05
python train.py --method MLP --dname pubmed --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0.05
python train.py --method MLP --dname coauthor_dblp --All_num_layers 5 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 1000 --runs 20 --cuda 0 --lr 0.001 --alpha 0.02
python train.py --method MLP --dname house-committees-100 --All_num_layers 3 --feature_noise 0.6 --MLP_hidden 512 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.0001 --alpha 0.1

# MLP
python train.py --method MLP --dname 20newsW100 --All_num_layers 3 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0
python train.py --method MLP --dname cora --All_num_layers 5 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0
python train.py --method MLP --dname citeseer --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0
python train.py --method MLP --dname NTU2012 --All_num_layers 4 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0
python train.py --method MLP --dname pubmed --All_num_layers 2 --feature_noise 0.0 --MLP_hidden 256 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.001 --alpha 0
python train.py --method MLP --dname coauthor_dblp --All_num_layers 5 --feature_noise 0.0 --MLP_hidden 512 --wd 0.0 --epochs 1000 --runs 20 --cuda 0 --lr 0.001 --alpha 0
python train.py --method MLP --dname house-committees-100 --All_num_layers 3 --feature_noise 0.6 --MLP_hidden 512 --wd 0.0 --epochs 500 --runs 20 --cuda 0 --lr 0.0001 --alpha 0
