python train_time_test.py --method HyperGCN --dname cora --All_num_layers 1 --MLP_num_layers 2 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 64 --HyperGCN_mediators --HyperGCN_fast --wd 0.0 --runs 60000 --feature_noise 0.0 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HyperGCN --dname citeseer --All_num_layers 1 --MLP_num_layers 2 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 64 --HyperGCN_mediators --HyperGCN_fast --wd 0.00001 --runs 60000 --feature_noise 0.0 --cuda 7 --lr 0.01 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HyperGCN --dname pubmed --All_num_layers 1 --MLP_num_layers 2 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 64 --HyperGCN_mediators --HyperGCN_fast --wd 0.00001 --runs 60000 --feature_noise 0.0 --cuda 7 --lr 0.01 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HyperGCN --dname coauthor_cora --All_num_layers 1 --MLP_num_layers 2 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 64 --HyperGCN_mediators --HyperGCN_fast --wd 0.00001 --runs 60000 --feature_noise 0.0 --cuda 7 --lr 0.01 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HyperGCN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 2 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 64 --HyperGCN_mediators --HyperGCN_fast --wd 0.00001 --runs 60000 --feature_noise 0.0 --cuda 7 --lr 0.01 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HyperGCN --dname 20newsW100 --All_num_layers 1 --MLP_num_layers 2 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 64 --HyperGCN_mediators --HyperGCN_fast --wd 0.00001 --runs 60000 --feature_noise 0.0 --cuda 7 --lr 0.01 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HyperGCN --dname NTU2012 --All_num_layers 1 --MLP_num_layers 2 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 64 --HyperGCN_mediators --HyperGCN_fast --wd 0.00001 --runs 60000 --feature_noise 0.0 --cuda 7 --lr 0.01 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HyperGCN --dname house-committees-100 --All_num_layers 1 --MLP_num_layers 2 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 64 --HyperGCN_mediators --HyperGCN_fast --wd 0.00001 --runs 60000 --feature_noise 0.6 --cuda 7 --lr 0.01 --perturb_type delete --perturb_prop 0.0

python train_time_test.py --method HGNN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HGNN --dname citeseer --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HGNN --dname coauthor_cora --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 128 --Classifier_hidden 128 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HGNN --dname 20newsW100 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.1 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HGNN --dname NTU2012 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HGNN --dname house-committees-100 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.6 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HGNN --dname pubmed --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HGNN --dname cora --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0

python train_time_test.py --method UniGCNII --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 8 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 128 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method UniGCNII --dname citeseer --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 128 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method UniGCNII --dname coauthor_cora --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 4 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 128 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method UniGCNII --dname 20newsW100 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 8 --Classifier_num_layers 1 --MLP_hidden 128 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method UniGCNII --dname NTU2012 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 8 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method UniGCNII --dname house-committees-100 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.6 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method UniGCNII --dname cora --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 8 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 128 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method UniGCNII --dname pubmed --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 128 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0

python train_time_test.py --method HCHA --HCHA_symdegnorm --dname cora --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HCHA --HCHA_symdegnorm --dname citeseer --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 128 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HCHA --HCHA_symdegnorm --dname pubmed --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HCHA --HCHA_symdegnorm --dname coauthor_cora --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 128 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HCHA --HCHA_symdegnorm --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HCHA --HCHA_symdegnorm --dname 20newsW100 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 64 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.1 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HCHA --HCHA_symdegnorm --dname NTU2012 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.0 --heads 1 --Classifier_num_layers 1 --MLP_hidden 256 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0
python train_time_test.py --method HCHA --HCHA_symdegnorm --dname house-committees-100 --All_num_layers 1 --MLP_num_layers 2 --feature_noise 0.6 --heads 1 --Classifier_num_layers 1 --MLP_hidden 512 --Classifier_hidden 256 --wd 0.0 --runs 60000 --cuda 7 --lr 0.001 --perturb_type delete --perturb_prop 0.0

dataset_list=( cora citeseer coauthor_cora coauthor_dblp 20newsW100 NTU2012 house-committees-100 pubmed )
lr=0.001
wd=0
cuda=0

runs=60000

for dname in in ${dataset_list[*]} 
do
    if [ "$dname" = "cora" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "citeseer" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "pubmed" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_cora" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_dblp" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "zoo" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 64 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "20newsW100" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "Mushroom" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "NTU2012" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "ModelNet40" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "yelp" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 128 \
            --Classifier_hidden 64 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "house-committees-100" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.6 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 128 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname} with noise 0.6"
    elif [ "$dname" = "walmart-trips-100" ]; then
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 1.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname} with noise 1.0"
        
        echo =============
        echo ">>>>  Model:AllDeepSets (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllDeepSets \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.6 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname} with noise 0.6"   
    fi
done


echo "Finished all training for AllDeepSets!"

dataset_list=( cora citeseer coauthor_cora coauthor_dblp 20newsW100 NTU2012 house-committees-100 pubmed )
lr=0.001
wd=0
cuda=0

runs=60000

for dname in in ${dataset_list[*]} 
do
    if [ "$dname" = "cora" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 4 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "citeseer" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "pubmed" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_cora" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 128 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "coauthor_dblp" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "zoo" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 64 \
            --Classifier_hidden 64 \
            --wd 0.00001 \
            --runs $runs \
            --cuda $cuda \
            --lr 0.01
        echo "Finished training on ${dname}"
    elif [ "$dname" = "20newsW100" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "Mushroom" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 128 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "NTU2012" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "ModelNet40" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "yelp" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.0 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 64 \
            --Classifier_hidden 64 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname}"
    elif [ "$dname" = "house-committees-100" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.6 \
            --heads 1 \
            --Classifier_num_layers 1 \
            --MLP_hidden 512 \
            --Classifier_hidden 256 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname} with noise 0.6"
    elif [ "$dname" = "walmart-trips-100" ]; then
        echo =============
        echo ">>>>  Model:AllSetTransformer (default), Dataset: ${dname}"  
        python train_time_test.py \
            --method AllSetTransformer \
            --dname $dname \
            --All_num_layers 1 \
            --MLP_num_layers 2 \
            --feature_noise 0.6 \
            --heads 8 \
            --Classifier_num_layers 1 \
            --MLP_hidden 256 \
            --Classifier_hidden 128 \
            --wd 0.0 \
            --runs $runs \
            --cuda $cuda \
            --lr $lr
        echo "Finished training on ${dname} with noise 0.6"   
    fi
done


echo "Finished all training for AllSetTransformer!"