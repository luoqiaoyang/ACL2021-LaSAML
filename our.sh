#dataset=amazon
#data_path="data/amazon.json"
#n_train_class=10
#n_val_class=5
#n_test_class=9

# dataset=fewrel
# data_path="data/fewrel.json"
# # data_path="data/fewrel_bert_uncase.json"
# n_train_class=65
# n_val_class=5
# n_test_class=10

#dataset=20newsgroup
#data_path="data/20news.json"
#n_train_class=8
#n_val_class=5
#n_test_class=7

dataset=huffpost
data_path="data/huffpost.json"
n_train_class=20
n_val_class=5
n_test_class=16
n_train_domain=1
n_val_domain=1
n_test_domain=1

# dataset=clinc150
# data_path="data/clinc150.json"
# # # Cross domain = False
# # n_train_class=60
# # n_val_class=15
# # n_test_class=75
# # n_train_domain=1
# # n_val_domain=1
# # n_test_domain=1
# # # Cross domain = True
# n_train_class=60
# n_val_class=15
# n_test_class=75
# n_train_domain=4
# n_val_domain=1
# n_test_domain=5

# dataset=banking77
# data_path="data/banking_data/"
# n_train_class=30
# n_val_class=15
# n_test_class=32
# n_train_domain=1
# n_val_domain=1
# n_test_domain=1

# dataset=arsc
# data_path="data/arsc/"
# n_train_class=19
# n_val_class=4
# n_test_class=4


#dataset=rcv1
#data_path="data/rcv1.json"
#n_train_class=37
#n_val_class=10
#n_test_class=24

# dataset=reuters
# data_path="data/reuters.json"
# n_train_class=15
# n_val_class=5
# n_test_class=11

pretrained_bert='bert-base-uncased'
bert_cache_dir='~/.pytorch_pretrained_bert/'

if [ "$dataset" = "fewrel" ]; then
    python src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 10 \
        --mode train \
        --embedding ebdnew \
        --classifier proto \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --auxiliary pos \
        --bert \
        --pretrained_bert $pretrained_bert \
        --bert_cache_dir $bert_cache_dir \
        --finetune_ebd \
        --use_pretransformer \
        --lr 2e-5 \
        --notqdm
else
    CUDA_VISIBLE_DEVICES=0 python src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 4 \
        --mode train \
        --embedding ebdnew \
        --classifier proto \
        --induct_hidden_dim 50 \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --n_train_domain=$n_train_domain \
        --n_val_domain=$n_val_domain \
        --n_test_domain=$n_test_domain \
        --bert \
        --pretrained_bert $pretrained_bert \
        --bert_cache_dir $bert_cache_dir \
        --finetune_ebd \
        --sup_feature cls \
        --que_feature cls \
        --lr 2e-5 \
        --seed 330 \
        --addCtagSup none \
        --addCtagQue none \
        --notqdm
fi

    # CUDA_VISIBLE_DEVICES=0  python src/main.py \
    #     --cuda 0 \
    #     --way 10 \
    #     --shot 1 \
    #     --query 15 \
    #     --mode train \
    #     --embedding meta \
    #     --classifier r2d2 \
    #     --dataset=$dataset \
    #     --data_path=$data_path \
    #     --n_train_class=$n_train_class \
    #     --n_val_class=$n_val_class \
    #     --n_test_class=$n_test_class \
    #     --meta_iwf \
    #     --meta_w_target \
    #     --lr 1e-3 \
    #     --notqdm

    # CUDA_VISIBLE_DEVICES=0  python src/main.py \
    #     --cuda 0 \
    #     --way 10 \
    #     --shot 5 \
    #     --query 2 \
    #     --mode train \
    #     --embedding meta \
    #     --cross_domain \
    #     --classifier r2d2 \
    #     --dataset=$dataset \
    #     --data_path=$data_path \
    #     --n_train_class=$n_train_class \
    #     --n_val_class=$n_val_class \
    #     --n_test_class=$n_test_class \
    #     --meta_iwf \
    #     --meta_w_target \
    #     --lr 1e-3 \
    #     --notqdm

        # CUDA_VISIBLE_DEVICES=0 python src/main.py \
        # --cuda 0 \
        # --way 5 \
        # --shot 1 \
        # --query 15 \
        # --mode train \
        # --embedding ebdnew \
        # --classifier r2d2 \
        # --dataset=$dataset \
        # --data_path=$data_path \
        # --n_train_class=$n_train_class \
        # --n_val_class=$n_val_class \
        # --n_test_class=$n_test_class \
        # --n_train_domain=$n_train_domain \
        # --n_val_domain=$n_val_domain \
        # --n_test_domain=$n_test_domain \
        # --bert \
        # --pretrained_bert $pretrained_bert \
        # --bert_cache_dir $bert_cache_dir \
        # --finetune_ebd \
        # --sup_feature cls \
        # --lr 2e-5 \
        # --seed 330 \
        # --addCtagSup one \
        # --use_alllabel_feature \
        # --alllabel_append special \
        # --alllabel_process lmbdaadd \
        # --lmbd_init 0.5 \
        # --clsTagSep . \
        # --notqdm


        # CUDA_VISIBLE_DEVICES=0 python src/main.py \
        # --cuda 0 \
        # --way 5 \
        # --shot 1 \
        # --query 15 \
        # --mode train \
        # --embedding ebdnew \
        # --classifier r2d2 \
        # --dataset=$dataset \
        # --data_path=$data_path \
        # --n_train_class=$n_train_class \
        # --n_val_class=$n_val_class \
        # --n_test_class=$n_test_class \
        # --n_train_domain=$n_train_domain \
        # --n_val_domain=$n_val_domain \
        # --n_test_domain=$n_test_domain \
        # --bert \
        # --pretrained_bert $pretrained_bert \
        # --bert_cache_dir $bert_cache_dir \
        # --finetune_ebd \
        # --sup_feature cls \
        # --lr 2e-5 \
        # --seed 330 \
        # --addCtagSup none \
        # --notqdm

        # CUDA_VISIBLE_DEVICES=0 python src/main.py \
        # --cuda 0 \
        # --way 10 \
        # --shot 5 \
        # --query 2 \
        # --mode train \
        # --embedding ebdnew \
        # --classifier mbc \
        # --induct_hidden_dim 50 \
        # --dataset=$dataset \
        # --data_path=$data_path \
        # --n_train_class=$n_train_class \
        # --n_val_class=$n_val_class \
        # --n_test_class=$n_test_class \
        # --n_train_domain=$n_train_domain \
        # --n_val_domain=$n_val_domain \
        # --n_test_domain=$n_test_domain \
        # --bert \
        # --pretrained_bert $pretrained_bert \
        # --bert_cache_dir $bert_cache_dir \
        # --finetune_ebd \
        # --sup_w_diff \
        # --sup_feature cls \
        # --que_feature cls \
        # --lr 2e-5 \
        # --lr2 1e-3 \
        # --seed 330 \
        # --addCtagSup one \
        # --notqdm

        # CUDA_VISIBLE_DEVICES=0 python src/main.py \
        # --cuda 0 \
        # --way 15 \
        # --shot 5 \
        # --query 2 \
        # --mode train \
        # --embedding ebdnew \
        # --classifier mbc \
        # --induct_hidden_dim 50 \
        # --dataset=$dataset \
        # --data_path=$data_path \
        # --n_train_class=$n_train_class \
        # --n_val_class=$n_val_class \
        # --n_test_class=$n_test_class \
        # --n_train_domain=$n_train_domain \
        # --n_val_domain=$n_val_domain \
        # --n_test_domain=$n_test_domain \
        # --bert \
        # --pretrained_bert $pretrained_bert \
        # --bert_cache_dir $bert_cache_dir \
        # --finetune_ebd \
        # --sup_feature cls \
        # --que_feature all \
        # --lr 2e-5 \
        # --lr2 2e-5 \
        # --seed 330 \
        # --addCtagSup one \
        # --addCtagQue all \
        # --zero_shot \
        # --zero_comb ct \
        # --notqdm