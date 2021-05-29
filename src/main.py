import os
import sys
import pickle
import signal
import argparse
import traceback

import torch
import numpy as np
import random

import embedding.factory as ebd
import classifier.factory as clf
import dataset.loader as loader
import train.factory as train_utils


def parse_args():
    parser = argparse.ArgumentParser(
            description="Few Shot Text Classification with Distributional Signatures")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="data/reuters.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="reuters",
                        help="name of the dataset. "
                        "Options: [20newsgroup, amazon, huffpost, "
                        "reuters, rcv1, fewrel]")
    parser.add_argument("--n_train_class", type=int, default=15,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=11,
                        help="number of meta-test classes")
    # extra arguments for domain configuration
    parser.add_argument("--n_train_domain", type=int, default=1,
                        help="number of meta-train domains")
    parser.add_argument("--n_val_domain", type=int, default=1,
                        help="number of meta-val domains")
    parser.add_argument("--n_test_domain", type=int, default=1,
                        help="number of meta-test domains")

    # extra arguments for data process
    parser.add_argument("--max_text_len_limits", type=int, default=24,
                        help="max number of tokens for each sample")
    parser.add_argument("--cross_domain", action="store_true", default=False,
                        help=("whether train and test the cross-domain ability of the model"))

    # extra arguments for Probing Word configurations
    parser.add_argument("--addCtagSup", type=str, default="none",
                        help="none: add nothing for support sample"
                        "one: add class relevant tag" 
                        "all: add all class tags"
                        "pretransformer: add label difference features after pretransformer")
    parser.add_argument("--addCtagQue", type=str, default="none",
                        help="none: add nothing for query sample"
                        "copy: copy query sample number of Way times and add each class tag" 
                        "all: add all class tags")
    parser.add_argument("--clsTagSep", type=str, default=".",
                        help="none: add nothing for splitting the labels"
                        ",: add 1010" 
                        ".: add 1012"
                        "sep: add 102")

    # extra arguments for BERT
    parser.add_argument("--fixbert", type=str, default="no",
                        help="no: do not fix bert"
                        "ebd: fix bert ebd" 
                        "encoder: fix bert encoder"
                        "all: fix all parameters in bert")
    parser.add_argument("--sup_feature", type=str, default="cls",
                        help="cls: get cls as the feature"
                        "sent: get avg of sentence feature" 
                        "rtag: get avg of relevant tag feature")
    parser.add_argument("--que_feature", type=str, default="cls",
                        help="cls: get cls as the feature"
                        "sent: get avg of sentence feature" 
                        "atag: get all avg of tag features")
    parser.add_argument("--use_alllabel_feature", action="store_true", default=False,
                        help=("whether train BERT with all label features, only use for support data"))
    parser.add_argument("--alllabel_append", type=str, default="special",
                        help="none: append nothing on all label feature"
                        "special: append cls, sep and class tag split")
    parser.add_argument("--alllabel_process", type=str, default="lmbdaadd",
                        help="lmbdaadd: add each label feature with weight lmbda with text feature (1-lmbda)"
                        "dynamicadd: use parameters to add each label feature with text feature in each dimension"
                        "concat: concat two type of features and then process with MLP to reduce the dimension"
                        "casecade: cascade two type of features and then process with post transformer")
    parser.add_argument("--lmbd_init", default=0.5, type=float,
                        help="percent of train data to allocate for val")
    parser.add_argument("--use_current_label_embedding", action="store_true", default=False,
                        help=("whether train BERT with current label embedding"))
    parser.add_argument("--sup_w_diff", action="store_true", default=False,
                        help=("whether support features minus other support features"))
    
    # extra arguments for PreTransformer
    parser.add_argument("--use_pretransformer", action="store_true", default=False,
                        help=("whether use pretransformer to get label feature difference"))
    parser.add_argument("--pretransformer_mode", type=str, default="label",
                        help="label: use pretransformer to process label only"
                        "all: use pretransformer to process all input")
    parser.add_argument("--nhead_pre", type=int, default=8,
                        help="number of head in each transformer encoder layer")
    parser.add_argument("--num_layers_pre", type=int, default=1,
                        help="number of encoder layers in each transformer")
    parser.add_argument("--label_mode_att_type", type=str, default="extract",
                        help="extract: extract corresponded label representation"
                        "ebd: use intput attention embedding to pay attention to current class"
                        "minus: use minus operation to get current label representation")
    parser.add_argument("--add_inputTypeEbd_pre", action="store_true", default=False,
                        help=("whether add input type embedding"))
    parser.add_argument("--use_mlp_sup", action="store_true", default=False,
                        help=("whether support data use mlp to process embedding after BERT"))           
    
    # extra arguments for PostTransformer
    parser.add_argument("--use_posttransformer_sup", action="store_true", default=False,
                        help=("whether support data use posttransformer to process embedding after BERT"))
    parser.add_argument("--use_posttransformer_que", action="store_true", default=False,
                        help=("whether query data use posttransformer to process embedding after BERT"))
    parser.add_argument("--use_posttransformer_lab", action="store_true", default=False,
                        help=("whether labels data use posttransformer to process embedding after BERT"))
    parser.add_argument("--use_transformer_alllabs", action="store_true", default=False,
                        help=("whether use transformer to process all labels w.o BERT"))
    parser.add_argument("--add_inputTypeEbd_post", action="store_true", default=False,
                        help=("whether use pretransformer to get label feature difference"))
    parser.add_argument("--nhead_post", type=int, default=8,
                        help="number of head in each transformer encoder layer")
    parser.add_argument("--nhead_div", type=int, default=8,
                        help="number of head in each transformer encoder layer")
    parser.add_argument("--dense_div", type=int, default=2048,
                        help="number of head in each transformer encoder layer")
    parser.add_argument("--num_layers_post", type=int, default=1,
                        help="number of encoder layers in each transformer")
    
    

    # extra arguments for Metric-based Classifier
    parser.add_argument("--sim", type=str, default="l2",
                        help="l2: use l2 distance"
                        "cos: use cosine distance")
    parser.add_argument("--zero_comb", type=str, default="ct",
                        help="l2: use l2 distance"
                        "cos: use cosine distance")

    # load bert embeddings for sent-level datasets (optional)
    parser.add_argument("--n_workers", type=int, default=10,
                        help="Num. of cores used for loading data. Set this "
                        "to zero if you want to use all the cpus.")
    parser.add_argument("--bert", default=False, action="store_true",
                        help=("set true if use bert embeddings "
                              "(only available for sent-level datasets: "
                              "huffpost, fewrel"))
    parser.add_argument("--bert_cache_dir", default=None, type=str,
                        help=("path to the cache_dir of transformers"))
    parser.add_argument("--pretrained_bert", default=None, type=str,
                        help=("path to the pre-trained bert embeddings."))

    # task configuration
    parser.add_argument("--way", type=int, default=5,
                        help="#classes for each task")
    parser.add_argument("--shot", type=int, default=5,
                        help="#support examples for each class for each task")
    parser.add_argument("--query", type=int, default=25,
                        help="#query examples for each class for each task")

    # train/test configuration
    # 20, 10, 20, 1000 for tars
    parser.add_argument("--train_epochs", type=int, default=50,
                        help="max num of training epochs")
    parser.add_argument("--train_episodes", type=int, default=100,
                        help="#tasks sampled during each training epoch")
    parser.add_argument("--val_episodes", type=int, default=100,
                        help="#asks sampled during each validation epoch")
    parser.add_argument("--test_episodes", type=int, default=1000,
                        help="#tasks sampled during each testing epoch")

    # settings for finetuning baseline
    parser.add_argument("--finetune_loss_type", type=str, default="softmax",
                        help="type of loss for finetune top layer"
                        "options: [softmax, dist]")
    parser.add_argument("--finetune_maxepochs", type=int, default=5000,
                        help="number epochs to finetune each task for (inner loop)")
    parser.add_argument("--finetune_episodes", type=int, default=10,
                        help="number tasks to finetune for (outer loop)")
    parser.add_argument("--finetune_split", default=0.8, type=float,
                        help="percent of train data to allocate for val"
                             "when mode is finetune")

    # model options
    parser.add_argument("--embedding", type=str, default="avg",
                        help=("document embedding method. Options: "
                              "[avg, tfidf, meta, oracle, cnn]"))
    parser.add_argument("--classifier", type=str, default="nn",
                        help=("classifier. Options: [nn, proto, r2d2, mlp]"))
    parser.add_argument("--auxiliary", type=str, nargs="*", default=[],
                        help=("auxiliary embeddings (used for fewrel). "
                              "Options: [pos, ent]"))

    # cnn configuration
    parser.add_argument("--cnn_num_filters", type=int, default=50,
                        help="Num of filters per filter size [default: 50]")
    parser.add_argument("--cnn_filter_sizes", type=int, nargs="+",
                        default=[3, 4, 5],
                        help="Filter sizes [default: 3]")

    # nn configuration
    parser.add_argument("--nn_distance", type=str, default="l2",
                        help=("distance for nearest neighbour. "
                              "Options: l2, cos [default: l2]"))

    # proto configuration
    parser.add_argument("--proto_hidden", nargs="+", type=int,
                        default=[300, 300],
                        help=("hidden dimension of the proto-net"))

    # maml configuration
    parser.add_argument("--maml", action="store_true", default=False,
                        help=("Use maml or not. "
                        "Note: maml has to be used with classifier=mlp"))
    parser.add_argument("--mlp_hidden", nargs="+", type=int, default=[300, 5],
                        help=("hidden dimension of the proto-net"))
    parser.add_argument("--maml_innersteps", type=int, default=10)
    parser.add_argument("--maml_batchsize", type=int, default=10)
    parser.add_argument("--maml_stepsize", type=float, default=1e-1)
    parser.add_argument("--maml_firstorder", action="store_true", default=False,
                        help="truncate higher order gradient")

    # lrd2 configuration
    parser.add_argument("--lrd2_num_iters", type=int, default=5,
                        help=("num of Newton steps for LRD2"))

    # induction networks configuration
    parser.add_argument("--induct_rnn_dim", type=int, default=128,
                        help=("Uni LSTM dim of induction network's encoder"))
    parser.add_argument("--induct_hidden_dim", type=int, default=100,
                        help=("tensor layer dim of induction network's relation"))
    parser.add_argument("--induct_iter", type=int, default=3,
                        help=("num of routings"))
    parser.add_argument("--induct_att_dim", type=int, default=64,
                        help=("attention projection dim of induction network"))

    # aux ebd configuration (for fewrel)
    parser.add_argument("--pos_ebd_dim", type=int, default=5,
                        help="Size of position embedding")
    parser.add_argument("--pos_max_len", type=int, default=40,
                        help="Maximum sentence length for position embedding")

    # base word embedding
    parser.add_argument("--wv_path", type=str,
                        default="./",
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default="wiki.en.vec",
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", action="store_true", default=False,
                        help=("Finetune embedding during meta-training"))

    # options for the distributional signatures
    parser.add_argument("--meta_idf", action="store_true", default=False,
                        help="use idf")
    parser.add_argument("--meta_iwf", action="store_true", default=False,
                        help="use iwf")
    parser.add_argument("--meta_w_target", action="store_true", default=False,
                        help="use target importance score")
    parser.add_argument("--meta_w_target_lam", type=float, default=1,
                        help="lambda for computing w_target")
    parser.add_argument("--meta_target_entropy", action="store_true", default=False,
                        help="use inverse entropy to model task-specific importance")
    parser.add_argument("--meta_ebd", action="store_true", default=False,
                        help="use word embedding into the meta model "
                        "(showing that revealing word identity harm performance)")

    # training options
    parser.add_argument("--seed", type=int, default=330, help="seed")
    parser.add_argument("--dropout", type=float, default=0.1, help="drop rate")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--lr2", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--patience", type=int, default=10, help="patience")
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient clipping")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="cuda device, -1 for cpu")
    parser.add_argument("--mode", type=str, default="test",
                        help=("Running mode."
                              "Options: [train, test, finetune]"
                              "[Default: test]"))
    parser.add_argument("--save", action="store_true", default=False,
                        help="train the model")
    parser.add_argument("--notqdm", action="store_true", default=False,
                        help="disable tqdm")
    parser.add_argument("--result_path", type=str, default="")
    parser.add_argument("--snapshot", type=str, default="",
                        help="path to the pretraiend weights")

    return parser.parse_args()


def print_args(args):
    """
        Print arguments (only show the relevant arguments)
    """
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        if args.embedding != "cnn" and attr[:4] == "cnn_":
            continue
        if args.classifier != "proto" and attr[:6] == "proto_":
            continue
        if args.classifier != "nn" and attr[:3] == "nn_":
            continue
        if args.embedding != "meta" and attr[:5] == "meta_":
            continue
        if args.embedding != "cnn" and attr[:4] == "cnn_":
            continue
        if args.classifier != "mlp" and attr[:4] == "mlp_":
            continue
        if args.classifier != "proto" and attr[:6] == "proto_":
            continue
        if "pos" not in args.auxiliary and attr[:4] == "pos_":
            continue
        if not args.maml and attr[:5] == "maml_":
            continue
        print("\t{}={}".format(attr.upper(), value))
    print("""
    (Credit: Maija Haavisto)                        /
                                 _,.------....___,.' ',.-.
                              ,-'          _,.--'        |
                            ,'         _.-'              .
                           /   ,     ,'                   `
                          .   /     /                     ``.
                          |  |     .                       \.\\
                ____      |___._.  |       __               \ `.
              .'    `---''       ``'-.--''`  \               .  \\
             .  ,            __               `              |   .
             `,'         ,-''  .               \             |    L
            ,'          '    _.'                -._          /    |
           ,`-.    ,'.   `--'                      >.      ,'     |
          . .'\\'   `-'       __    ,  ,-.         /  `.__.-      ,'
          ||:, .           ,'  ;  /  / \ `        `.    .      .'/
          j|:D  \          `--'  ' ,'_  . .         `.__, \   , /
         / L:_  |                 .  '' :_;                `.'.'
         .    '''                  ''''''                    V
          `.                                 .    `.   _,..  `
            `,_   .    .                _,-'/    .. `,'   __  `
             ) \`._        ___....----''  ,'   .'  \ |   '  \  .
            /   `. '`-.--''         _,' ,'    use_posttransformer_lab `---' |    `./  |
           .   _  `'''--.._____..--'   ,             '         |
           | .' `. `-.                /-.           /          ,
           | `._.'    `,_            ;  /         ,'          .
          .'          /| `-.        . ,'         ,           ,
          '-.__ __ _,','    '`-..___;-...__   ,.'\ ____.___.'
          `'^--'..'   '-`-^-''--    `-^-'`.'''''''`.,^.`.--' mh
    """)


def set_seed(seed):
    """
        Setting random seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()

    print_args(args)

    set_seed(args.seed)


    # load data

    train_data, val_data, test_data, vocab = loader.load_dataset(args)
    
    # initialize model
    model = {}
    model["ebd"] = ebd.get_embedding(vocab, args)
    model["clf"] = clf.get_classifier(model["ebd"].ebd_dim, args)


    if args.mode == "train":
        # train model on train_data, early stopping based on val_data
        train_utils.train(train_data, val_data, model, args)

    elif args.mode == "finetune":
        # sample an example from each class during training
        way = args.way
        query = args.query
        shot = args.shot
        args.query = 1
        args.shot= 1
        args.way = args.n_train_class
        train_utils.train(train_data, val_data, model, args)
        # restore the original N-way K-shot setting
        args.shot = shot
        args.query = query
        args.way = way

    # testing on validation data: only for not finetune
    # In finetune, we combine all train and val classes and split it into train
    # and validation examples.
    if args.mode != "finetune":
        print("start test by val data")
        state = "val"
        val_acc, val_std = train_utils.test(val_data, model, args,
                                            args.val_episodes, state=state)
    else:
        val_acc, val_std = 0, 0

    test_acc, test_std = train_utils.test(test_data, model, args,
                                          args.test_episodes, state="test")

    if args.result_path:
        directory = args.result_path[:args.result_path.rfind("/")]
        if not os.path.exists(directory):
            os.mkdirs(directory)

        result = {
            "test_acc": test_acc,
            "test_std": test_std,
            "val_acc": val_acc,
            "val_std": val_std
        }

        for attr, value in sorted(args.__dict__.items()):
            result[attr] = value

        with open(args.result_path, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        exc_info = sys.exc_info()
        traceback.print_exception(*exc_info)
        os.killpg(0, signal.SIGKILL)

    exit(0)
