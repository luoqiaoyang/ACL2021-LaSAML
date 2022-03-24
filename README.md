# ACL2021-LaSAML
The repo for ACL2021 findings paper - [Don't Miss the Labels: Label-semantic Argumented Meta-Learner for Few-Shot Text Classification](https://aclanthology.org/2021.findings-acl.245.pdf)



## Overview

### Abstract

Increasing studies leverage pre-trained language models and meta-learning frameworks to solve few-shot text classification problems. Most of the current studies focus on building a meta-learner from the information of input texts but ignore abundant semantic information beneath class labels. In this work, we show that class-label information can be utilized for extracting more discriminative feature representation of the input text from a pretrained language model like BERT, and can achieve a performance boost when the samples are scarce. Building on top of this discovery, we propose a framework called Labelsemantic augmented meta-learner (LaSAML) to make full use of label semantics. We systematically investigate various factors in this framework and show that it can be plugged into the existing few-shot text classification system. Through extensive experiments, we demonstrate that the few-shot text classification system upgraded by LaSAML can lead to significant performance improvement over its original counterparts.



### Motivation

**Banking**

Class 1: *can you give me a hand paying my water bill*

Class 2: *tell me the last day I can pay my gas bill*

**Travel**

Class 3: *what are some fun activities to do in Colorado*

Class 4: *tell me about any travel alerts issued for Germany*

Considering the two groups of examples above. These four samples belonging to different intent classes, giving only one sample for each class without the definition of its label, it may cause ambiguity even for human to understand the key semantic difference behind these samples. 

However, this ambiguity can be easily resolved if the class definition or simply the class name is provided for human. According to this, we made an interesting observation that the BERT will extract more discriminative features if we append the class name to the input sentence. Motivated by the observation, this work explores **how to better leverage the semantic information beneath class names for few-shot learning**. Our key idea is to use **meta-learning to encourage the features extracted from class-name-appended samples to be more class-relevant** and compatible to the query features. Please find more details in our [paper](https://aclanthology.org/2021.findings-acl.245.pdf). 

If you find that this project helps your research, please consider citing the related paper:

```
@inproceedings{luo2021don,
  title={Don’t Miss the Labels: Label-semantic Augmented Meta-Learner for Few-Shot Text Classification},
  author={Luo, Qiaoyang and Liu, Lingqiao and Lin, Yuhao and Zhang, Wei},
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  pages={2773--2782},
  year={2021}
}
```



## Code

### Step 1. Download the Model

Please go to the [huggingface ](https://huggingface.co/models) to download the BERT Base model and change the `bert_cache_dir` into the BERT folder in `our.sh`.



### Step 2. Train and Test with different Settings

**Our model - LaSAML-PN** has lots of variations based on 

1\. whether appending label features on support data, query data 

```
# v1. each support sample appends its corresponding label, query samples remain unchanged
--addCtagSup one \
--addCtagQue none \
# v2. each support sample appends its corresponding label, query samples appends all class labels
--addCtagSup one \
--addCtagQue all \
```

you can also try `--addCtagSup all` which appends all class labels to each samples, but may not get good results as just appending the corresponding label for each sample. **Making sure `addCtagSup` cannot be none for LaSAML** 

2\. whether extracting sample features from `CLS` token, from appended label, from whole sentence or combination sets of them.

```
# v1. both sup features and query features are extracted from CLS token
--sup_feature cls \
--que_feature cls \
# v2. support features extracted from the mean pooling of entire support sentence, query features from CLS token
--sup_feature sent \
--que_feature cls \
# v3. support features extracted from the mean pooling of appended label, query features from CLS token
--sup_feature tag \
--que_feature cls \
# v3 support features from CLS + sentence, query features from CLS token
--sup_feature comb_cs \
--que_feature cls \
# v4. support features from CLS + appended label, query features from CLS token
--sup_feature comb_ct \
--que_feature cls \
# v5. support features from sentence + appended label, query features from CLS token
--sup_feature comb_st \
--que_feature cls \
# v6. support features from CLS + appended label + sentence, query features from CLS token
--sup_feature comb_all \
--que_feature cls \
# v7. support features from CLS + appended label + sentence, then processing it with MLP, query features from CLS token
--sup_feature mlp_all \
--que_feature cls \
# v8. support features from CLS + appended label + sentence, then processing it with MLP and softmax, query features from CLS token
--sup_feature comb_att \
--que_feature cls \
```

for the query feature,  you may also choose from sentence or all labels.

the following setting is for the result of LaSAML-PN in table 2 of the paper

```
--classifier mbc \
--sup_feature cls \
--que_feature cls \
--addCtagSup one \
--addCtagQue none \
```



**Baseline models** do not use appended labels in both support data and query data, so change the `addCtagSup` and `addCtagQue` to none

```
--addCtagSup none \
--addCtagQue none \
```



### Note

1. Different initialisations will cause slight difference in final performance.
2. Choosing larger `query` size sometimes leads better performance.






Reference Repo: [Distributional-Signatures](https://github.com/YujiaBao/Distributional-Signatures)
