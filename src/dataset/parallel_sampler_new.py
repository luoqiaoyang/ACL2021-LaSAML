import time
import datetime
from multiprocessing import Process, Queue, cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from pytorch_transformers import BertModel
from transformers import BertModel, BertTokenizer

import dataset.utils as utils
import dataset.stats as stats



class ParallelSamplerNew():
    def __init__(self, data, args, num_episodes=None, state="train"):
        """
            Input Type Indicator:
            - 0: Padding
            - 1: CLS
            - 2: SEP
            - 3: Sentence
            - 4: class tag spliter, e.g. ",", "."
            - 5 - n: class tag / class feature
        """
        self.data = data 
        self.args = args
        self.state = state
        self.num_episodes = num_episodes

        self.CLS_id = 1
        self.SEP_id = 2
        self.Sent_id = 3
        self.TSpt_id = 4
        self.CTag_id = 5 # start from 5,

        self.addCtagSup = self.args.addCtagSup # none: add nothing, one: add class relevant tag, all: add all class tags
        self.addCtagQue = self.args.addCtagQue # none: add nothing, all: add all class tags
        self.clsTagSep = self.args.clsTagSep # seperate each class tag, none: add nothing, ",": add 1010, ".": add 1012 
        
        self.crossDomain = self.args.cross_domain # whether pick the cross domain data
        self.copyQuery = False # whether copy query number of classes times
        self.clsTagFormat = "all" # avg: avg ebd of class tag, all: all ebd of all class tag
        # self.addEntities = False
        # self.addCtagebd = False
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.long),
                                  requires_grad=False)

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def _reidx_y(self, YS, YQ):
        '''
            Map the labels into 0,..., way
            @param YS: batch_size
            @param YQ: batch_size

            @return YS_new: batch_size
            @return YQ_new: batch_size
        '''
        unique1, inv_S = torch.unique(YS, sorted=True, return_inverse=True)
        unique2, inv_Q = torch.unique(YQ, sorted=True, return_inverse=True)

        if len(unique1) != len(unique2):
            raise ValueError(
                'Support set classes are different from the query set')

        if len(unique1) != self.args.way:
            raise ValueError(
                'Support set classes are different from the number of ways')

        if int(torch.sum(unique1 - unique2).item()) != 0:
            raise ValueError(
                'Support set classes are different from the query set classes')

        Y_new = torch.arange(start=0, end=self.args.way, dtype=unique1.dtype,
                device=unique1.device)

        return Y_new[inv_S], Y_new[inv_Q]

    
    def _get_sample_from_task_ids(self,randm_task_id,task_data):
        support_examples = []
        query_examples = []
        # task_examples = self.data[randm_task_id]
        
        random_idx = [np.random.permutation(len(task_data[randm_task_id[i]])) for i in range(self.args.way)]
        for i in range(self.args.way):
            support_idx = random_idx[i][0:self.args.shot]
            query_idx = random_idx[i][self.args.shot:self.args.shot+self.args.query]
            support_examples.extend([task_data[randm_task_id[i]][idx] for _, idx in enumerate(support_idx)]) 
            query_examples.extend([task_data[randm_task_id[i]][idx] for _, idx in enumerate(query_idx)])
        return support_examples, query_examples
    
    
    def _convet_examples_to_tensor(self, support_example, query_example):

        support = {}
        query = {}
        sup_raw_token_len_wo = []  # each support sample's token id length w/o CLS and SEP
        que_raw_token_len_wo = []  # each query sample's token id length w/o CLS and SEP
        sup_raw_token_len = []  # each support sample's token id length w CLS and SEP
        que_raw_token_len = []  # each query sample's token id length w CLS and SEP
        sup_token_len = []  # token id length w CLS, SEP and Class Tag(s) of each support sample
        que_token_len = []  # token id length w CLS, SEP and Class Tag(s) of each query sample
        sup_max_len = 0  # default max length of support sample
        que_max_len = 0  # default max length of query sample
        sup_label_ids = torch.zeros(self.args.way*self.args.shot, dtype=torch.long)  # label ids for each support sample
        que_label_ids = torch.zeros(self.args.way*self.args.query, dtype=torch.long)  # label ids for each query sample
        sup_current_label_len = torch.zeros(self.args.way*self.args.shot, dtype=torch.float) # label length for each support sample
        que_current_label_len = torch.zeros(self.args.way*self.args.query, dtype=torch.float) # label length for each query sample

        label_token_len = []  # each label token len
        total_label_token_len = 0  # all labels token len in total
        

        all_label_token_ids = []
        all_label_raw = []
        labels_tokens_list = []
        # get all unique labels and max length of support data
        
        
        for i in range(len(support_example)):
            # 1. get all unique labels and length
            if i%self.args.shot == 0:
                all_label_token_ids.append(support_example[i].label_token_id)
                labels_tokens_list.extend(support_example[i].label_token_id)
                all_label_raw.append(support_example[i].label_raw)
                tmp_label_len = len(support_example[i].label_token_id)
                label_token_len.append(tmp_label_len)
                # if self.clasTagSep == none, there is no split symbol among each labels when group all class tag together
                total_label_token_len = total_label_token_len + tmp_label_len if self.clsTagSep == "none" else total_label_token_len + tmp_label_len + 1
            
            # 2. calculate the max length of support sample
            tmp_text_len = len(support_example[i].text_token_id)
            if self.addCtagSup != "one":
                sup_max_len = sup_max_len if sup_max_len > tmp_text_len else tmp_text_len
            else:
                # tmp length: each support sample length + related label length + 1 for "SEP" token
                tmp_total_len = tmp_text_len+len(support_example[i].label_token_id)+1
                sup_max_len = sup_max_len if sup_max_len > tmp_total_len else tmp_total_len
            # 3. get the each text length of support sample
            sup_raw_token_len_wo.append(tmp_text_len - 2) # length exclude CLS and SEP
            sup_raw_token_len.append(tmp_text_len)
            sup_current_label_len[i] = len(support_example[i].label_token_id)
            # 4. construct the label id for support             
            sup_label_ids[i] = support_example[i].label_id
        
        # number of classes of class (difference) features, 1 for SEP features
        if self.addCtagSup == "all":
            sup_max_len = sup_max_len + total_label_token_len + 1
        elif self.addCtagSup == "pretransformer":
            sup_max_len = sup_max_len + 2

        # construct tensors for support data
        support_tokens = torch.zeros(self.args.way*self.args.shot, sup_max_len, dtype=torch.long)
        support_token_type_ids = torch.zeros(self.args.way*self.args.shot, sup_max_len, dtype=torch.long)
        support_current_label_ids = torch.zeros(self.args.way*self.args.shot, sup_max_len, dtype=torch.long) # 1 for label position, 0 for rest
        support_current_sent_ids = torch.zeros(self.args.way*self.args.shot, sup_max_len, dtype=torch.long) # 1 for text position, 0 for rest


        for i, sample in enumerate(support_example):
            tmp_text_len = sup_raw_token_len[i]
            # Basic Input format: [CLS] + Sentence + [SEP]
            # Input type:           1,       3...,     2
            support_tokens[i][0:tmp_text_len] = torch.tensor(sample.text_token_id, dtype=torch.long)
            support_token_type_ids[i][0] = self.CLS_id # CLS token type id: 1
            support_token_type_ids[i][1:tmp_text_len-1] = self.Sent_id  # text token type id: 3
            support_token_type_ids[i][tmp_text_len-1] = self.SEP_id  # SEP token type id: 2
            support_current_sent_ids[i][1:tmp_text_len-1] = 1
            if self.addCtagSup == "none":
                sup_token_len.append(tmp_text_len)
            elif self.addCtagSup == "pretransformer":
                type_indicator = 5
                support_token_type_ids[i][tmp_text_len] = type_indicator
                support_token_type_ids[i][tmp_text_len+1] = self.SEP_id
                support_current_label_ids[i][tmp_text_len] = 1
                sup_token_len.append(tmp_text_len+2)
            elif self.addCtagSup == "one":
                # Input format: [CLS] + Sentence + [SEP] + Class Tag + [SEP]
                # Input type:   1,      3...,       2,      5...,        2
                type_indicator = 5
                tmp_label_len = len(sample.label_token_id)
                support_tokens[i][tmp_text_len: tmp_text_len+tmp_label_len] = torch.tensor(sample.label_token_id, dtype=torch.long)
                support_tokens[i][tmp_text_len+tmp_label_len] = 102 # SEP for the last token
                support_token_type_ids[i][tmp_text_len: tmp_text_len+tmp_label_len] = type_indicator
                support_token_type_ids[i][tmp_text_len+tmp_label_len] = self.SEP_id
                support_current_label_ids[i][tmp_text_len: tmp_text_len+tmp_label_len] = 1
                sup_token_len.append(tmp_text_len+tmp_label_len+1)
            else:
                # Input format: [CLS] + Sentence + [SEP] + Class Tag1 , Class Tag2 , Class Tag 3 , ...  , + [SEP]
                # Input type:   1     , 3...     ,  2,      5...,     4, 6...,     4, 7...,      4, ... 4,    2
                type_indicator = 5
                for j in range(len(all_label_token_ids)):
                    tmp_label_len = len(all_label_token_ids[j])
                    support_tokens[i][tmp_text_len: tmp_text_len+tmp_label_len] = torch.tensor(all_label_token_ids[j], dtype=torch.long)
                    support_token_type_ids[i][tmp_text_len: tmp_text_len+tmp_label_len] = torch.tensor(np.repeat(type_indicator,tmp_label_len))
                    support_current_label_ids[i][tmp_text_len: tmp_text_len+tmp_label_len] = 1 if all_label_token_ids[j]==sample.label_token_id else 0
                    if self.clsTagSep == ",":
                        support_tokens[i][tmp_text_len+tmp_label_len] = 1010
                        support_token_type_ids[i][tmp_text_len+tmp_label_len] = self.TSpt_id
                        tmp_text_len = tmp_text_len+tmp_label_len+1
                    elif self.clsTagSep == ".":
                        support_tokens[i][tmp_text_len+tmp_label_len] = 1012
                        support_token_type_ids[i][tmp_text_len+tmp_label_len] = self.TSpt_id
                        tmp_text_len = tmp_text_len+tmp_label_len+1
                    elif self.clsTagSep == "sep":
                        support_tokens[i][tmp_text_len+tmp_label_len] = 102
                        support_token_type_ids[i][tmp_text_len+tmp_label_len] = self.TSpt_id
                        tmp_text_len = tmp_text_len+tmp_label_len+1
                    else:
                        tmp_text_len = tmp_text_len+tmp_label_len
                    type_indicator += 1
                support_tokens[i][tmp_text_len] = 102
                support_token_type_ids[i][tmp_text_len] = self.SEP_id
                sup_token_len.append(tmp_text_len+1)

        sup_token_len = torch.tensor(sup_token_len, dtype=torch.long)   

        
        for i in range(len(query_example)):
            # 1. calculate the max length of query sample
            tmp_text_len = len(query_example[i].text_token_id)
            if self.addCtagQue != "one":
                que_max_len = que_max_len if que_max_len > tmp_text_len else tmp_text_len
            else:
                # tmp length: each query sample length + related label length + 1 for "SEP" token
                tmp_total_len = tmp_text_len+len(query_example[i].label_token_id)+1
                que_max_len = que_max_len if que_max_len > tmp_total_len else tmp_total_len
            # 2. get the each text length of query sample
            que_raw_token_len_wo.append(tmp_text_len - 2)
            que_raw_token_len.append(tmp_text_len)
            que_current_label_len[i] = len(query_example[i].label_token_id)
            # 3. construct the label id for query
            que_label_ids[i] = query_example[i].label_id
        
        if self.addCtagQue == "all":
            que_max_len = que_max_len + total_label_token_len + 1

        # construct tensors for support data
        query_tokens = torch.zeros(self.args.way*self.args.query, que_max_len, dtype=torch.long)
        query_token_type_ids = torch.zeros(self.args.way*self.args.query, que_max_len, dtype=torch.long)
        query_current_label_ids = torch.zeros(self.args.way*self.args.query, que_max_len, dtype=torch.long)
        query_current_sent_ids = torch.zeros(self.args.way*self.args.query, que_max_len, dtype=torch.long) # 1 for text position, 0 for rest
        for i, sample in enumerate(query_example):
            tmp_text_len = que_raw_token_len[i]
            # Basic Input format: [CLS] + Sentence + [SEP]
            # Input type:           1,       3...,     2
            query_tokens[i][0:tmp_text_len] = torch.tensor(sample.text_token_id, dtype=torch.long)
            query_token_type_ids[i][0] = self.CLS_id # CLS token type id: 1
            query_token_type_ids[i][1:tmp_text_len-1] = self.Sent_id  # text token type id: 3
            query_token_type_ids[i][tmp_text_len-1] = self.SEP_id  # SEP token type id: 2
            query_current_sent_ids[i][1:tmp_text_len-1] = 1
            if self.addCtagQue == "none":
                que_token_len.append(tmp_text_len)
            elif self.addCtagQue == "one":
                # Normally, the query cannot append class relevant tag !!!
                # Input format: [CLS] + Sentence + [SEP] + Class Tag + [SEP]
                # Input type:   1,      3...,       2,      5...,        2
                type_indicator = 5
                tmp_label_len = len(sample.label_token_id)
                query_tokens[i][tmp_text_len: tmp_text_len+tmp_label_len] = torch.tensor(sample.label_token_id, dtype=torch.long)
                query_tokens[i][tmp_text_len+tmp_label_len] = 102 # SEP for the last token
                query_token_type_ids[i][tmp_text_len: tmp_text_len+tmp_label_len] = type_indicator
                query_token_type_ids[i][tmp_text_len+tmp_label_len] = self.SEP_id
                query_current_label_ids[i][tmp_text_len: tmp_text_len+tmp_label_len] = 1
                que_token_len.append(tmp_text_len+tmp_label_len+1)
            else:
                # Input format: [CLS] + Sentence + [SEP] + Class Tag1 , Class Tag2 , Class Tag 3 , ...  , + [SEP]
                # Input type:   1     , 3...     ,  2,      5...,     4, 6...,     4, 7...,      4, ... 4,    2
                type_indicator = 5
                for j in range(len(all_label_token_ids)):
                    tmp_label_len = len(all_label_token_ids[j])
                    query_tokens[i][tmp_text_len: tmp_text_len+tmp_label_len] = torch.tensor(all_label_token_ids[j], dtype=torch.long)
                    query_token_type_ids[i][tmp_text_len: tmp_text_len+tmp_label_len] = torch.tensor(np.repeat(type_indicator,tmp_label_len))
                    if self.clsTagSep == ",":
                        query_tokens[i][tmp_text_len+tmp_label_len] = 1010
                        query_token_type_ids[i][tmp_text_len+tmp_label_len] = self.TSpt_id
                        tmp_text_len = tmp_text_len+tmp_label_len+1
                    elif self.clsTagSep == ".":
                        query_tokens[i][tmp_text_len+tmp_label_len] = 1012
                        query_token_type_ids[i][tmp_text_len+tmp_label_len] = self.TSpt_id
                        tmp_text_len = tmp_text_len+tmp_label_len+1
                    elif self.clsTagSep == "sep":
                        support_tokens[i][tmp_text_len+tmp_label_len] = 102
                        support_token_type_ids[i][tmp_text_len+tmp_label_len] = self.TSpt_id
                        tmp_text_len = tmp_text_len+tmp_label_len+1
                    else:
                        tmp_text_len = tmp_text_len+tmp_label_len
                    type_indicator += 1
                
                query_tokens[i][tmp_text_len] = 102
                query_token_type_ids[i][tmp_text_len] = 2
                que_token_len.append(tmp_text_len+1)

        que_token_len = torch.tensor(que_token_len, dtype=torch.long)

        sup_label_new_ids, que_label_new_ids = self._reidx_y(sup_label_ids, que_label_ids)
        sup_label_ids_onehot = self._label2onehot(sup_label_new_ids)
        que_label_ids_onehot = self._label2onehot(que_label_new_ids)

        # _, sup_label_unique_ids = torch.unique_consecutive(sup_label_ids, return_inverse=True)
        # _, que_label_unique_ids = torch.unique_consecutive(que_label_ids, return_inverse=True)
        all_label_token_ids_grouped = []
        if self.args.use_alllabel_feature:
            # construct tensors for all labels
            if self.args.alllabel_append == "special":
                all_label_token_ids_grouped.extend([101]) # append cls token
            for i in range(self.args.way):
                all_label_token_ids_grouped.extend(all_label_token_ids[i])
                if self.clsTagSep == ",":
                    all_label_token_ids_grouped.extend([1010])
                elif self.clsTagSep == ".":
                    all_label_token_ids_grouped.extend([1012])
                elif self.clsTagSep == "sep":
                    all_label_token_ids_grouped.extend([102])
            if self.args.alllabel_append == "special" and self.clsTagSep !="sep":
                all_label_token_ids_grouped.extend([102])
            all_label_token_ids_positions = torch.zeros(self.args.way, len(all_label_token_ids_grouped), dtype=torch.float)
            
            tmp_pos = 1 if self.args.alllabel_append == "special" else 0
            for i in range(self.args.way):
                tmp_len = len(all_label_token_ids[i])
                # print("i, start, end: ", i, tmp_pos, tmp_pos+tmp_len)
                all_label_token_ids_positions[i][tmp_pos: tmp_pos+tmp_len] = 1
                if self.clsTagSep == "none":
                    tmp_pos = tmp_pos+tmp_len
                else:
                    tmp_pos = tmp_pos+tmp_len+1
        else:
            all_label_token_ids_positions = torch.zeros(0)
        all_label_token_ids_grouped = torch.tensor(all_label_token_ids_grouped, dtype=torch.long)
        all_label_token_lens = torch.tensor(label_token_len, dtype=torch.float)

        support = {
            'token_ids': support_tokens.cuda(),
            'token_len': sup_token_len.cuda(),
            'token_type_ids': support_token_type_ids.cuda(),
            'current_sent_ids': support_current_sent_ids.cuda(),
            'current_sent_len': torch.tensor(sup_raw_token_len_wo, dtype=torch.float).cuda(),
            'current_label_ids': support_current_label_ids.cuda(),
            'current_label_len': sup_current_label_len.cuda(),
            'label_ids': sup_label_ids.cuda(),
            'label_new_ids': sup_label_new_ids.cuda(),
            'label_ids_onehot': sup_label_ids_onehot.cuda(),
            'all_label_token_lens': all_label_token_lens.cuda(),
            'all_label_token_ids_grouped': all_label_token_ids_grouped.cuda(),
            'all_label_token_ids_positions': all_label_token_ids_positions.cuda(),
            'is_support': True
        }

        query = {
            'token_ids': query_tokens.cuda(),
            'token_len': que_token_len.cuda(),
            'token_type_ids': query_token_type_ids.cuda(),
            'current_sent_ids': query_current_sent_ids.cuda(),
            'current_sent_len': torch.tensor(que_raw_token_len_wo,dtype=torch.float).cuda(),
            'current_label_ids': query_current_label_ids.cuda(),
            'current_label_len': que_current_label_len.cuda(),
            'label_ids': que_label_ids.cuda(),
            'label_new_ids': que_label_new_ids.cuda(),
            'label_ids_onehot': que_label_ids_onehot.cuda(),
            'all_label_token_lens': all_label_token_lens.cuda(),
            'all_label_token_ids_grouped': all_label_token_ids_grouped.cuda(),
            'all_label_token_ids_positions': all_label_token_ids_positions.cuda(),
            'is_support': False
        }
        return support, query

    
    def get_epoch(self):
        if self.state == "train":
            for _ in range(self.num_episodes):
                # step 1. determine if the task is cross-domain or in-domain
                randm_domain_ids = np.random.permutation(len(self.args.train_domains))[0]
                # import pdb; pdb.set_trace()
                task_data  = self.data[randm_domain_ids]
                # step 2. determine the classes for each task and get samples from task ids
                randm_class_ids = np.random.permutation(len(self.args.train_classes))[0:self.args.way]
                support_examples, query_examples = self._get_sample_from_task_ids(randm_class_ids, task_data)
                # step 3. construct support and query data
                support, query = self._convet_examples_to_tensor(support_examples, query_examples)
                # import pdb; pdb.set_trace()
                yield support, query
        elif self.state == "val":
            for _ in range(self.num_episodes):
                # step 1. determine if the task is cross-domain or in-domain
                randm_domain_ids = np.random.permutation(len(self.args.val_domains))[0]
                task_data = self.data[randm_domain_ids]
                # step 2. determine the classes for each task and get samples from task ids
                randm_class_ids = np.random.permutation(len(self.args.val_classes))[0:self.args.way]
                support_examples, query_examples = self._get_sample_from_task_ids(randm_class_ids, task_data)
                # step 3. construct support and query data
                support, query = self._convet_examples_to_tensor(support_examples, query_examples)
                # import pdb; pdb.set_trace()
                yield support, query
        else:
            for _ in range(self.num_episodes):
                # step 1. determine if the task is cross-domain or in-domain
                randm_domain_ids = np.random.permutation(len(self.args.test_domains))[0]
                task_data = self.data[randm_domain_ids]
                # step 2. determine the classes for each task and get samples from task ids
                randm_class_ids = np.random.permutation(len(self.args.test_classes))[0:self.args.way]
                support_examples, query_examples = self._get_sample_from_task_ids(randm_class_ids, task_data)
                # step 3. construct support and query data
                support, query = self._convet_examples_to_tensor(support_examples, query_examples)
                # import pdb; pdb.set_trace()
                yield support, query