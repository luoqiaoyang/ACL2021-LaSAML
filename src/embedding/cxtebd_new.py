import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class CXTEBDNEW(nn.Module):
    '''
        An embedding layer directly returns precomputed BERT
        embeddings.
    '''
    def __init__(self, args):
        '''
            pretrained_model_name_or_path, cache_dir: check huggingface's codebase for details
            finetune_ebd: finetuning bert representation or not during
            meta-training
            return_seq: return a sequence of bert representations, or [cls]
        '''
        super(CXTEBDNEW, self).__init__()

        # Step 1. Get ebd of all the labels

        self.args = args

        print("{}, Loading pretrained bert".format(
            datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S')), flush=True)

        self.model = BertModel.from_pretrained(self.args.pretrained_bert,
                                               cache_dir=self.args.bert_cache_dir)

        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size
        
        self.lam_sup = nn.Parameter(torch.tensor(self.args.lmbd_init, dtype=torch.float))
        self.fea_avg = nn.AvgPool1d(self.args.shot, stride=self.args.shot)

        if self.args.sup_feature == "mlp_all":
            self.mlp_supAll1 = nn.Linear(self.ebd_dim*3, self.ebd_dim)
            self.mlp_supAll2 = nn.Linear(self.ebd_dim, self.ebd_dim)
            self.layernorm_supAll1 = nn.LayerNorm(self.ebd_dim*3)
            self.layernorm_supAll2 = nn.LayerNorm(self.ebd_dim)
            self.dropout1 = nn.Dropout(0.1)
        elif self.args.sup_feature == "comb_att":
            self.mlp_supAll = nn.Linear(self.ebd_dim, 1)
            self.softmax = nn.Softmax(dim=1)

        if self.args.sup_w_diff:
            self.scaler = self.args.way*self.args.shot/((self.args.way-1)*self.args.shot)
            self.fea_scaler = nn.Parameter(torch.tensor(self.scaler, dtype=torch.float))

    def get_bert_ebd(self, token_ids):
        # get the embedding via BERT token id
        position_ids = None
        inputs_embeds = None
        input_shape = token_ids.size()
        device = token_ids.device
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if self.args.fixbert == "ebd" or self.args.fixbert == "all":
            with torch.no_grad():
                embedding_output = self.model.embeddings(
                    input_ids=token_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
                )
        else:
            embedding_output = self.model.embeddings(
                input_ids=token_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
        return embedding_output

    def get_bert_encoder(self, bert_embedding, extended_attention_mask):
        
        head_mask = None
        head_mask = self.model.get_head_mask(head_mask, self.model.config.num_hidden_layers)
        encoder_hidden_states = None
        encoder_extended_attention_mask = None
        output_attentions = self.model.config.output_attentions
        output_hidden_states = (self.model.config.output_hidden_states)
        encoder_outputs = self.model.encoder(
            bert_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        return encoder_outputs
    
    def get_pretransformer(self, input_ids, label_new_ids, label_ids_onehot, is_support):
        # get the label embedding via BERT token id
        input_ebd = self.get_bert_ebd(input_ids.unsqueeze(0)).squeeze() if self.args.pretransformer_mode == "label" else self.get_bert_ebd(input_ids)
        input_ebd = self.pretransformer(input_ebd,label_new_ids, label_ids_onehot, is_support)
        return input_ebd


    def get_posttransformer(self, ebd, input_type_id):
        # get the label embedding via BERT token id
        cls_ebd = self.posttransformer(ebd, input_type_id)
        return cls_ebd

    def get_bert(self, data):
        '''
            Return the last layer of bert's representation
            @param: bert_id: batch_size * max_text_len+2
            @param: text_len: text_len

            @return: last_layer: batch_size * max_text_len
        '''

        text_token_ids = data['token_ids']
        text_len = data['token_len']
        text_type_id = data['token_type_ids']
        current_sent_ids = data['current_sent_ids']
        current_sent_len = data['current_sent_len']
        current_label_ids = data['current_label_ids']
        current_label_len = data['current_label_len']
        label_new_ids = data['label_new_ids']
        label_ids_onehot = data['label_ids_onehot']
        all_label_token_lens = data['all_label_token_lens']
        all_label_token_ids_grouped = data['all_label_token_ids_grouped']
        all_label_token_ids_positions = data['all_label_token_ids_positions']
        is_support = data['is_support']

        input_shape = text_token_ids.size()
        device = text_token_ids.device

        len_range = torch.arange(input_shape[-1], device=device,
                dtype=text_len.dtype).expand(*input_shape)
        attention_mask = (len_range < text_len.unsqueeze(-1)).long()

        
        extended_attention_mask: torch.Tensor = self.model.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.args.use_pretransformer:
            if self.args.pretransformer_mode == "label":
                text_embedding_output = self.get_bert_ebd(text_token_ids)
                if is_support:
                    label_ebd = self.get_pretransformer(all_label_token_ids_grouped, label_new_ids, label_ids_onehot, is_support)
                    text_embedding_output[:, text_len-2] = label_ebd
                    text_embedding_output[:, text_len-1] = text_embedding_output[:, text_len-3] # embedding of first SEP = second SEP
            else:
                text_embedding_output = self.get_pretransformer(text_token_ids, text_type_id, label_ids_onehot, is_support)
        else:
            # print("no pretransformer")
            text_embedding_output = self.get_bert_ebd(text_token_ids)
            if self.args.use_current_label_embedding:
                current_label_ebd = self.current_label_embedding(current_label_ids)
                text_embedding_output = text_embedding_output + current_label_ebd

        if self.args.fixbert == "encoder" or self.args.fixbert == "all":
                with torch.no_grad():
                    encoder_outputs = self.get_bert_encoder(text_embedding_output, extended_attention_mask)
        else:
            # print("no fixbert")
            encoder_outputs = self.get_bert_encoder(text_embedding_output, extended_attention_mask)
        
        last_layer = encoder_outputs[0]

        if self.args.use_alllabel_feature:
            all_label_ebd = self.get_bert_ebd(all_label_token_ids_grouped.unsqueeze(0))
            if self.args.use_transformer_alllabs:
                # print("lab use post transformer")
                all_label_ebd = all_label_ebd.squeeze()
                all_label_outputs = self.get_posttransformer(all_label_ebd, label_new_ids)
                all_label_outputs = all_label_outputs.unsqueeze(0)
            else:
                all_label_attention_mask = torch.ones(all_label_token_ids_grouped.shape, dtype=torch.long, device=device).unsqueeze(0)
                all_label_extended_attention_mask: torch.Tensor = self.model.get_extended_attention_mask(all_label_attention_mask, all_label_ebd.shape[:2], device)
                all_label_outputs = self.get_bert_encoder(all_label_ebd, all_label_extended_attention_mask)[0]
            each_label_outputs = torch.bmm(all_label_token_ids_positions.unsqueeze(0), all_label_outputs).squeeze()
            each_label_outputs = each_label_outputs/all_label_token_lens.unsqueeze(1)
            if self.args.use_posttransformer_lab:
                # print("lab use post transformer")
                each_label_outputs = self.get_posttransformer(each_label_outputs, label_new_ids)
        
        if is_support:
            if self.args.sup_feature == "cls":
                last_layer = last_layer[:,0,:]
            elif self.args.sup_feature == "sent":
                last_layer = torch.sum(last_layer*current_sent_ids.unsqueeze(-1), dim=1)/current_sent_len.unsqueeze(-1)
            elif self.args.sup_feature == "tag":
                last_layer = torch.sum(last_layer*current_label_ids.unsqueeze(-1), dim=1)/current_label_len.unsqueeze(-1)
            elif self.args.sup_feature == "comb_cs":
                last_cls= last_layer[:,0,:]
                last_sent = torch.sum(last_layer*current_sent_ids.unsqueeze(-1), dim=1)/current_sent_len.unsqueeze(-1)
                last_layer = (1-self.lam_sup)*last_cls + self.lam_sup*last_sent
            elif self.args.sup_feature == "comb_ct":
                last_cls= last_layer[:,0,:]
                last_tag = torch.sum(last_layer*current_label_ids.unsqueeze(-1), dim=1)/current_label_len.unsqueeze(-1)
                last_layer = (1-self.lam_sup)*last_cls + self.lam_sup*last_tag
            elif self.args.sup_feature == "comb_st":
                last_sent = torch.sum(last_layer*current_sent_ids.unsqueeze(-1), dim=1)/current_sent_len.unsqueeze(-1)
                last_tag = torch.sum(last_layer*current_label_ids.unsqueeze(-1), dim=1)/current_label_len.unsqueeze(-1)
                last_layer = (1-self.lam_sup)*last_sent + self.lam_sup*last_tag
            elif self.args.sup_feature == "comb_all":
                last_cls= last_layer[:,0,:]
                last_sent = torch.sum(last_layer*current_sent_ids.unsqueeze(-1), dim=1)/current_sent_len.unsqueeze(-1)
                last_tag = torch.sum(last_layer*current_label_ids.unsqueeze(-1), dim=1)/current_label_len.unsqueeze(-1)
                last_layer = (last_cls + last_sent + last_tag)/3
            elif self.args.sup_feature == "mlp_all":
                last_cls= last_layer[:,0,:]
                last_sent = torch.sum(last_layer*current_sent_ids.unsqueeze(-1), dim=1)/current_sent_len.unsqueeze(-1)
                last_tag = torch.sum(last_layer*current_label_ids.unsqueeze(-1), dim=1)/current_label_len.unsqueeze(-1)
                last_layer = torch.cat((last_cls, last_sent, last_tag), dim=1)
                last_layer = self.layernorm_supAll1(last_layer)
                last_layer = self.dropout1(F.relu(self.mlp_supAll1(last_layer)))
                last_layer = self.mlp_supAll2(self.layernorm_supAll2(last_layer))
            elif self.args.sup_feature == "comb_att":
                last_cls= last_layer[:,0,:]
                last_sent = torch.sum(last_layer*current_sent_ids.unsqueeze(-1), dim=1)/current_sent_len.unsqueeze(-1)
                last_tag = torch.sum(last_layer*current_label_ids.unsqueeze(-1), dim=1)/current_label_len.unsqueeze(-1)
                last_layer = torch.stack((last_cls, last_sent, last_tag), dim=1)
                att_score = self.mlp_supAll(last_layer)
                att_score = self.softmax(att_score)
                last_layer = last_layer*att_score
                last_layer = torch.sum(last_layer, dim=1)

            if self.args.sup_w_diff:
                for i in range(self.args.way):
                    current_class_feature = torch.sum(last_layer[i*self.args.shot: (i+1)*self.args.shot], dim=0)
                    all_class_features = torch.sum(last_layer, dim=0)
                    difference_features = (all_class_features - current_class_feature)/((self.args.way-1)*self.args.shot)
                    for j in range(self.args.shot):
                        last_layer[i*self.args.shot+j] = (last_layer[i*self.args.shot+j] - difference_features)*self.fea_scaler

            if self.args.use_alllabel_feature:
                if self.args.shot > 1:
                    if self.args.alllabel_process == "lmbdaadd" or self.args.alllabel_process == "dynamicadd" or self.args.alllabel_process == "concat":
                        each_label_outputs_copy = torch.zeros(last_layer.shape, dtype=last_layer.dtype, device=device)
                        for i in range(self.args.shot):
                            for j in range(self.args.way):
                                each_label_outputs_copy[i*5+j] = each_label_outputs[i]
                        each_label_outputs = each_label_outputs_copy

                if self.args.alllabel_process == "lmbdaadd":
                    # print("support use lmbdaadd ")
                    last_layer = (1-self.lam_fea)*last_layer + self.lam_fea*each_label_outputs
                elif self.args.alllabel_process == "dynamicadd":
                    tmp_tensor = torch.ones(self.ebd_dim, dtype=torch.float, device=device)
                    last_layer = (tmp_tensor-self.lam_fea)*last_layer + self.lam_fea*each_label_outputs
                elif self.args.alllabel_process == "concat":
                    last_layer = torch.cat((last_layer, each_label_outputs), dim=1)
                    last_layer = last_layer.unsqueeze(0)
                    last_layer = self.div_transformer(last_layer)
                    last_layer = last_layer.squeeze()
                
            if self.args.use_mlp_sup:
                # print("support use mlp ")
                last_layer = self.dense_sup(last_layer)
            
            if self.args.use_posttransformer_sup:
                # print("support use post transformer ")
                if self.args.use_alllabel_feature and self.args.alllabel_process == "casecade":
                    tmp_layer = torch.zeros(self.args.way*(self.args.shot+1), self.ebd_dim, dtype=last_layer.dtype, device=device)
                    tmp_label_new_ids = torch.zeros(self.args.way*(self.args.shot+1), dtype=label_new_ids.dtype, device=device)
                    for i in range(self.args.way):
                        for j in range(self.args.shot+1):
                            tmp_layer[i*(self.args.shot+1)+j] = last_layer[i*(self.args.shot)+j] if j!= self.args.shot else each_label_outputs[i]
                            tmp_label_new_ids[i*(self.args.shot+1)+j] = label_new_ids[i*(self.args.shot)+j] if j!= self.args.shot else label_new_ids[i*(self.args.shot)+j-1]
                    last_layer = self.get_posttransformer(tmp_layer, tmp_label_new_ids)
                elif self.args.use_alllabel_feature and self.args.alllabel_process == "sptransformer":
                    tmp_all_class_feature = []
                    for i in range(self.args.way):
                        tmp_class_feature = torch.zeros(self.args.shot+1, self.ebd_dim, dtype=last_layer.dtype, device=device)
                        for j in range(self.args.shot+1):
                            tmp_class_feature[j] = last_layer[i*self.args.shot+j] if j!= self.args.shot else each_label_outputs[i]
                        
                        tmp_all_class_feature.extend(self.get_posttransformer(tmp_class_feature, label_new_ids))
                    last_layer = torch.stack(tmp_all_class_feature)
                else:
                    last_layer = last_layer.unsqueeze(0)
                    last_layer = self.div_transformer(last_layer)
                    last_layer = last_layer.squeeze()
        else:
            if self.args.que_feature == "cls":
                last_layer = last_layer[:,0,:]
            elif self.args.que_feature == "sent":
                last_layer = torch.sum(last_layer*current_sent_ids.unsqueeze(-1), dim=1)/current_sent_len.unsqueeze(-1)
            elif self.args.que_feature == "tag":
                all_features = []
                for i in range(last_layer.shape[0]):
                    sample_feature = []
                    for j in range(self.args.way):
                        tmp_position_id = [1 if text_type_id[i][k] == (j+5) else 0 for k in range(text_type_id.shape[1])]
                        tmp_position_id = torch.tensor(tmp_position_id, dtype=torch.float, device=device)
                        class_feature = torch.sum(last_layer[i]*tmp_position_id.unsqueeze(-1), dim=0)/all_label_token_lens[j]
                        sample_feature.append(class_feature)
                    sample_feature = torch.stack(sample_feature)
                    all_features.append(sample_feature)
                last_layer = torch.stack(all_features)
            else:
                last_cls = last_layer[:,0,:]
                last_sent = torch.sum(last_layer*current_sent_ids.unsqueeze(-1), dim=1)/current_sent_len.unsqueeze(-1)
                all_features = []
                for i in range(last_layer.shape[0]):
                    sample_feature = []
                    for j in range(self.args.way):
                        tmp_position_id = [1 if text_type_id[i][k] == (j+5) else 0 for k in range(text_type_id.shape[1])]
                        tmp_position_id = torch.tensor(tmp_position_id, dtype=torch.float, device=device)
                        class_feature = torch.sum(last_layer[i]*tmp_position_id.unsqueeze(-1), dim=0)/all_label_token_lens[j]
                        sample_feature.append(class_feature)
                    sample_feature = torch.stack(sample_feature)
                    all_features.append(sample_feature)
                last_tag = torch.stack(all_features)
                last_layer = []
                if self.args.zero_comb == "ct":
                    last_layer.append(last_cls)
                else:
                    last_layer.append(last_sent)
                last_layer.append(last_tag)
            if self.args.use_posttransformer_que:
                last_layer = self.get_posttransformer(last_layer, label_new_ids)

        return last_layer


    def forward(self, data, weights=None):
        '''
            @param data: key 'ebd' = batch_size * max_text_len * embedding_dim
            @return output: batch_size * max_text_len * embedding_dim
        '''
        if self.args.finetune_ebd:
            return self.get_bert(data)
        else:
            with torch.no_grad():
                return self.get_bert(data)
