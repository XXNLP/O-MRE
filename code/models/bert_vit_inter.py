import torch.cuda
import numpy as np
from torch import nn, Tensor, device
from torchcrf import CRF

from .bert_vit_inter_base_model import BertVitInterBaseModel
from transformers.modeling_outputs import TokenClassifierOutput
from .matrix_re_loss import MatrixRELoss, BottleneckLoss, MatrixRELossWithMiddleLayerAttentionBI


class REClassifier(nn.Module):
    def __init__(self, re_label_mapping=None, config=None, tokenizer=None):
        super().__init__()
        self.text_config = config
        num_relation_labels = len(re_label_mapping)
        self.classifier = nn.Linear(2 * self.text_config.hidden_size, num_relation_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")  # <s> id: 30522
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")  # <o> id: 30526

    def forward(self, input_ids=None, output_state=None):
        (output_state, vision_hidden_states, text_hidden_states) = output_state
        last_hidden_state, pooler_output = output_state.last_hidden_state, output_state.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape

        entity_hidden_state = torch.Tensor(bsz, 2 * hidden_size)  # batch, 2*hidden
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        if torch.cuda.is_available():
            entity_hidden_state = entity_hidden_state.to('cuda')
        logits = self.classifier(entity_hidden_state)
        return logits

class REMatrixClassifier(nn.Module):

    def __init__(self, re_label_mapping=None, config=None, tokenizer=None):
        super().__init__()
        self.text_config = config
        hidden_size = config.hidden_size
        num_class = len(re_label_mapping)
        self.attn_score = MultiHeadAttention(input_size=hidden_size,
                                            output_size=num_class * hidden_size,
                                             num_heads=num_class)
    
    def forward(self, input_ids=None, output_state=None):

        (output_state, vision_hidden_states, text_hidden_states) = output_state
        last_hidden_state, pooler_output = output_state.last_hidden_state, output_state.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape

        score = self.attn_score(last_hidden_state, last_hidden_state)  # BS * NTL * SL * SL

        score = score.sigmoid()

        #return score, subject_output_logits, tokens_atts
        return score  # BS * NTL * SL * SL
        


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, output_size, num_heads, output_attentions=False):
        super(MultiHeadAttention, self).__init__()
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        self.d_model_size = input_size

        self.depth = int(output_size / self.num_heads)

        self.Wq = torch.nn.Linear(input_size, output_size)
        self.Wk = torch.nn.Linear(input_size, output_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)  # BS * SL * NH * H
        return x.permute([0, 2, 1, 3])  # BS * NH * SL * H

    def forward(self, k, q):  # BS * SL * HS
        batch_size = q.shape[0]

        q = self.Wq(q)  # BS * SL * OUT
        k = self.Wk(k)  # BS * SL * OUT

        # q = F.dropout(q, 0.8, training=self.training)
        # k = F.dropout(k, 0.8, training=self.training)

        q = self.split_into_heads(q, batch_size)  # BS * NH * SL * H
        k = self.split_into_heads(k, batch_size)  # BS * NH * SL * H

        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2))
        attn_score = attn_score / np.sqrt(k.shape[-1])

        # scaled_attention = output[0].permute([0, 2, 1, 3])
        # attn = output[1]
        # original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        # output = self.dense(original_size_attention)

        return attn_score

class BertVitInterMatrixReModelBI(nn.Module):
    def __init__(self,
                 re_label_mapping=None,
                 tokenizer=None,
                 args=None,
                 vision_config=None,
                 text_config=None,
                 clip_model_dict=None,
                 bert_model_dict=None, ):
        super().__init__()
        self.args = args
        #print(vision_config)
        #print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config

        vision_config.device = args.device
        self.model = BertVitInterBaseModel(vision_config, text_config, args)

        # test load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            if 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        self.model.load_state_dict(model_dict)
        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args
        # RE classifier
        self.re_classifier = REMatrixClassifier(re_label_mapping=re_label_mapping, config=text_config,
                                          tokenizer=tokenizer)
        #self.matrix_loss = MatrixRELoss()
        
        #self.bottleneck_loss = BottleneckLoss(BIAttentitonDim=768)
        #self.bottleneck_loss_norm = BottleneckLossNorm()
        
        #self.bottleneck_loss_initial = BottleneckLoss(BIAttentitonDim=768)
        #self.bottleneck_loss_middle = BottleneckLoss(BIAttentitonDim=768)
        self.matrix_loss = MatrixRELossWithMiddleLayerAttentionBI(BIAttentitonDim=768)


        
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
            matrix_labels=None,
            task='re',
            epoch=0,
    ):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,
                            output_hidden_states=True, )
        if task == 're':
            (output_state, vision_hidden_states, text_hidden_states) = output
            bert_vit_matrix_logits = self.re_classifier(output_state=output, input_ids=input_ids)


            vision_all_hidden_states, text_all_hidden_states = output_state.hidden_states[0],output_state.hidden_states[1]
            if labels is not None:
                #label_loss_bert_vit = self.matrix_loss(bert_vit_matrix_logits, matrix_labels)
                #bottleneck_loss = self.bottleneck_loss(vision_all_hidden_states[11],text_all_hidden_states[11])                
                
                #totall_loss = label_loss_bert_vit+bottleneck_loss
                #return totall_loss, bert_vit_matrix_logits
                label_matrix_loss_fn = self.matrix_loss #nn.CrossEntropyLoss()
                label_loss_bert_vit = label_matrix_loss_fn(bert_vit_matrix_logits, matrix_labels,vision_hidden_states,text_hidden_states)
                #latbel_loss_bert_vit = label_loss_bert_vit+bottleneck_loss
                return label_loss_bert_vit, bert_vit_matrix_logits



# Bert VIT Matrix 84.0 bi at last hidden_states layer
class BertVitInterMatrixReModelBI_best(nn.Module):
    def __init__(self,
                 re_label_mapping=None,
                 tokenizer=None,
                 args=None,
                 vision_config=None,
                 text_config=None,
                 clip_model_dict=None,
                 bert_model_dict=None, ):
        super().__init__()
        self.args = args
        #print(vision_config)
        #print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config

        vision_config.device = args.device
        self.model = BertVitInterBaseModel(vision_config, text_config, args)

        # test load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            if 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        self.model.load_state_dict(model_dict)
        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args
        # RE classifier
        self.re_classifier = REMatrixClassifier(re_label_mapping=re_label_mapping, config=text_config,
                                          tokenizer=tokenizer)
        #self.matrix_loss = MatrixRELoss()
        
        #self.bottleneck_loss = BottleneckLoss(BIAttentitonDim=768)
        #self.bottleneck_loss_norm = BottleneckLossNorm()
        
        #self.bottleneck_loss_initial = BottleneckLoss(BIAttentitonDim=768)
        #self.bottleneck_loss_middle = BottleneckLoss(BIAttentitonDim=768)
        self.matrix_loss = MatrixRELossWithMiddleLayerAttentionBI(BIAttentitonDim=768)


        
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
            matrix_labels=None,
            task='re',
            epoch=0,
    ):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,
                            output_hidden_states=True, )
        if task == 're':
            (output_state, vision_hidden_states, text_hidden_states) = output
            bert_vit_matrix_logits = self.re_classifier(output_state=output, input_ids=input_ids)


            vision_all_hidden_states, text_all_hidden_states = output_state.hidden_states[0],output_state.hidden_states[1]
            if labels is not None:
                #label_loss_bert_vit = self.matrix_loss(bert_vit_matrix_logits, matrix_labels)
                #bottleneck_loss = self.bottleneck_loss(vision_all_hidden_states[11],text_all_hidden_states[11])                
                
                #totall_loss = label_loss_bert_vit+bottleneck_loss
                #return totall_loss, bert_vit_matrix_logits
                label_matrix_loss_fn = self.matrix_loss #nn.CrossEntropyLoss()
                label_loss_bert_vit = label_matrix_loss_fn(bert_vit_matrix_logits, matrix_labels,vision_hidden_states,text_hidden_states)
                #latbel_loss_bert_vit = label_loss_bert_vit+bottleneck_loss
                return label_loss_bert_vit, bert_vit_matrix_logits


# Bert VIT Matrix
class BertVitInterMatrixReModel(nn.Module):
    def __init__(self,
                 re_label_mapping=None,
                 tokenizer=None,
                 args=None,
                 vision_config=None,
                 text_config=None,
                 clip_model_dict=None,
                 bert_model_dict=None, ):
        super().__init__()
        self.args = args
        #print(vision_config)
        #print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config

        vision_config.device = args.device
        self.model = BertVitInterBaseModel(vision_config, text_config, args)

        # test load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            if 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        self.model.load_state_dict(model_dict)
        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args
        # RE classifier
        self.re_classifier = REMatrixClassifier(re_label_mapping=re_label_mapping, config=text_config,
                                          tokenizer=tokenizer)
        self.matrix_loss = MatrixRELoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
            matrix_labels=None,
            task='re',
            epoch=0,
    ):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,
                            output_hidden_states=True, )
        if task == 're':
            bert_vit_matrix_logits = self.re_classifier(output_state=output, input_ids=input_ids)

            if labels is not None:
                label_matrix_loss_fn = self.matrix_loss #nn.CrossEntropyLoss()
                label_loss_bert_vit = label_matrix_loss_fn(bert_vit_matrix_logits, matrix_labels)

                return label_loss_bert_vit, bert_vit_matrix_logits


# Bert VIT
class BertVitInterReModel(nn.Module):
    def __init__(self,
                 re_label_mapping=None,
                 tokenizer=None,
                 args=None,
                 vision_config=None,
                 text_config=None,
                 clip_model_dict=None,
                 bert_model_dict=None, ):
        super().__init__()
        self.args = args
        #print(vision_config)
        #print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config

        vision_config.device = args.device
        self.model = BertVitInterBaseModel(vision_config, text_config, args)

        # test load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            if 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        self.model.load_state_dict(model_dict)
        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args
        # RE classifier
        self.re_classifier = REClassifier(re_label_mapping=re_label_mapping, config=text_config,
                                          tokenizer=tokenizer)
        
        self.bottleneck_loss = BottleneckLoss(BIAttentitonDim=768)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
            matrix_labels=None,
            task='re',
            epoch=0,
    ):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,
                            output_hidden_states=True, )
        if task == 're':
            (output_state, vision_hidden_states, text_hidden_states) = output
            bert_vit_logits = self.re_classifier(output_state=output, input_ids=input_ids)

            if labels is not None:
                label_ce_loss_fn = nn.CrossEntropyLoss()
                label_loss_bert_vit = label_ce_loss_fn(bert_vit_logits, labels.view(-1))

                bottleneck_loss = self.bottleneck_loss(vision_hidden_states,text_hidden_states)
                total_loss= label_loss_bert_vit+bottleneck_loss

                return total_loss, bert_vit_logits


class BertViTInterNerModel(nn.Module):#NER模型
    def __init__(self,
                 label_list,
                 args,
                 vision_config,
                 text_config,
                 logger=None, ):
        super(BertViTInterNerModel, self).__init__()
        self.args = args
        #print(vision_config)
        #print(text_config)
        self.vision_config = vision_config
        self.text_config = text_config
        self.model = BertVitInterBaseModel(vision_config, text_config, self.args)

        self.num_labels = len(label_list) + 1  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.text_config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.batch_id = 0

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        bsz = input_ids.size(0)

        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,
                            output_hidden_states=True, )
        (output_state, _, _) = output
        sequence_output = output_state.last_hidden_state  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)  # bsz, len, labels

        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
        return TokenClassifierOutput(
            loss=loss,
            logits=logits
        )
