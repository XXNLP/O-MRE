import random
import os
import torch
import numpy as np
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,BertTokenizerFast
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class MREProcessor(object):
    def __init__(self, data_path, re_path, bert_name, clip_processor=None, aux_processor=None, rcnn_processor=None):
        self.data_path = data_path
        self.re_path = re_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<s>', '</s>', '<o>', '</o>']})
        self.entity_marker = {
            'head_start':self.tokenizer.convert_tokens_to_ids("<s>"),  # <s> id: 30522
            'head_end':self.tokenizer.convert_tokens_to_ids("</s>"),  # </s> id: 30523
            'tail_start':self.tokenizer.convert_tokens_to_ids("<o>"),  # <o> id: 30524
            'tail_end':self.tokenizer.convert_tokens_to_ids("</o>")  # </o> id: 30525
        }
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        if 'mnre' in load_file:
            with open(load_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
                for i, line in enumerate(lines):
                    line = ast.literal_eval(line)  # str to dict
                    words.append(line['token'])
                    relations.append(line['relation'])
                    heads.append(line['h'])  # {name, pos}
                    tails.append(line['t'])
                    imgids.append(line['img_id'])
                    dataid.append(i)

            assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

            aux_imgs = None
            aux_path = self.data_path[mode + "_auximgs"]
            aux_imgs = torch.load(aux_path)           # {"ori_img_id":id_pred_yolo_crop_num0, .....}
            rcnn_imgs = torch.load(self.data_path[mode + '_img2crop'])  #detected objects {"ori_imgs_name":[crops1,crops2,crop3...]"
            return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}
        
        elif 'jmere' in load_file:
            with open(load_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
                lines = ast.literal_eval(lines[0]) 
                for i, line in enumerate(lines):
                    for label in line['label_list']:
                        words.append(line['token'])
                        imgids.append(line['img_id'])
                        relations.append(label[0]['relation'])
                        head = {'name': label[0]['beg_ent']['name'], 'pos': label[0]['beg_ent']['pos']}
                        tail = {'name': label[0]['sec_ent']['name'], 'pos': label[0]['sec_ent']['pos']}
                        heads.append(head)
                        tails.append(tail)  
                        dataid.append(i)

            assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

            aux_imgs = None
            aux_path = self.data_path[mode + "_auximgs"]
            aux_imgs = torch.load(aux_path)           # {"ori_img_id":id_pred_yolo_crop_num0, .....}
            rcnn_imgs = torch.load(self.data_path[mode + '_img2crop'])  #detected objects {"ori_imgs_name":[crops1,crops2,crop3...]"
            return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}


    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

    # relation and corresponding train samples
    def get_rel2id(self, train_path):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        re2id = {key: [] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id
    

class MREOverlapProcessor(object):
    def __init__(self, data_path, re_path, bert_name, clip_processor=None, aux_processor=None, rcnn_processor=None):
        self.data_path = data_path
        self.re_path = re_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': [f"<e{i}>" for i in range(0,8)]+[f"<e{i}/>" for i in range(0,8)]})

        self.entity_marker = { f"<e{i}>": self.tokenizer.convert_tokens_to_ids(f"<e{i}>") for i in range(0,8)}
        self.entity_marker.update({ f"<e{i}/>": self.tokenizer.convert_tokens_to_ids(f"<e{i}/>") for i in range(0,8)})

        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        #del some no rel instances if train else false
        del_no_rel_ins = True if mode =='train' else False

        if 'mnre' in load_file:
            with open(load_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
                for i, line in enumerate(lines):
                    line = ast.literal_eval(line)  # str to dict
                    words.append(line['token'])
                    relations.append(line['relation'])
                    heads.append(line['h'])  # {name, pos}
                    tails.append(line['t'])
                    imgids.append(line['img_id'])
                    dataid.append(i)

            assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

            aux_imgs = None
            aux_path = self.data_path[mode + "_auximgs"]
            aux_imgs = torch.load(aux_path)           # {"ori_img_id":id_pred_yolo_crop_num0, .....}
            rcnn_imgs = torch.load(self.data_path[mode + '_img2crop'])  #detected objects {"ori_imgs_name":[crops1,crops2,crop3...]"

            data = {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}
            

            data = self.merge_samples_and_build_entity_lists(data,self.get_relation_dict())
            data = self.filter_entity_pairs_with_inconsistence_rel(data,del_no_rel_ins=True,del_ratio=0.5)
            return data
        
        elif 'jmere' in load_file:
            with open(load_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
                lines = ast.literal_eval(lines[0]) 
                for i, line in enumerate(lines):
                    for label in line['label_list']:
                        words.append(line['token'])
                        imgids.append(line['img_id'])
                        relations.append(label[0]['relation'])
                        head = {'name': label[0]['beg_ent']['name'], 'pos': label[0]['beg_ent']['pos']}
                        tail = {'name': label[0]['sec_ent']['name'], 'pos': label[0]['sec_ent']['pos']}
                        heads.append(head)
                        tails.append(tail)  
                        dataid.append(i)

            assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

            aux_imgs = None
            aux_path = self.data_path[mode + "_auximgs"]
            aux_imgs = torch.load(aux_path)           # {"ori_img_id":id_pred_yolo_crop_num0, .....}
            rcnn_imgs = torch.load(self.data_path[mode + '_img2crop'])  #detected objects {"ori_imgs_name":[crops1,crops2,crop3...]"
            data = {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}

            data = self.merge_samples_and_build_entity_lists(data,self.get_relation_dict())
            data = self.filter_entity_pairs_with_inconsistence_rel(data,del_no_rel_ins=True,del_ratio=0.5)
            return data


    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

    # relation and corresponding train samples
    def get_rel2id(self, train_path):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        re2id = {key: [] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id
    
    def merge_samples_and_build_entity_lists(self, data, rel2id):
        """
        合并重复样本，构建 entity_list 和 entity_pair_list, 同时保留 imgids、aux_imgs 和 rcnn_imgs。data 来自 load_ori_data
        
        参数:
            - data: 数据字典，包含 words, relations, heads, tails, imgids, dataid, aux_imgs, rcnn_imgs。
            - rel2id: 关系到 ID 的映射字典。
        
        返回:
            - merged_data: 合并后的数据列表。
        """
        sentence_idx = {}  # 基于句子的哈希值去重
        merged_data = []

        for idx, (words, relation, head, tail, imgid, dataid) in enumerate(
            zip(data["words"], data["relations"], data["heads"], data["tails"], data["imgids"], data["dataid"])
        ):
            # 提取 imgid 的文件名（不含后缀）以匹配 rcnn_imgs
            imgid_key = os.path.splitext(imgid)[0]
            
            # 句子唯一标识
            sentence_key = " ".join(words)
            sentence_hash = hash(sentence_key)
            
            # 如果句子已存在，合并信息
            if sentence_hash in sentence_idx:
                idx = sentence_idx[sentence_hash]
                existing_sample = merged_data[idx]

                # 检查 head 和 tail 是否已存在于 entity_list
                head_entity = {"name": head["name"], "pos": head["pos"]}
                tail_entity = {"name": tail["name"], "pos": tail["pos"]}
                
                head_id = next((i for i, e in enumerate(existing_sample["entity_list"]) if e == head_entity), None)
                if head_id is None:
                    head_id = len(existing_sample["entity_list"])
                    existing_sample["entity_list"].append(head_entity)

                tail_id = next((i for i, e in enumerate(existing_sample["entity_list"]) if e == tail_entity), None)
                if tail_id is None:
                    tail_id = len(existing_sample["entity_list"])
                    existing_sample["entity_list"].append(tail_entity)
                
                # 检查是否已存在相同的 head_id 和 tail_id 但关系不同的情况
                entity_pair = [head_id, tail_id, rel2id[relation], dataid]
                if not any(ep[0] == head_id and ep[1] == tail_id and ep[2] != rel2id[relation] for ep in existing_sample["entity_pair_list"]):
                    existing_sample["entity_pair_list"].append(entity_pair)
                else:
                    # 如果关系不同，允许重叠的实体对添加到 entity_pair_list
                    for ep in existing_sample["entity_pair_list"]:
                        if ep[0] == head_id and ep[1] == tail_id and ep[2] != rel2id[relation]:
                            # 如果已经有相同实体对但关系不同，则添加新关系对
                            existing_sample["entity_pair_list"].append(entity_pair)
                            break
                
                # 合并 imgids
                if imgid not in existing_sample["imgids"]:
                    existing_sample["imgids"].append(imgid)
                
                # 合并 aux_imgs
                if dataid in data["aux_imgs"]:
                    if data["aux_imgs"][dataid] not in existing_sample["aux_imgs"]:
                        existing_sample["aux_imgs"].extend(data["aux_imgs"][dataid])

                # 合并 rcnn_imgs
                if imgid_key in data["rcnn_imgs"]:
                    for crop in data["rcnn_imgs"][imgid_key]:
                        if crop not in existing_sample["rcnn_imgs"]:
                            existing_sample["rcnn_imgs"].append(crop)
            else:
                # 创建新的样本
                head_entity = {"name": head["name"], "pos": head["pos"]}
                tail_entity = {"name": tail["name"], "pos": tail["pos"]}
                
                new_sample = {
                    "words": words,
                    "entity_list": [head_entity, tail_entity],
                    "entity_pair_list": [[0, 1, rel2id[relation], dataid]],
                    "imgids": [imgid],
                    "aux_imgs": data["aux_imgs"].get(dataid, []),
                    "rcnn_imgs": data["rcnn_imgs"].get(imgid_key, []),
                    "merged_id":idx
                }
                merged_data.append(new_sample)
                sentence_idx[sentence_hash] = len(merged_data) - 1

        return merged_data
    
    def filter_entity_pairs_with_inconsistence_rel(self, merged_data, del_no_rel_ins=True, del_ratio=0.3):
        """
        删除矛盾的关系，比如标记的有其他关系，又存在一个样本标记没有关系，同时删除只有一个关系对且标记为没有关系的样本，根据句子长短排序后，按照比例删除

        参数：
        - entity_pair_list: 原始的四元组列表，格式为 [头实体ID, 尾实体ID, 关系ID, 原始数据ID]
        - del_no_rel_ins = ['single','double','triple']: 删除样本中只有一个no关系的样本
        - del_ratio=0.3: 删除的比例

        返回：
        - filtered_list: 过滤后的四元组列表
        - deleted_duplicate_id: 被删除样本的原始数据ID列表
        """
        from collections import defaultdict

        deleted_duplicate_id = []
        deleted_single_no_rel_id=[]
        # 存储每对实体的所有关系
        for id, item in enumerate(merged_data): 
            entity_dict = defaultdict(list)
            entity_pair_list = item["entity_pair_list"]
            
            
            # 构建字典 { (头实体, 尾实体): [(关系ID, 原始数据ID)] }
            for head, tail, relation, orig_id in entity_pair_list:
                entity_dict[(head, tail)].append((relation, orig_id))
            
            # 过滤操作
            filtered_list = []

            
            for (head, tail), relations in entity_dict.items():
                # 筛选出所有非0关系的条目
                non_zero_relations = [item for item in relations if item[0] != 0]
                
                if non_zero_relations:
                    # 如果有非0关系，保留这些条目
                    for relation, orig_id in non_zero_relations:
                        filtered_list.append([head, tail, relation, orig_id])
                    # 删除所有关系为0的条目，并记录原始ID
                    zero_relations = [item for item in relations if item[0] == 0]
                    deleted_duplicate_id.extend([item[1] for item in zero_relations])
                else:
                    # 如果没有非0关系，保留关系为0的条目
                    for relation, orig_id in relations:
                        filtered_list.append([head, tail, relation, orig_id])
            merged_data[id]['entity_pair_list'] = filtered_list

            # 根据 entity_pair_list 长度,按照(merged_id号，句子长度)记录
            if len(item['entity_pair_list'])==1 and item['entity_pair_list'][0][2]==0:
                deleted_single_no_rel_id.append((id,len(item['words'])))
        

        if del_ratio > 0 and del_no_rel_ins:
            # 按句子长度升序排序（假设 item 中有 length 属性表示句子长度）
            deleted_single_no_rel_id=sorted(deleted_single_no_rel_id, key=lambda x: x[1])
            num_to_delete = int(len(deleted_single_no_rel_id) * del_ratio)
            deleted_single_no_rel_id = deleted_single_no_rel_id[:num_to_delete]
            deleted_single_no_rel_id = [item[0] for item in deleted_single_no_rel_id]
            # 删除样本并记录 ID

            merged_data = np.array(merged_data)
            merged_data = np.delete(merged_data, deleted_single_no_rel_id)
            merged_data = merged_data.tolist()
            
            #deleted_duplicate_id
            #deleted_single_no_rel_id
        return merged_data


 

class MREOverlapDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64,
                 mode="train", write_path=None, do_test=False) -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.rcnn_img_path = 'data/mnre/'
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.aux_processor = self.processor.aux_processor
        self.rcnn_processor = self.processor.rcnn_processor
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        self.write_path = write_path
        self.do_test = do_test

    def __len__(self):
        return len(self.data_dict)



    def __getitem__(self, idx):
        word_list, eps, epl, imgid, aux_imgs, rcnn_imgs = self.data_dict[idx]['words'], \
                                                     self.data_dict[idx]['entity_list'], \
                                                     self.data_dict[idx]['entity_pair_list'],  \
                                                     self.data_dict[idx]['imgids'],\
                                                     self.data_dict[idx]['aux_imgs'],\
                                                     self.data_dict[idx]['rcnn_imgs']
        

        item_id = self.data_dict[idx]['merged_id']
        # [CLS] ... <e0> entity1 <e0/> ... <e7> entity7 <e7/> .. [SEP]
       
        # insert ['<e0>', '<e0/>', '<e1>', '<e1/>',...'<e7>', '<e7/>']})

        extend_word_list = self.add_entity_labels(word_list, eps)

        extend_word_list = " ".join(extend_word_list)

        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)

        re_label = [ep[2] for ep in epl]  # label to id
        #将标签全部填充为关系数量大小,因为在overlapp中label的标签会转化为tensor，他们长度必须一致
        label_size = len(self.processor.get_relation_dict().keys())
        re_label.extend([0] * (label_size - len(re_label))) if len(re_label) < label_size else re_label

        matrix_lable = self.merge_one_hot_labels(epl, input_ids, self.tokenizer)
        
        # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            if self.aux_img_path is not None:
                # detected object img
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid[0].split(".")[0]  #由于合并后的imgid可能会存在两张，但全部都是一样的图，所以都选择第一张0
                if len(self.data_dict[idx]['aux_imgs'])>0:
                    aux_img_paths = self.data_dict[idx]['aux_imgs']
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]

                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)

                # padding
                for i in range(3 - len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    if len(self.data_dict[idx]['rcnn_imgs'])>0:
                        rcnn_img_paths = self.data_dict[idx]['rcnn_imgs']
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]

                    # select 3 img
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)

                    # padding
                    for i in range(3 - len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    if self.write_path is not None and self.mode == 'test' and self.do_test:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, extend_word_list, imgid, matrix_lable
                    else:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, matrix_lable

                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs, matrix_lable   #aux_imgs are the corped imgs
            #return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, matrix_lable
        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), matrix_lable

    def merge_one_hot_labels(self, entity_pair_list, input_ids, tokenizer=None):
        seq_len = len(input_ids)
        #initial zero matrix labels
        matrix_labels = torch.zeros((len(self.re_dict), seq_len, seq_len))
        for ep in entity_pair_list:
            #generate one matrix label for one entity pair
            em = MREOverlapDataset.get_entity_mask(input_ids, self.processor.entity_marker,ep)
            rel_id = ep[2]
            matrix_labels[rel_id] += em

        # keep all emements in the matraix are 0 or 1
        if matrix_labels.max().item() > 1:
            matrix_labels = (matrix_labels > 0).float()
        return matrix_labels

    
    @staticmethod
    def get_entity_mask(input_ids,entity_marker, entity_pair):
        seq_len = input_ids.shape[-1]
        #initial zero tensor
        entity_1 = torch.zeros((seq_len))
        entity_2 = torch.zeros((seq_len))
        #get corresponding idx of start and end positions
        head_pos,tail_pos = [entity_marker[f"<e{entity_pair[0]}>"],entity_marker[f"<e{entity_pair[0]}/>"]], \
                            [entity_marker[f"<e{entity_pair[1]}>"],entity_marker[f"<e{entity_pair[1]}/>"]]
        
        head_start_idx = input_ids.eq(head_pos[0]).nonzero().item()
        head_end_idx = input_ids.eq(head_pos[1]).nonzero().item()+1
        tail_start_idx = input_ids.eq(tail_pos[0]).nonzero().item()
        tail_end_idx = input_ids.eq(tail_pos[1]).nonzero().item()+1

        entity_1[head_start_idx:head_end_idx] = 1
        entity_2[tail_start_idx:tail_end_idx] = 1

        res = entity_1.unsqueeze(1).repeat_interleave(seq_len, dim=1)
        res += entity_2.unsqueeze(1).repeat_interleave(seq_len, dim=1).t()
        
        res = (res == 2).float()
        return res
    
    def add_entity_labels(self, word_list, entities):
        """
        给定一个词列表和多个实体位置信息，在对应的实体周围添加带编号的标签（<e1>, </e1>, <e2>, </e2>, ...）。
        
        参数：
        - word_list: 句子的词列表
        - entities: 实体列表，格式为 [{'name': 'entity_name', 'pos': [start, end]}, ...]。
        其中每个实体位置是一个字典，包含 `name`（实体名称）和 `pos`（实体位置的起始和结束索引，格式为 [start, end]）。

        返回：
        - extend_word_list: 经过修改后的词列表，其中包括带编号的实体标签。
        """
        # 初始化扩展后的词列表
        extend_word_list = []

        # 遍历 word_list 的索引和内容，同时检查是否需要插入实体标签
        for i, word in enumerate(word_list):
            # 检查是否是某个实体的开始或结束位置
            for idx, entity in enumerate(entities):
                start, end = entity['pos']
                if i == start:  # 起始位置，插入开标签
                    extend_word_list.append(f'<e{idx}>')
                if i == end:  # 结束位置，插入闭标签
                    extend_word_list.append(f'<e{idx}/>')
            
            # 添加当前词
            extend_word_list.append(word)
        
        # 检查是否有未关闭的标签（理论上不会出现，但加以保护）
        for idx, entity in enumerate(entities):
            if entity['pos'][1] == len(word_list):  # 如果实体的结束位置是句子的最后一个单词
                extend_word_list.append(f'<e{idx}/>')

        return extend_word_list


class MREDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64,
                 mode="train", write_path=None, do_test=False) -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.rcnn_img_path = 'data/mnre/'
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.aux_processor = self.processor.aux_processor
        self.rcnn_processor = self.processor.rcnn_processor
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        self.write_path = write_path
        self.do_test = do_test

    def __len__(self):
        return len(self.data_dict['words'])


    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
                                                     self.data_dict['heads'][idx], self.data_dict['tails'][idx], \
                                                     self.data_dict['imgids'][idx]
        
        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        in_head = False
        in_tail = False

        for i, word in enumerate(word_list):
            if i == head_pos[0]:
                in_head = True
                extend_word_list.append('<s>')
            elif i == head_pos[1] and in_head:
                in_head = False
                extend_word_list.append('</s>')
                
            if i == tail_pos[0]:
                in_tail = True
                extend_word_list.append('<o>')
            elif i == tail_pos[1] and in_tail:
                in_tail = False
                extend_word_list.append('</o>')
                
            extend_word_list.append(word)

        # 检查是否需要在末尾添加 '</s>' 或 '</o>'
        if in_head:  # 如果还在头实体中，说明头实体是最后一个单词
            extend_word_list.append('</s>')
        if in_tail:  # 如果还在尾实体中，说明尾实体是最后一个单词
            extend_word_list.append('</o>')

        extend_word_list = " ".join(extend_word_list)

        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)

        re_label = self.re_dict[relation]  # label to id
        
        matrix_lable = self.generate_matrix_label(head_pos,tail_pos, re_label, input_ids, self.tokenizer)

        # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            if self.aux_img_path is not None:
                # detected object img
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]

                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)

                # padding
                for i in range(3 - len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    if imgid in self.data_dict['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict['rcnn_imgs'][imgid]
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]

                    # select 3 img
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)

                    # padding
                    for i in range(3 - len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    if self.write_path is not None and self.mode == 'test' and self.do_test:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, extend_word_list, imgid, matrix_lable
                    else:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, matrix_lable

                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs, matrix_lable   #aux_imgs are the corped imgs
            #return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, matrix_lable
        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), matrix_lable

    def __getitem__triplet(self, idx):
        
        self.data_dict = MREDataset.merge_samples_and_build_entity_lists(self.data_dict, self.re_dict)
        
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx],self.data_dict['relations'][idx], \
                                                     self.data_dict['heads'][idx], self.data_dict['tails'][idx], \
                                                     self.data_dict['imgids'][idx]

        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        in_head = False
        in_tail = False

        for i, word in enumerate(word_list):
            if i == head_pos[0]:
                in_head = True
                extend_word_list.append('<s>')
            elif i == head_pos[1] and in_head:
                in_head = False
                extend_word_list.append('</s>')
                
            if i == tail_pos[0]:
                in_tail = True
                extend_word_list.append('<o>')
            elif i == tail_pos[1] and in_tail:
                in_tail = False
                extend_word_list.append('</o>')
                
            extend_word_list.append(word)

        # 检查是否需要在末尾添加 '</s>' 或 '</o>'
        if in_head:  # 如果还在头实体中，说明头实体是最后一个单词
            extend_word_list.append('</s>')
        if in_tail:  # 如果还在尾实体中，说明尾实体是最后一个单词
            extend_word_list.append('</o>')

        extend_word_list = " ".join(extend_word_list)

        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True,
                                                 padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)

        re_label = self.re_dict[relation]  # label to id
        
        matrix_lable = self.generate_matrix_label(head_pos,tail_pos, re_label, input_ids, self.tokenizer)

        # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            if self.aux_img_path is not None:
                # detected object img
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]

                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)

                # padding
                for i in range(3 - len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    if imgid in self.data_dict['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict['rcnn_imgs'][imgid]
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]

                    # select 3 img
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)

                    # padding
                    for i in range(3 - len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    if self.write_path is not None and self.mode == 'test' and self.do_test:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, extend_word_list, imgid, matrix_lable
                    else:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, matrix_lable

                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs, matrix_lable   #aux_imgs are the corped imgs
            #return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, matrix_lable
        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), matrix_lable



    def generate_matrix_label(self, head_pos,tail_pos, rel_id, input_ids, tokenizer=None):

        seq_len = len(input_ids)
        
        #initial zero matrix labels
        matrix_labels = torch.zeros((len(self.re_dict), seq_len, seq_len))
        
        #generate one matrix label for one entity pair
        em = MREDataset.get_entity_mask(input_ids, self.processor.entity_marker)

        matrix_labels[rel_id] += em
        
        # keep all emements in the matraix are 0 or 1
        if matrix_labels.max().item() > 1:
            matrix_labels = (matrix_labels > 0).float()
        return matrix_labels

    def get_entity_smarker(self):
        
        #head_start = self.tokenizer.convert_tokens_to_ids("<s>")  # <s> id: 30522
        #head_end = self.tokenizer.convert_tokens_to_ids("</s>")  # </s> id: 30523
        #tail_start = self.tokenizer.convert_tokens_to_ids("<o>")  # <o> id: 30524
        #tail_end= self.tokenizer.convert_tokens_to_ids("</o>")  # </o> id: 30525
        #head_pos_id, tail_pos_id = [head_start, head_end], [tail_start, tail_end]
        return self.processor.entity_marker
    @staticmethod
    def get_entity_mask(input_ids,entity_marker):
        seq_len = input_ids.shape[-1]
        #initial zero tensor
        entity_1 = torch.zeros((seq_len))
        entity_2 = torch.zeros((seq_len))
        #get corresponding idx of start and end positions
        head_pos,tail_pos = [entity_marker['head_start'],entity_marker['head_end']], [entity_marker['tail_start'],entity_marker['tail_end']]
        head_start_idx = input_ids.eq(head_pos[0]).nonzero().item()
        head_end_idx = input_ids.eq(head_pos[1]).nonzero().item()+1
        tail_start_idx = input_ids.eq(tail_pos[0]).nonzero().item()
        tail_end_idx = input_ids.eq(tail_pos[1]).nonzero().item()+1

        entity_1[head_start_idx:head_end_idx] = 1
        entity_2[tail_start_idx:tail_end_idx] = 1

        res = entity_1.unsqueeze(1).repeat_interleave(seq_len, dim=1)
        res += entity_2.unsqueeze(1).repeat_interleave(seq_len, dim=1).t()
        
        res = (res == 2).float()
        return res
    
class MREOverlapProcessor_test(object):
    def __init__(self, data_path, re_path, bert_name, clip_processor=None, aux_processor=None, rcnn_processor=None):
        self.data_path = data_path
        self.re_path = re_path
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_name, do_lower_case=True)
        #self.tokenizer.add_special_tokens({'additional_special_tokens': [f"<e{i}>" for i in range(0,8)]+[f"<e{i}/>" for i in range(0,8)]})
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<e>","<s>" ]})  
        self.entity_marker = { f"<e>": self.tokenizer.convert_tokens_to_ids('<e>')}
        self.entity_marker.update({ f"<s>": self.tokenizer.convert_tokens_to_ids(f"<s>")})
        #self.tokenizer.add_special_tokens({'additional_special_tokens': ['<e0>', '<e0/>', '<e1>', '<e1/>']})
        #self.entity_marker = {
        #    '<e0>':self.tokenizer.convert_tokens_to_ids("<e0>"),  # <s> id: 30522
        #    '<e0/>':self.tokenizer.convert_tokens_to_ids("<e0/>"),  # </s> id: 30523
        #    '<e1>':self.tokenizer.convert_tokens_to_ids("<e1/>"),  # <o> id: 30524
        #    '<e1/>':self.tokenizer.convert_tokens_to_ids("<e1/>")  # </o> id: 30525
        #}
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        #del some no rel instances if train else false
        del_no_rel_ins = True if mode =='train' else False

        if 'mnre' in load_file:
            with open(load_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
                for i, line in enumerate(lines):
                    line = ast.literal_eval(line)  # str to dict
                    words.append(line['token'])
                    relations.append(line['relation'])
                    heads.append(line['h'])  # {name, pos}
                    tails.append(line['t'])
                    imgids.append(line['img_id'])
                    dataid.append(i)

            assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

            aux_imgs = None
            aux_path = self.data_path[mode + "_auximgs"]
            aux_imgs = torch.load(aux_path)           # {"ori_img_id":id_pred_yolo_crop_num0, .....}
            rcnn_imgs = torch.load(self.data_path[mode + '_img2crop'])  #detected objects {"ori_imgs_name":[crops1,crops2,crop3...]"

            data = {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}
            

            #data = self.merge_samples_and_build_entity_lists(data,self.get_relation_dict())
            #data = self.filter_entity_pairs_with_inconsistence_rel(data,del_no_rel_ins=True,del_ratio=0.9)
            return data
        
        elif 'jmere' in load_file:
            with open(load_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
                lines = ast.literal_eval(lines[0]) 
                for i, line in enumerate(lines):
                    for label in line['label_list']:
                        words.append(line['token'])
                        imgids.append(line['img_id'])
                        relations.append(label[0]['relation'])
                        head = {'name': label[0]['beg_ent']['name'], 'pos': label[0]['beg_ent']['pos']}
                        tail = {'name': label[0]['sec_ent']['name'], 'pos': label[0]['sec_ent']['pos']}
                        heads.append(head)
                        tails.append(tail)  
                        dataid.append(i)

            assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

            aux_imgs = None
            aux_path = self.data_path[mode + "_auximgs"]
            aux_imgs = torch.load(aux_path)           # {"ori_img_id":id_pred_yolo_crop_num0, .....}
            rcnn_imgs = torch.load(self.data_path[mode + '_img2crop'])  #detected objects {"ori_imgs_name":[crops1,crops2,crop3...]"
            data = {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}

            #data = self.merge_samples_and_build_entity_lists(data,self.get_relation_dict())
            #data = self.filter_entity_pairs_with_inconsistence_rel(data,del_no_rel_ins=True,del_ratio=0.5)
            return data


    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

    # relation and corresponding train samples
    def get_rel2id(self, train_path):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        re2id = {key: [] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id
    
    def merge_samples_and_build_entity_lists(self, data, rel2id):
        """
        合并重复样本，构建 entity_list 和 entity_pair_list, 同时保留 imgids、aux_imgs 和 rcnn_imgs。data 来自 load_ori_data
        
        参数:
            - data: 数据字典，包含 words, relations, heads, tails, imgids, dataid, aux_imgs, rcnn_imgs。
            - rel2id: 关系到 ID 的映射字典。
        
        返回:
            - merged_data: 合并后的数据列表。
        """
        sentence_idx = {}  # 基于句子的哈希值去重
        merged_data = []

        for idx, (words, relation, head, tail, imgid, dataid) in enumerate(
            zip(data["words"], data["relations"], data["heads"], data["tails"], data["imgids"], data["dataid"])
        ):
            # 提取 imgid 的文件名（不含后缀）以匹配 rcnn_imgs
            imgid_key = os.path.splitext(imgid)[0]
            
            # 句子唯一标识
            sentence_key = " ".join(words)
            sentence_hash = hash(sentence_key)
            
            # 如果句子已存在，合并信息
            if sentence_hash in sentence_idx:
                idx = sentence_idx[sentence_hash]
                existing_sample = merged_data[idx]

                # 检查 head 和 tail 是否已存在于 entity_list
                head_entity = {"name": head["name"], "pos": head["pos"]}
                tail_entity = {"name": tail["name"], "pos": tail["pos"]}
                
                head_id = next((i for i, e in enumerate(existing_sample["entity_list"]) if e == head_entity), None)
                if head_id is None:
                    head_id = len(existing_sample["entity_list"])
                    existing_sample["entity_list"].append(head_entity)

                tail_id = next((i for i, e in enumerate(existing_sample["entity_list"]) if e == tail_entity), None)
                if tail_id is None:
                    tail_id = len(existing_sample["entity_list"])
                    existing_sample["entity_list"].append(tail_entity)
                
                # 检查是否已存在相同的 head_id 和 tail_id 但关系不同的情况
                entity_pair = [head_id, tail_id, rel2id[relation], dataid]
                if not any(ep[0] == head_id and ep[1] == tail_id and ep[2] != rel2id[relation] for ep in existing_sample["entity_pair_list"]):
                    existing_sample["entity_pair_list"].append(entity_pair)
                else:
                    # 如果关系不同，允许重叠的实体对添加到 entity_pair_list
                    for ep in existing_sample["entity_pair_list"]:
                        if ep[0] == head_id and ep[1] == tail_id and ep[2] != rel2id[relation]:
                            # 如果已经有相同实体对但关系不同，则添加新关系对
                            existing_sample["entity_pair_list"].append(entity_pair)
                            break
                
                # 合并 imgids
                if imgid not in existing_sample["imgids"]:
                    existing_sample["imgids"].append(imgid)
                
                # 合并 aux_imgs
                if dataid in data["aux_imgs"]:
                    if data["aux_imgs"][dataid] not in existing_sample["aux_imgs"]:
                        existing_sample["aux_imgs"].extend(data["aux_imgs"][dataid])

                # 合并 rcnn_imgs
                if imgid_key in data["rcnn_imgs"]:
                    for crop in data["rcnn_imgs"][imgid_key]:
                        if crop not in existing_sample["rcnn_imgs"]:
                            existing_sample["rcnn_imgs"].append(crop)
            else:
                # 创建新的样本
                head_entity = {"name": head["name"], "pos": head["pos"]}
                tail_entity = {"name": tail["name"], "pos": tail["pos"]}
                
                new_sample = {
                    "words": words,
                    "entity_list": [head_entity, tail_entity],
                    "entity_pair_list": [[0, 1, rel2id[relation], dataid]],
                    "imgids": [imgid],
                    "aux_imgs": data["aux_imgs"].get(dataid, []),
                    "rcnn_imgs": data["rcnn_imgs"].get(imgid_key, []),
                    "merged_id":idx
                }
                merged_data.append(new_sample)
                sentence_idx[sentence_hash] = len(merged_data) - 1

        return merged_data
    
    def filter_entity_pairs_with_inconsistence_rel(self, merged_data, del_no_rel_ins=True, del_ratio=0.3):
        """
        删除矛盾的关系，比如标记的有其他关系，又存在一个样本标记没有关系，同时删除只有一个关系对且标记为没有关系的样本，根据句子长短排序后，按照比例删除

        参数：
        - entity_pair_list: 原始的四元组列表，格式为 [头实体ID, 尾实体ID, 关系ID, 原始数据ID]
        - del_no_rel_ins = ['single','double','triple']: 删除样本中只有一个no关系的样本
        - del_ratio=0.3: 删除的比例

        返回：
        - filtered_list: 过滤后的四元组列表
        - deleted_duplicate_id: 被删除样本的原始数据ID列表
        """
        from collections import defaultdict

        deleted_duplicate_id = []
        deleted_single_no_rel_id=[]
        # 存储每对实体的所有关系
        for id, item in enumerate(merged_data): 
            entity_dict = defaultdict(list)
            entity_pair_list = item["entity_pair_list"]
            
            
            # 构建字典 { (头实体, 尾实体): [(关系ID, 原始数据ID)] }
            for head, tail, relation, orig_id in entity_pair_list:
                entity_dict[(head, tail)].append((relation, orig_id))
            
            # 过滤操作
            filtered_list = []

            
            for (head, tail), relations in entity_dict.items():
                # 筛选出所有非0关系的条目
                non_zero_relations = [item for item in relations if item[0] != 0]
                
                if non_zero_relations:
                    # 如果有非0关系，保留这些条目
                    for relation, orig_id in non_zero_relations:
                        filtered_list.append([head, tail, relation, orig_id])
                    # 删除所有关系为0的条目，并记录原始ID
                    zero_relations = [item for item in relations if item[0] == 0]
                    deleted_duplicate_id.extend([item[1] for item in zero_relations])
                else:
                    # 如果没有非0关系，保留关系为0的条目
                    for relation, orig_id in relations:
                        filtered_list.append([head, tail, relation, orig_id])
            merged_data[id]['entity_pair_list'] = filtered_list

            # 根据 entity_pair_list 长度,按照(merged_id号，句子长度)记录
            if len(item['entity_pair_list'])==1 and item['entity_pair_list'][0][2]==0:
                deleted_single_no_rel_id.append((id,len(item['words'])))
        

        if del_ratio > 0 and del_no_rel_ins:
            # 按句子长度升序排序（假设 item 中有 length 属性表示句子长度）
            deleted_single_no_rel_id=sorted(deleted_single_no_rel_id, key=lambda x: x[1])
            num_to_delete = int(len(deleted_single_no_rel_id) * del_ratio)
            deleted_single_no_rel_id = deleted_single_no_rel_id[:num_to_delete]
            deleted_single_no_rel_id = [item[0] for item in deleted_single_no_rel_id]
            # 删除样本并记录 ID

            merged_data = np.array(merged_data)
            merged_data = np.delete(merged_data, deleted_single_no_rel_id)
            merged_data = merged_data.tolist()
            
            #deleted_duplicate_id
            #deleted_single_no_rel_id
        return merged_data


 

class MREOverlapDataset_test(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64,
                 mode="train", write_path=None, do_test=False) -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.rcnn_img_path = 'data/mnre/'
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.aux_processor = self.processor.aux_processor
        self.rcnn_processor = self.processor.rcnn_processor
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        self.write_path = write_path
        self.do_test = do_test

    def __len__(self):
        return len(self.data_dict['words'])



    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
                                                     self.data_dict['heads'][idx], self.data_dict['tails'][idx], \
                                                     self.data_dict['imgids'][idx]
        
        
        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <e0> entity1 </e0> ... <e7> entity7 </e7> .. [SEP]
       
        # insert ['<e0>', '</e0>', '<e1>', '</e1>',...'<e7>', '</e7>']})
        eps=[head_d,tail_d]
        """
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        in_head = False
        in_tail = False
        extend_word_list = []
        in_head = False
        in_tail = False

        for i, word in enumerate(word_list):
            if i == head_pos[0]:
                in_head = True
                extend_word_list.append('<s>')
            elif i == head_pos[1] and in_head:
                in_head = False
                extend_word_list.append('</s>')
                
            if i == tail_pos[0]:
                in_tail = True
                extend_word_list.append('<o>')
            elif i == tail_pos[1] and in_tail:
                in_tail = False
                extend_word_list.append('</o>')
                
            extend_word_list.append(word)

        # 检查是否需要在末尾添加 '</s>' 或 '</o>'
        if in_head:  # 如果还在头实体中，说明头实体是最后一个单词
            extend_word_list.append('</s>')
        if in_tail:  # 如果还在尾实体中，说明尾实体是最后一个单词
            extend_word_list.append('</o>')
        """

        
        #extend_word_list = self.add_entity_labels(word_list, eps)       
        
        extend_word_list = " ".join(word_list)

        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True,
                                                 padding='max_length',return_offsets_mapping=True)
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)
        
        offsets = encode_dict['offset_mapping']
        re_label = self.re_dict[relation]  # label to id
        #将标签全部填充为关系数量大小,因为在overlapp中label的标签会转化为tensor，他们长度必须一致
        #label_size = len(self.processor.get_relation_dict().keys())
        #re_label.extend([0] * (label_size - len(re_label))) if len(re_label) < label_size else re_label
        epl=[[0,1,re_label]]

        head_d['pos'] = self.get_new_entity_position_v2(word_list,head_d,offsets)
        self.data_dict['heads'][idx]['pos']=head_d['pos']
        tail_d['pos'] = self.get_new_entity_position_v2(word_list,tail_d,offsets)
        self.data_dict['tails'][idx]['pos']=tail_d['pos']
        #临时转换成overlap 数据类型
        temp_head=head_d
        temp_tail=tail_d
        temp_head['pos']=[head_d['pos']]
        temp_tail['pos']=[tail_d['pos']]
        entities = [temp_head,temp_tail]
        #entities_positon = self.get_entities_positions(word_list,entities,offsets)

        matrix_lable = self.merge_one_hot_labels_pos(epl, input_ids, entities)
        
        # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            if self.aux_img_path is not None:
                # detected object img
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]

                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                    aux_imgs.append(aux_img)

                # padding
                for i in range(3 - len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    if imgid in self.data_dict['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict['rcnn_imgs'][imgid]
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]

                    # select 3 img
                    for i in range(min(3, len(rcnn_img_paths))):
                        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                        rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                        rcnn_imgs.append(rcnn_img)

                    # padding
                    for i in range(3 - len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    if self.write_path is not None and self.mode == 'test' and self.do_test:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, extend_word_list, imgid, matrix_lable
                    else:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, matrix_lable

                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs, matrix_lable  #aux_imgs are the corped imgs
            #return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, matrix_lable
        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), matrix_lable

    def merge_one_hot_labels(self, entity_pair_list, input_ids, tokenizer=None):
        seq_len = len(input_ids)
        #initial zero matrix labels
        matrix_labels = torch.zeros((len(self.re_dict), seq_len, seq_len))
        for ep in entity_pair_list:
            #generate one matrix label for one entity pair
            em = MREOverlapDataset_test.get_entity_mask(input_ids, self.processor.entity_marker,ep)
            #em = MREOverlapDataset_test.get_entity_mask_ori(input_ids, self.processor.entity_marker)
            rel_id = ep[2]
            matrix_labels[rel_id] += em

        # keep all emements in the matraix are 0 or 1
        if matrix_labels.max().item() > 1:
            matrix_labels = (matrix_labels > 0).float()
        return matrix_labels
    
    def merge_one_hot_labels_pos(self, entity_pair_list, input_ids, entities):
        seq_len = len(input_ids)
        #initial zero matrix labels
        matrix_labels = torch.zeros((len(self.re_dict), seq_len, seq_len))
        for ep in entity_pair_list:
            #generate one matrix label for one entity pair
            em = MREOverlapDataset_test.get_entity_mask_pos(input_ids, ep, entities)
            #em = MREOverlapDataset_test.get_entity_mask_ori(input_ids, self.processor.entity_marker)
            rel_id = ep[2]
            matrix_labels[rel_id] += em

        # keep all emements in the matraix are 0 or 1
        if matrix_labels.max().item() > 1:
            matrix_labels = (matrix_labels > 0).float()
        return matrix_labels

    def get_entity_mask_ori(input_ids,entity_marker):
        seq_len = input_ids.shape[-1]
        #initial zero tensor
        entity_1 = torch.zeros((seq_len))
        entity_2 = torch.zeros((seq_len))
        #get corresponding idx of start and end positions
        head_pos,tail_pos = [entity_marker['<e0>'],entity_marker['<e0/>']], [entity_marker['<e1>'],entity_marker['<e1/>']]
        head_start_idx = input_ids.eq(head_pos[0]).nonzero().item()
        head_end_idx = input_ids.eq(head_pos[1]).nonzero().item()+1
        tail_start_idx = input_ids.eq(tail_pos[0]).nonzero().item()
        tail_end_idx = input_ids.eq(tail_pos[1]).nonzero().item()+1

        entity_1[head_start_idx:head_end_idx] = 1
        entity_2[tail_start_idx:tail_end_idx] = 1

        res = entity_1.unsqueeze(1).repeat_interleave(seq_len, dim=1)
        res += entity_2.unsqueeze(1).repeat_interleave(seq_len, dim=1).t()
        
        res = (res == 2).float()
        return res
    
    def get_entity_mask_pos(input_ids, entity_pair, entities):
        seq_len = input_ids.shape[-1]
        #initial zero tensor
        entity_1 = torch.zeros((seq_len))
        entity_2 = torch.zeros((seq_len))
        #get corresponding idx of start and end positions
        #head_pos,tail_pos = [entity_marker['<e0>'],entity_marker['<e0/>']], [entity_marker['<e1>'],entity_marker['<e1/>']]
        head_start_idx = entities[entity_pair[0]]['pos'][0][0]
        head_end_idx = entities[entity_pair[0]]['pos'][0][1]
        tail_start_idx = entities[entity_pair[1]]['pos'][0][0]
        tail_end_idx = entities[entity_pair[1]]['pos'][0][1]

        #head_start_idx = positions.eq(entity_pair[0]+1).nonzero()[0].item()
        #head_end_idx = positions.eq(entity_pair[0]+1).nonzero()[-1].item()+1
        #tail_start_idx = positions.eq(entity_pair[1]+1).nonzero()[0].item()
        #tail_end_idx = positions.eq(entity_pair[1]+1).nonzero()[-1].item()+1

        entity_1[head_start_idx:head_end_idx] = 1
        entity_2[tail_start_idx:tail_end_idx] = 1

        res = entity_1.unsqueeze(1).repeat_interleave(seq_len, dim=1)
        res += entity_2.unsqueeze(1).repeat_interleave(seq_len, dim=1).t()
        
        res = (res == 2).float()
        return res
    
    @staticmethod
    def get_entity_mask(input_ids,entity_marker, entity_pair):
        seq_len = input_ids.shape[-1]
        #initial zero tensor
        entity_1 = torch.zeros((seq_len))
        entity_2 = torch.zeros((seq_len))
        #get corresponding idx of start and end positions
        #head_pos,tail_pos = [entity_marker[f"<e{entity_pair[0]}>"],entity_marker[f"<e{entity_pair[0]}/>"]], \
        #                    [entity_marker[f"<e{entity_pair[1]}>"],entity_marker[f"<e{entity_pair[1]}/>"]]
        head_pos,tail_pos = [entity_marker[f"<e>"],entity_marker[f"<s>"]], \
                            [entity_marker[f"<e>"],entity_marker[f"<s>"]]

        head_start_idx = input_ids.eq(head_pos[0]).nonzero()[entity_pair[0]].item()
        head_end_idx = input_ids.eq(head_pos[1]).nonzero()[entity_pair[0]].item()+1
        tail_start_idx = input_ids.eq(tail_pos[0]).nonzero()[entity_pair[1]].item()
        tail_end_idx = input_ids.eq(tail_pos[1]).nonzero()[entity_pair[1]].item()+1

        entity_1[head_start_idx:head_end_idx] = 1
        entity_2[tail_start_idx:tail_end_idx] = 1

        res = entity_1.unsqueeze(1).repeat_interleave(seq_len, dim=1)
        res += entity_2.unsqueeze(1).repeat_interleave(seq_len, dim=1).t()
        
        res = (res == 2).float()
        return res
    
    def add_entity_labels(self, word_list, entities):
        """
        给定一个词列表和多个实体位置信息，在对应的实体周围添加带编号的标签（<e0>, </e0>, <e1>, </e1>, ...）。
        
        参数：
        - word_list: 句子的词列表
        - entities: 实体列表，格式为 [{'name': 'entity_name', 'pos': [start, end]}, ...]。
        其中每个实体位置是一个字典，包含 `name`（实体名称）和 `pos`（实体位置的起始和结束索引，格式为 [start, end]）。

        返回：
        - extend_word_list: 经过修改后的词列表，其中包括带编号的实体标签。
        """
        # 初始化扩展后的词列表
        extend_word_list = []

        # 遍历 word_list 的索引和内容，同时检查是否需要插入实体标签
        for i, word in enumerate(word_list):
            # 检查是否是某个实体的开始或结束位置
            for idx, entity in enumerate(entities):
                start, end = entity['pos']
                if i == start:  # 起始位置，插入开标签
                    extend_word_list.append(f'<e>')
                if i == end:  # 结束位置，插入闭标签
                    extend_word_list.append(f'<s>')
            
            # 添加当前词
            extend_word_list.append(word)
        
        # 检查是否有未关闭的标签（理论上不会出现，但加以保护）
        for idx, entity in enumerate(entities):
            if entity['pos'][1] == len(word_list):  # 如果实体的结束位置是句子的最后一个单词
                extend_word_list.append(f'<s>')

        return extend_word_list   

    def get_entities_positions(self, word_list, entities, offsets):
        """
        给定实体列表包含多个实体位置信息，返回实体位置信息sentence = "Barack Obama was born in Hawaii."
            position_embeddings = [1, 1, 0, 0, 0, 2, 0]  # Head = 1, Tail = 2。
        
        参数：
        - word_list: 句子单词列表，格式为['Barack', 'Obama', 'was', 'born', 'in', 'Hawaii','.']
        - epl: 实体列表，格式为 [{'name': 'entity_name', 'pos': [start, end]}, ...]。
        其中每个实体位置是一个字典，包含 `name`（实体名称）和 `pos`（实体位置的起始和结束索引，格式为 [start, end]）。
        - offsets: 经过tokenizer 转换过后的map

        返回：
        - position:  [1, 1, 0, 0, 0, 2, 0] # Head = 1, Tail = 2。多个实体则一次类推
        """ 
        # 初始化 position 列表，所有位置默认为 0
        position = [0] * len(offsets)

        # 遍历实体列表，标记实体的起始和结束位置
        for idx, entity in enumerate(entities):
            # 获取实体的起始和结束索引（基于 word_list）
            start, end = entity['pos']

            # 遍历 offsets，对应到 tokenizer 分词后的 token
            for i, (start_offset, end_offset) in enumerate(offsets):
                if start_offset == 0 and end_offset == 0:  # 跳过特殊 token
                    continue
                if start_offset >= len(" ".join(word_list[:start])) and end_offset <= len(" ".join(word_list[:end])):
                    position[i] = idx + 1  # 标记为实体的索引 + 1

        return position

    def get_new_entity_position_v2(self,word_list, entity, offsets):
        """
        将实体在 word_list 中的位置信息转化为 tokenizer 分词后的新位置。

        Args:
            word_list (list): 原始的单词列表。
            entity (dict): 单个实体信息，包含 'name' 和 'pos'（起始和结束索引）。
            offsets (list): tokenizer 的 offset_mapping，表示每个 token 的起止位置。

        Returns:
            tuple: 实体在 tokenizer 分词后的位置范围 (new_start, new_end)。
        """
        # 获取实体的起始和结束索引
        start_idx, end_idx = entity['pos']
        
        # 计算实体在文本中的字符范围
        text = " ".join(word_list)  # 将 word_list 还原为完整句子
        entity_text = " ".join(word_list[start_idx:end_idx])  # 获取实体对应的子串
        entity_start_char = text.find(entity_text)  # 实体起始字符位置
        entity_end_char = entity_start_char + len(entity_text)  # 实体结束字符位置
        
        # 初始化新的位置
        new_start, new_end = -1, -1

        # 遍历 offsets，找到分词后实体的范围
        for i, (start_offset, end_offset) in enumerate(offsets):
            if start_offset == 0 and end_offset == 0:  # 跳过特殊 token
                continue
            # 如果当前 token 的范围包含实体的起始位置
            if new_start == -1 and start_offset <= entity_start_char < end_offset:
                new_start = i
            # 如果当前 token 的范围包含实体的结束位置
            if start_offset < entity_end_char <= end_offset:
                new_end = i + 1  # 结束位置是开区间
                break

        return [new_start, new_end]        
        