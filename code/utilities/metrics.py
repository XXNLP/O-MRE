
import torch
import random
import pandas as pd
import numpy as np
from processor import MREDataset,MREOverlapDataset, MREOverlapDataset_test
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support, accuracy_score, confusion_matrix, multilabel_confusion_matrix,top_k_accuracy_score
CORRECT = "correct"
TOTAL = "total"
TRUE_POSITIVE = "ture_positive"
FALSE_NEGATIVE = "false_neative"
FALSE_POSITIVE = "false_positive"
TRUE_NEGATIVE = "true_negative"
COMPLETE_CORRECT = "complete_correct"  #in multi label predicte label and gold label must be complete


def eval_result(true_labels, pred_result, rel2id, logger, use_name=False):
    correct = 0
    total = len(true_labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
        if name in rel2id:
            if use_name:
                neg = name
            else:
                neg = rel2id[name]
            break
    for i in range(total):
        if use_name:
            golden = true_labels[i]
        else:
            golden = true_labels[i]

        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0

    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    logger.info('Evaluation result: {}.'.format(result))
    return result


class F1Metric(object):
    def __init__(self, multi_label=False, na_id=-1, ignore_na=True, print_error_prob=0, rel2id=None, entity_marker=None):
        self.print_error_prob = print_error_prob
        self.multi_label = multi_label if len(entity_marker)==4 else True
        self.na_id = na_id
        for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
            if name in rel2id:
                if ignore_na:
                    self.na_id=rel2id[name]
                break
        self.entity_marker = entity_marker
        self.ignore_na = ignore_na
        self.id2rel = None
        self.rel2id = rel2id
        if rel2id is not None:
            self.id2rel = {value: key for key, value in rel2id.items()}

        self.triple_count = 5
        self.mention_count = 1
        self.reset()
        self.matrix = []
        self.tsne_data=[]
    
    @staticmethod
    def get_dynamic_mask(logist_matrix, threshold=0.5):
        #递归深度限制运行
        def findContinuousRegions_ver1(matrix):
            #if not matrix or not matrix[0]:
            #    return []

            rows, cols = len(matrix), len(matrix[0])
            visited = [[False] * cols for _ in range(rows)]
            regions = []

            def dfs(r, c):
                if r < 0 or r >= rows or c < 0 or c >= cols or matrix[r][c] == 0 or visited[r][c]:
                    return None
                visited[r][c] = True
                region = {"top_left": (r, c), "bottom_right": (r, c), "area": 1}
                
                # 递归地搜索相邻的1
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    neighbor_region = dfs(nr, nc)
                    if neighbor_region:
                        # 如果找到相邻区域，合并区域
                        region["area"] += neighbor_region["area"]
                        # 更新区域的边界
                        region["top_left"] = (min(region["top_left"][0], neighbor_region["top_left"][0]),
                                                min(region["top_left"][1], neighbor_region["top_left"][1]))
                        region["bottom_right"] = (max(region["bottom_right"][0], neighbor_region["bottom_right"][0]),
                                                    max(region["bottom_right"][1], neighbor_region["bottom_right"][1]))
                        # 将相邻区域标记为已处理
                        neighbor_region["area"] = 0
                return region

            for r in range(rows):
                for c in range(cols):
                    if matrix[r][c] == 1 and not visited[r][c]:
                        
                        region = dfs(r, c)
                        if region and region["area"] > 0:
                            regions.append(region)

            regions.sort(key=lambda x: x['area'], reverse=True)
            return regions
        
        def findContinuousRegions(matrix):
            #if not matrix or not matrix[0]:
            #    return []

            rows, cols = len(matrix), len(matrix[0])
            visited = [[False] * cols for _ in range(rows)]
            regions = []

            def iterate_dfs(r, c, region):
                stack = [(r, c)]
                region["area"] = 0
                region["top_left"] = (r, c)
                region["bottom_right"] = (r, c)

                while stack:
                    nr, nc = stack.pop()
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and matrix[nr][nc] == 1:
                        visited[nr][nc] = True
                        region["area"] += 1
                        region["top_left"] = (min(region["top_left"][0], nr), min(region["top_left"][1], nc))
                        region["bottom_right"] = (max(region["bottom_right"][0], nr), max(region["bottom_right"][1], nc))
                        # 将相邻节点压入栈中
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            stack.append((nr + dr, nc + dc))

            for r in range(rows):
                for c in range(cols):
                    if matrix[r][c] == 1 and not visited[r][c]:
                        region = {"top_left": (r, c), "bottom_right": (r, c), "area": 0}
                        iterate_dfs(r, c, region)
                        if region["area"] > 0:
                            regions.append(region)

            #regions.sort(key=lambda x: x['area'], reverse=True)
            return regions
        
        def dynamic_mask(seq_len, top_left, bottom_right):
            #initial zero tensor
            entity_1 = torch.zeros((seq_len))
            entity_2 = torch.zeros((seq_len))

            head_pos,tail_pos = (top_left[0],bottom_right[0]),(top_left[1],bottom_right[1])
            head_start_idx = head_pos[0]
            head_end_idx = head_pos[1]
            tail_start_idx = tail_pos[0]
            tail_end_idx = tail_pos[1]
            
            entity_1[head_start_idx:head_end_idx] = 1
            entity_2[tail_start_idx:tail_end_idx] = 1

            res = entity_1.unsqueeze(1).repeat_interleave(seq_len, dim=1)
            res += entity_2.unsqueeze(1).repeat_interleave(seq_len, dim=1).t()
            
            res = (res == 2).float()
            return res
        
        default_region = [{"top_left": (1,1), "bottom_right":(127,127)}]

        seq_len = logist_matrix.shape[-1]
        logist_matrix = (logist_matrix > threshold).float()
        regions = []
        for i in logist_matrix:
            regions.extend(findContinuousRegions(i))
        regions.sort(key=lambda x: x['area'], reverse=True)
        regions = regions if len(regions)>0 else default_region
        top_left,bottom_right = regions[0]["top_left"],regions[0]["bottom_right"]
        dynamic_mask = dynamic_mask(seq_len, top_left,bottom_right)
        return dynamic_mask 
    
    # dynamic mask method for evaluation
    def eval_for_dynamic(self, pred_result_logits, data_list: list, input_ids, start_id=-1, end_id=-1):
        def _split_data_list(start_id,end_id):
            '''
            for mnre:
                parameters: data_list: {'words': [words], 'relations': [relations], 'heads': [heads], 'tails': [tails], 'imgids': [imgids],
                    'dataid': [dataid], 'aux_imgs': [aux_imgs], "rcnn_imgs": [rcnn_imgs]}
                    start_id: int
                    end_id: int
                
                return: data_list: {'token':[],
                                    'entity_pair_list':[[head,tail,r],[]],
                                    'entity_list:['pos':[[start,end]],'name':[[entity1],[entity2]]]}
            '''
            token = data_list['words'][start_id:end_id]
            head = data_list['heads'][start_id:end_id]
            tail = data_list['tails'][start_id:end_id]
            relations = data_list['relations'][start_id:end_id]

            entity_pair_list = [[[0,1,self.rel2id[r]]] for _, _, r in zip(head, tail, relations)]
            entity_list = [[{'pos':[h['pos']], 'name':[h['name']]}
                            ,{'pos':[t['pos']], 'name':[t['name']]}] for h,t in zip(head, tail)]
            data = [{'token':tok, 'entity_pair_list': epl, 'entity_list':el ,'labels': la} for tok,epl,el,la in zip(token,entity_pair_list,entity_list,relations)]
            return data

        if start_id!=-1 and end_id!=-1: # chang data format
            data_list = _split_data_list(start_id, end_id)

        preds_one_hot = []
        golds_one_hot = []

        preds_one_hot_without_na = []
        golds_one_hot_without_na = []

        normal_id_one_hot = []
        overlap_id_one_hot = []
        multi_id_one_hot = []
        without_na_one_hot = []
        
        mention_count_id = [[] for _ in range(self.mention_count)]
        triple_count_id = [[] for _ in range(self.triple_count)]

        batch_pred_labels = []
        batch_true_labels = []
        batch_pred_logits = []

        
        #self.matrix.extend(pred_result)
        #self.tsne_data.extend(data_list)
            
        pred_result = (pred_result_logits.detach().cpu().numpy())  # B*NL*SL*SL
    
        for i, d in enumerate(data_list):
            is_print = random.random() < self.print_error_prob
            epl_c = d["entity_pair_list"].copy()
            epl_c = [ep for ep in epl_c if ep[2] != self.na_id]

            is_normal_data = F1Metric.is_normal(epl_c)
            is_multi_label_data = F1Metric.is_multi_label(epl_c)
            is_over_lapping_data = F1Metric.is_over_lapping(epl_c)
            triple_count = len(epl_c)
            triple_count = min(triple_count, len(self.triple_count_res) - 1)

            #test mentions resutl
            mention_count = max([len(i['pos']) for i in d['entity_list']])
            mention_count = min(mention_count, len(self.mention_count_res) - 1)

            checked_epl_id = []
            epl = d["entity_pair_list"]
            #sl = d["sdp_list"]
            #sdp_list = list(set([i for x in sl for i in x]))
            for e_idx, ep in enumerate(epl):
                if e_idx in checked_epl_id:
                    continue
                checked_epl_id.append(e_idx)
                pr = torch.from_numpy(pred_result[i])
                em = F1Metric.get_dynamic_mask(pr)
                #em = MREDataset.get_entity_mask(input_ids[i],self.entity_marker).to(pr.device)  # SL*SL
                gold_label = [ep[2]]
                for _e_idx, _ep in enumerate(epl):
                    if _e_idx in checked_epl_id:
                        continue
                    if _ep[0] == ep[0] and _ep[1] == ep[1]:
                        gold_label.append(_ep[2])
                        checked_epl_id.append(_e_idx)
    
                if len(pr.shape)>1:  # use matrix label with em
                    _res = ((pr * em).sum(dim=(1, 2)) / em.sum()).cpu().numpy()  # NL
                else:
                    _res = pr.cpu().numpy()
                
                # store every predicted logits value
                pred_logits = _res
                
                res = [0] * len(_res)
                #res = torch.zeros_like(_res) 
                if self.multi_label:
                    if self.na_id > -1 and _res[self.na_id] > 0.5:
                        res[self.na_id] = 1
                    else:
                        # res[self.na_id] = 0
                        res = (_res > 0.5).astype(int)  # NL
                else:
                    res = _res.copy()
                    res = (res == max(_res)).astype(int)
                    if sum(res)==0:
                        res[self.na_id]=1 
                

                gold = [0] * len(_res)
                gold = np.array(gold,np.float32)
                gold[gold_label] = 1

                preds_one_hot.append(res)
                golds_one_hot.append(gold)

                #for without_na
                res_na = res.copy()
                gold_na = gold.copy()
                
                if self.na_id>-1:
                    res_na[self.na_id] = 0 
                    gold_na[self.na_id] = 0

                preds_one_hot_without_na.append(res_na)
                golds_one_hot_without_na.append(gold_na)

                # for without na relation
                na_relation_id = [0]+[1]*(len(_res)-1)
                without_na_one_hot.append(na_relation_id) 
                # for normal data
                normal_id = [1]*len(_res) if is_normal_data else [0]*len(_res)
                normal_id_one_hot.append(normal_id) 
                # for overlapping data
                overlap_id = [1]*len(_res) if is_over_lapping_data else [0]*len(_res)
                overlap_id_one_hot.append(overlap_id) 
                # for multi data
                multi_id = [1]*len(_res) if is_multi_label_data else [0]*len(_res)
                multi_id_one_hot.append(multi_id) 
                # for triple count
                for count in range(len(triple_count_id)):
                    m_i = [1]*len(_res) if count==triple_count else [0]*len(_res)
                    triple_count_id[count].append(m_i)
                
                # for mention count
                for count in range(len(mention_count_id)):
                    m_i = [1]*len(_res) if count==mention_count else [0]*len(_res)
                    mention_count_id[count].append(m_i)


                gold_label_str = [g if self.id2rel is None else self.id2rel[g] for g in gold_label]

                # get predicted and gold labels correspoding ids, if predicted is multi label the gold label should get equal gold labels
                pred_labels = [idx for idx, s in enumerate(res) if s == 1]
                gold_labels = [idx for idx, s in enumerate(gold) if s == 1]
                if self.multi_label:
                    multi_true_label = ((res+gold)!=0).astype(int)
                    gold_labels = [idx if idx in gold_labels else self.na_id for idx, s in enumerate(multi_true_label) if s == 1]
                    pred_labels = [idx if idx in pred_labels else self.na_id for idx, s in enumerate(multi_true_label) if s == 1]

                assert len(pred_labels)==len(gold_labels)

                pred_str = [idx if self.id2rel is None else self.id2rel[idx] for idx in pred_labels]

                #self.df = self.df.append({"Text": " ".join(d["token"]),
                #                        "Subject": d["entity_list"][ep[0]],
                #                        "Object": d["entity_list"][ep[1]],
                #                        "Gold Label": ",".join(gold_label_str),
                #                       "Predict": ",".join(pred_str),
                #                        }, ignore_index=True, )

                #preds = pred_result_logits.argmax(-1)
                
                batch_true_labels.extend(gold_labels)
                batch_pred_labels.extend(pred_labels)
                batch_pred_logits.append(pred_logits)
        
        
        self.re_pred_one_hot_labels.extend(golds_one_hot_without_na)
        self.re_true_one_hot_labels.extend(preds_one_hot_without_na)

        self.re_true_labels.extend(batch_true_labels)
        self.re_pred_labels.extend(batch_pred_labels)
        self.re_pred_logits.extend(batch_pred_logits)
        #pred_logits = torch.tensor(pred_logits)
        labels = torch.tensor(batch_true_labels)
        #self.re_pred_logits.extend(pred_logits)
        re_pred_ranks = 1 + torch.argsort(torch.argsort(torch.tensor(batch_pred_logits), dim=1, descending=True), dim=1,
                                            descending=False)[torch.arange(labels.shape[0]), labels]
        re_pred_ranks = re_pred_ranks.float()
        for rel_id in range(len(self.rel2id)):
            ranks = re_pred_ranks[labels == rel_id]
            for k in range(len(self.rel2id)):
                self.hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + self.hits[rel_id, k + 1]
                
            #"""


    #for mmore
    def eval_ori(self, pred_result_logits, data_list: list, input_ids, start_id=-1, end_id=-1):
        def _split_data_list(start_id,end_id):
            '''
            for mnre:
                parameters: data_list: {'words': [words], 'relations': [relations], 'heads': [heads], 'tails': [tails], 'imgids': [imgids],
                    'dataid': [dataid], 'aux_imgs': [aux_imgs], "rcnn_imgs": [rcnn_imgs]}
                    start_id: int
                    end_id: int
                
                return: data_list: {'token':[],
                                    'entity_pair_list':[[head,tail,r],[]],
                                    'entity_list:['pos':[[start,end]],'name':[[entity1],[entity2]]]}
            '''
            # using original data
            if isinstance(data_list, dict):
                token = data_list['words'][start_id:end_id]
                head = data_list['heads'][start_id:end_id]
                tail = data_list['tails'][start_id:end_id]
                relations = data_list['relations'][start_id:end_id]

                entity_pair_list = [[[0,1,self.rel2id[r]]] for _, _, r in zip(head, tail, relations)]
                entity_list = [[{'pos':[h['pos']], 'name':[h['name']]}
                                ,{'pos':[t['pos']], 'name':[t['name']]}] for h,t in zip(head, tail)]
                data = [{'token':tok, 'entity_pair_list': epl, 'entity_list':el ,'labels': la} for tok,epl,el,la in zip(token,entity_pair_list,entity_list,relations)]
                return data
            # using overlapping data
            else:
                data= data_list[start_id:end_id]
                return data

        if start_id!=-1 and end_id!=-1: # change data format
            data_list = _split_data_list(start_id, end_id)

        preds_one_hot = []
        golds_one_hot = []

        preds_one_hot_without_na = []
        golds_one_hot_without_na = []

        normal_id_one_hot = []
        overlap_id_one_hot = []
        multi_id_one_hot = []
        without_na_one_hot = []
        
        mention_count_id = [[] for _ in range(self.mention_count)]
        triple_count_id = [[] for _ in range(self.triple_count)]

        batch_pred_labels = []
        batch_true_labels = []
        batch_pred_logits = []

        
        #self.matrix.extend(pred_result)
        #self.tsne_data.extend(data_list)
            
        pred_result = (pred_result_logits.detach().cpu().numpy())  # B*NL*SL*SL
    
        for i, d in enumerate(data_list):
            is_print = random.random() < self.print_error_prob
            epl_c = d["entity_pair_list"].copy()
            epl_c = [ep for ep in epl_c if ep[2] != self.na_id]

            is_normal_data = F1Metric.is_normal(epl_c)
            is_multi_label_data = F1Metric.is_multi_label(epl_c)
            is_over_lapping_data = F1Metric.is_over_lapping(epl_c)
            triple_count = len(epl_c)
            triple_count = min(triple_count, len(self.triple_count_res) - 1)

            #test mentions resutl
            mention_count = max([len(i['pos']) for i in d['entity_list']])
            mention_count = min(mention_count, len(self.mention_count_res) - 1)

            checked_epl_id = []
            epl = d["entity_pair_list"]
            #sl = d["sdp_list"]
            #sdp_list = list(set([i for x in sl for i in x]))
            for e_idx, ep in enumerate(epl):
                if e_idx in checked_epl_id:
                    continue
                checked_epl_id.append(e_idx)
                pr = torch.from_numpy(pred_result[i])
                #em = MREOverlapDataset.get_entity_mask(input_ids[i],self.entity_marker,ep).to(pr.device)  # SL*SL
                #em = MREDataset.get_entity_mask(input_ids[i],self.entity_marker)
                #em = MREOverlapDataset_test.get_entity_mask_ori(input_ids[i], self.entity_marker).to(pr.device)

                entities_positon = torch.tensor(self.get_entities_positions(d['tokens'],d['entity_list'],offsets))
                em = MREOverlapDataset_test.get_entity_mask(input_ids[i],self.entity_marker,ep).to(pr.device)  # SL*SL
                gold_label = [ep[2]]
                for _e_idx, _ep in enumerate(epl):
                    if _e_idx in checked_epl_id:
                        continue
                    if _ep[0] == ep[0] and _ep[1] == ep[1]:
                        gold_label.append(_ep[2])
                        checked_epl_id.append(_e_idx)
    
                if len(pr.shape)>1:  # use matrix label with em
                    _res = ((pr * em).sum(dim=(1, 2)) / em.sum()).cpu().numpy()  # NL
                else:
                    _res = pr.cpu().numpy()
                
                # store every predicted logits value
                pred_logits = _res
                
                res = [0] * len(_res)
                #res = torch.zeros_like(_res) 
                if self.multi_label:
                    if self.na_id > -1 and _res[self.na_id] > 0.5:
                        res[self.na_id] = 1
                    else:
                        # res[self.na_id] = 0
                        res = (_res > 0.5).astype(int)  # NL
                else:
                    res = _res.copy()
                    res = (res == max(_res)).astype(int)
                

                gold = [0] * len(_res)
                gold = np.array(gold,np.float32)
                gold[gold_label] = 1

                preds_one_hot.append(res)
                golds_one_hot.append(gold)

                #for without_na
                res_na = res.copy()
                gold_na = gold.copy()
                
                if self.na_id>-1:
                    res_na[self.na_id] = 0 
                    gold_na[self.na_id] = 0

                preds_one_hot_without_na.append(res_na)
                golds_one_hot_without_na.append(gold_na)

                # for without na relation
                na_relation_id = [0]+[1]*(len(_res)-1)
                without_na_one_hot.append(na_relation_id) 
                # for normal data
                normal_id = [1]*len(_res) if is_normal_data else [0]*len(_res)
                normal_id_one_hot.append(normal_id) 
                # for overlapping data
                overlap_id = [1]*len(_res) if is_over_lapping_data else [0]*len(_res)
                overlap_id_one_hot.append(overlap_id) 
                # for multi data
                multi_id = [1]*len(_res) if is_multi_label_data else [0]*len(_res)
                multi_id_one_hot.append(multi_id) 
                # for triple count
                for count in range(len(triple_count_id)):
                    m_i = [1]*len(_res) if count==triple_count else [0]*len(_res)
                    triple_count_id[count].append(m_i)
                
                # for mention count
                for count in range(len(mention_count_id)):
                    m_i = [1]*len(_res) if count==mention_count else [0]*len(_res)
                    mention_count_id[count].append(m_i)


                gold_label_str = [g if self.id2rel is None else self.id2rel[g] for g in gold_label]

                # get predicted and gold labels correspoding ids, if predicted is multi label the gold label should get equal gold labels
                pred_labels = [idx for idx, s in enumerate(res) if s == 1]
                gold_labels = [idx for idx, s in enumerate(gold) if s == 1]
                if self.multi_label:
                    multi_true_label = ((res+gold)!=0).astype(int)
                    gold_labels = [idx if idx in gold_labels else self.na_id for idx, s in enumerate(multi_true_label) if s == 1]
                    pred_labels = [idx if idx in pred_labels else self.na_id for idx, s in enumerate(multi_true_label) if s == 1]

                assert len(pred_labels)==len(gold_labels)

                pred_str = [idx if self.id2rel is None else self.id2rel[idx] for idx in pred_labels]

                #self.df = self.df.append({"Text": " ".join(d["token"]),
                #                        "Subject": d["entity_list"][ep[0]],
                #                        "Object": d["entity_list"][ep[1]],
                #                        "Gold Label": ",".join(gold_label_str),
                #                       "Predict": ",".join(pred_str),
                #                        }, ignore_index=True, )

                #preds = pred_result_logits.argmax(-1)
                
                batch_true_labels.extend(gold_labels)
                batch_pred_labels.extend(pred_labels)
                batch_pred_logits.append(pred_logits)
        
        
        self.re_pred_one_hot_labels.extend(golds_one_hot_without_na)
        self.re_true_one_hot_labels.extend(preds_one_hot_without_na)

        self.re_true_labels.extend(batch_true_labels)
        self.re_pred_labels.extend(batch_pred_labels)
        self.re_pred_logits.extend(batch_pred_logits)
        #pred_logits = torch.tensor(pred_logits)
        labels = torch.tensor(batch_true_labels)
        #self.re_pred_logits.extend(pred_logits)

        # count hints
        #re_pred_ranks = 1 + torch.argsort(torch.argsort(torch.tensor(batch_pred_logits), dim=1, descending=True), dim=1,
        #                                    descending=False)[torch.arange(labels.shape[0]), labels]
        #re_pred_ranks = re_pred_ranks.float()
        #for rel_id in range(len(self.rel2id)):
        #    ranks = re_pred_ranks[labels == rel_id]
        #    for k in range(len(self.rel2id)):
        #        self.hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + self.hits[rel_id, k + 1]
                
        #"""


        # for F1 score all
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot,golds_one_hot)
        #self.res[CORRECT] += tp_and_tn
        self.res[TRUE_POSITIVE] += tp
        self.res[FALSE_NEGATIVE] += fn
        self.res[FALSE_POSITIVE] += fp
        self.res[TRUE_NEGATIVE] += tn
        self.res[TOTAL] += total
        #"""
        #for without no relation F1 score
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na, without_na_one_hot)
        #self.without_na_res[CORRECT] += tp_and_tn
        self.without_na_res[TRUE_POSITIVE] += tp
        self.without_na_res[FALSE_NEGATIVE] += fn
        self.without_na_res[FALSE_POSITIVE] += fp
        self.without_na_res[TRUE_NEGATIVE] += tn
        self.without_na_res[TOTAL] += total

        #for normal relation F1 score
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,normal_id_one_hot)
        #self.normal_res[CORRECT] += tp_and_tn
        self.normal_res[TRUE_POSITIVE] += tp
        self.normal_res[FALSE_NEGATIVE] += fn
        self.normal_res[FALSE_POSITIVE] += fp
        self.normal_res[TRUE_NEGATIVE] += tn
        self.normal_res[TOTAL] += total
        #for overlap relation F1 score
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,overlap_id_one_hot)
        #self.over_lapping_res[CORRECT] += tp_and_tn
        self.over_lapping_res[TRUE_POSITIVE] += tp
        self.over_lapping_res[FALSE_NEGATIVE] += fn
        self.over_lapping_res[FALSE_POSITIVE] += fp
        self.over_lapping_res[TRUE_NEGATIVE] += tn
        self.over_lapping_res[TOTAL] += total

        #for multi relation F1 score
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,multi_id_one_hot)
        #self.multi_label_res[CORRECT] += tp_and_tn
        self.multi_label_res[TRUE_POSITIVE] += tp
        self.multi_label_res[FALSE_NEGATIVE] += fn
        self.multi_label_res[FALSE_POSITIVE] += fp
        self.multi_label_res[TRUE_NEGATIVE] += tn
        self.multi_label_res[TOTAL] += total

        #for triple count relation F1 score
        for i in range(self.triple_count):
            tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,triple_count_id[i])
            #self.triple_count_res[i][CORRECT] += tp_and_tn
            self.triple_count_res[i][TRUE_POSITIVE] += tp
            self.triple_count_res[i][FALSE_NEGATIVE] += fn
            self.triple_count_res[i][FALSE_POSITIVE] += fp
            self.mention_count_res[i][TRUE_NEGATIVE] += tn
            self.triple_count_res[i][TOTAL] += total

        #for mention count relation F1 score
        for i in range(self.mention_count):
            tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,mention_count_id[i])    
            #self.mention_count_res[i][CORRECT] += tp_and_tn
            self.mention_count_res[i][TRUE_POSITIVE] += tp
            self.mention_count_res[i][FALSE_NEGATIVE] += fn
            self.mention_count_res[i][FALSE_POSITIVE] += fp
            self.mention_count_res[i][TRUE_NEGATIVE] += tn
            self.mention_count_res[i][TOTAL] += total
        #else: #no single-label
        #    pass

        #    preds = pred_result_logits.argmax(-1)
        #    labels = torch.tensor([self.rel2id[r] for r in data_list['labels']])
        #    self.re_true_labels.extend(labels.view(-1).detach().cpu().tolist())
        #    self.re_pred_labels.extend(preds.view(-1).detach().cpu().tolist())
        #    self.re_pred_logits.extend(pred_result_logits.detach().cpu().tolist())
        #    re_pred_ranks = 1 + torch.argsort(torch.argsort(pred_result_logits, dim=1, descending=True), dim=1,
        #                                        descending=False)[
        #        torch.arange(labels.shape[0]), labels]
        #    re_pred_ranks = re_pred_ranks.float()
        #   for rel_id in range(len(self.rel2id)):
        #        ranks = re_pred_ranks[labels == rel_id]
        #        for k in range(len(self.rel2id)):
        #            self.hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + self.hits[rel_id, k + 1]
            
        #for mmore
    def eval(self, pred_result_logits, data_list: list, input_ids, start_id=-1, end_id=-1):
        def _split_data_list(start_id,end_id):
            '''
            for mnre:
                parameters: data_list: {'words': [words], 'relations': [relations], 'heads': [heads], 'tails': [tails], 'imgids': [imgids],
                    'dataid': [dataid], 'aux_imgs': [aux_imgs], "rcnn_imgs": [rcnn_imgs]}
                    start_id: int
                    end_id: int
                
                return: data_list: {'token':[],
                                    'entity_pair_list':[[head,tail,r],[]],
                                    'entity_list:['pos':[[start,end]],'name':[[entity1],[entity2]]]}
            '''
            # using original data
            if isinstance(data_list, dict):
                token = data_list['words'][start_id:end_id]
                head = data_list['heads'][start_id:end_id]
                tail = data_list['tails'][start_id:end_id]
                relations = data_list['relations'][start_id:end_id]

                entity_pair_list = [[[0,1,self.rel2id[r]]] for _, _, r in zip(head, tail, relations)]
                entity_list = [[{'pos':[h['pos']], 'name':[h['name']]}
                                ,{'pos':[t['pos']], 'name':[t['name']]}] for h,t in zip(head, tail)]
                data = [{'token':tok, 'entity_pair_list': epl, 'entity_list':el ,'labels': la} for tok,epl,el,la in zip(token,entity_pair_list,entity_list,relations)]
                return data
            # using overlapping data
            else:
                data= data_list[start_id:end_id]
                return data

        if start_id!=-1 and end_id!=-1: # change data format
            data_list = _split_data_list(start_id, end_id)

        preds_one_hot = []
        golds_one_hot = []

        preds_one_hot_without_na = []
        golds_one_hot_without_na = []

        normal_id_one_hot = []
        overlap_id_one_hot = []
        multi_id_one_hot = []
        without_na_one_hot = []
        
        mention_count_id = [[] for _ in range(self.mention_count)]
        triple_count_id = [[] for _ in range(self.triple_count)]

        batch_pred_labels = []
        batch_true_labels = []
        batch_pred_logits = []

        batch_ep_pred_labels = []
        batch_ep_true_labels = []
        #self.matrix.extend(pred_result)
        #self.tsne_data.extend(data_list)
        batch_em = []
            
        pred_result = (pred_result_logits.detach().cpu().numpy())  # B*NL*SL*SL
    
        for i, d in enumerate(data_list):
            is_print = random.random() < self.print_error_prob
            epl_c = d["entity_pair_list"].copy()
            epl_c = [ep for ep in epl_c if ep[2] != self.na_id]

            is_normal_data = F1Metric.is_normal(epl_c)
            is_multi_label_data = F1Metric.is_multi_label(epl_c)
            is_over_lapping_data = F1Metric.is_over_lapping(epl_c)
            triple_count = len(epl_c)
            triple_count = min(triple_count, len(self.triple_count_res) - 1)

            #test mentions resutl
            mention_count = max([len(i['pos']) for i in d['entity_list']])
            mention_count = min(mention_count, len(self.mention_count_res) - 1)

            checked_epl_id = []
            epl = d["entity_pair_list"]
            el = d['entity_list']
            #sl = d["sdp_list"]
            #sdp_list = list(set([i for x in sl for i in x]))

            # store labels per instance
            batch_true_labels_per_ins = []
            batch_pred_labels_per_ins = []
            batch_pred_logits_per_ins = []
            # store entity mask
            batch_em_per_ins = []

            for e_idx, ep in enumerate(epl):
                if e_idx in checked_epl_id:
                    continue
                checked_epl_id.append(e_idx)
                pr = torch.from_numpy(pred_result[i])
                em = MREOverlapDataset.get_entity_mask(input_ids[i],self.entity_marker,ep).to(pr.device)  # SL*SL
                #em = MREDataset.get_entity_mask(input_ids[i],self.entity_marker)
                #em = MREOverlapDataset_test.get_entity_mask_ori(input_ids[i], self.entity_marker).to(pr.device)

                #em = MREOverlapDataset.get_entity_mask(input_ids[i],ep, el).to(pr.device)  # SL*SL
                batch_em_per_ins.append(em)
                gold_label = [ep[2]]
                # record this entity pair , if it has a relation not NA relation 

                for _e_idx, _ep in enumerate(epl):
                    if _e_idx in checked_epl_id:
                        continue
                    if _ep[0] == ep[0] and _ep[1] == ep[1]:
                        gold_label.append(_ep[2])
                        checked_epl_id.append(_e_idx)
    
                if len(pr.shape)>1:  # use matrix label with em
                    _res = ((pr * em).sum(dim=(1, 2)) / em.sum()).cpu().numpy()  # NL
                else:
                    _res = pr.cpu().numpy()
                
                # store every predicted logits value
                pred_logits = _res
                
                res = [0] * len(_res)
                #res = torch.zeros_like(_res) 
                if self.multi_label:
                    if self.na_id > -1 and _res[self.na_id] > 0.5:
                        res[self.na_id] = 1
                    else:
                        # res[self.na_id] = 0
                        res = (_res > 0.5).astype(int)  # NL
                else:
                    res = _res.copy()
                    res = (res == max(_res)).astype(int)
                

                gold = [0] * len(_res)
                gold = np.array(gold,np.float32)
                gold[gold_label] = 1

                preds_one_hot.append(res)
                golds_one_hot.append(gold)


                #for without_na
                res_na = res.copy()
                gold_na = gold.copy()
                
                if self.na_id>-1:
                    res_na[self.na_id] = 0 
                    gold_na[self.na_id] = 0

                preds_one_hot_without_na.append(res_na)
                golds_one_hot_without_na.append(gold_na)

                # for without na relation
                na_relation_id = [0]+[1]*(len(_res)-1)
                without_na_one_hot.append(na_relation_id) 
                # for normal data
                normal_id = [1]*len(_res) if is_normal_data else [0]*len(_res)
                normal_id_one_hot.append(normal_id) 
                # for overlapping data
                overlap_id = [1]*len(_res) if is_over_lapping_data else [0]*len(_res)
                overlap_id_one_hot.append(overlap_id) 
                # for multi data
                multi_id = [1]*len(_res) if is_multi_label_data else [0]*len(_res)
                multi_id_one_hot.append(multi_id) 
                # for triple count
                for count in range(len(triple_count_id)):
                    m_i = [1]*len(_res) if count==triple_count else [0]*len(_res)
                    triple_count_id[count].append(m_i)
                
                # for mention count
                for count in range(len(mention_count_id)):
                    m_i = [1]*len(_res) if count==mention_count else [0]*len(_res)
                    mention_count_id[count].append(m_i)


                gold_label_str = [g if self.id2rel is None else self.id2rel[g] for g in gold_label]

                # get predicted and gold labels correspoding ids, if predicted is multi label the gold label should get equal gold labels
                pred_labels = [idx for idx, s in enumerate(res) if s == 1]
                gold_labels = [idx for idx, s in enumerate(gold) if s == 1]
                

                if self.multi_label:
                    multi_true_label = ((res+gold)!=0).astype(int)
                    gold_labels = [idx if idx in gold_labels else self.na_id for idx, s in enumerate(multi_true_label) if s == 1]
                    pred_labels = [idx if idx in pred_labels else self.na_id for idx, s in enumerate(multi_true_label) if s == 1]

                assert len(pred_labels)==len(gold_labels)
                #assert len(ep_pred_label)==len(ep_gold_label)

                #pred_str = [idx if self.id2rel is None else self.id2rel[idx] for idx in pred_labels]
            
                #self.df = self.df.append({"Text": " ".join(d["token"]),
                #                        "Subject": d["entity_list"][ep[0]],
                #                        "Object": d["entity_list"][ep[1]],
                #                        "Gold Label": ",".join(gold_label_str),
                #                       "Predict": ",".join(pred_str),
                #                        }, ignore_index=True, )

                #preds = pred_result_logits.argmax(-1)
                
                batch_true_labels.extend(gold_labels)
                batch_pred_labels.extend(pred_labels)
                batch_pred_logits.append(pred_logits)

                #if predicted relation is not na means that this entity pair predicted right
                ep_gold_label = [1 if i>0 else 0 for i in gold_labels]
                ep_pred_label = [1 if i>0 else 0 for i in pred_labels] 

                batch_ep_true_labels.extend(ep_gold_label)
                batch_ep_pred_labels.extend(ep_pred_label)
        
                batch_true_labels_per_ins.append(gold_labels)
                batch_pred_labels_per_ins.append(pred_labels)
                batch_pred_logits_per_ins.append(pred_logits)
            batch_em.append(batch_em_per_ins)

            self.re_true_labels_per_ins.append(batch_true_labels_per_ins)
            self.re_pred_labels_per_ins.append(batch_pred_labels_per_ins)
            self.re_pred_logits_per_ins.append(batch_pred_logits_per_ins)
            
        self.re_pred_one_hot_labels.extend(golds_one_hot_without_na)
        self.re_true_one_hot_labels.extend(preds_one_hot_without_na)

        self.re_true_labels.extend(batch_true_labels)
        self.re_pred_labels.extend(batch_pred_labels)
        self.re_pred_logits.extend(batch_pred_logits)

        self.ep_true_labels.extend(batch_ep_true_labels)
        self.ep_pred_labels.extend(batch_ep_pred_labels)
        #pred_logits = torch.tensor(pred_logits)
        labels = torch.tensor(batch_true_labels)
        #self.re_pred_logits.extend(pred_logits)



        # count hints
        #re_pred_ranks = 1 + torch.argsort(torch.argsort(torch.tensor(batch_pred_logits), dim=1, descending=True), dim=1,
        #                                    descending=False)[torch.arange(labels.shape[0]), labels]
        #re_pred_ranks = re_pred_ranks.float()
        #for rel_id in range(len(self.rel2id)):
        #    ranks = re_pred_ranks[labels == rel_id]
        #    for k in range(len(self.rel2id)):
        #        self.hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + self.hits[rel_id, k + 1]
                
        #"""


        # for F1 score all
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot,golds_one_hot)
        #self.res[CORRECT] += tp_and_tn
        self.res[TRUE_POSITIVE] += tp
        self.res[FALSE_NEGATIVE] += fn
        self.res[FALSE_POSITIVE] += fp
        self.res[TRUE_NEGATIVE] += tn
        self.res[TOTAL] += total
        #"""
        #for without no relation F1 score
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na, without_na_one_hot)
        #self.without_na_res[CORRECT] += tp_and_tn
        self.without_na_res[TRUE_POSITIVE] += tp
        self.without_na_res[FALSE_NEGATIVE] += fn
        self.without_na_res[FALSE_POSITIVE] += fp
        self.without_na_res[TRUE_NEGATIVE] += tn
        self.without_na_res[TOTAL] += total

        #for normal relation F1 score
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,normal_id_one_hot)
        #self.normal_res[CORRECT] += tp_and_tn
        self.normal_res[TRUE_POSITIVE] += tp
        self.normal_res[FALSE_NEGATIVE] += fn
        self.normal_res[FALSE_POSITIVE] += fp
        self.normal_res[TRUE_NEGATIVE] += tn
        self.normal_res[TOTAL] += total
        #for overlap relation F1 score
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,overlap_id_one_hot)
        #self.over_lapping_res[CORRECT] += tp_and_tn
        self.over_lapping_res[TRUE_POSITIVE] += tp
        self.over_lapping_res[FALSE_NEGATIVE] += fn
        self.over_lapping_res[FALSE_POSITIVE] += fp
        self.over_lapping_res[TRUE_NEGATIVE] += tn
        self.over_lapping_res[TOTAL] += total

        #for multi relation F1 score
        tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,multi_id_one_hot)
        #self.multi_label_res[CORRECT] += tp_and_tn
        self.multi_label_res[TRUE_POSITIVE] += tp
        self.multi_label_res[FALSE_NEGATIVE] += fn
        self.multi_label_res[FALSE_POSITIVE] += fp
        self.multi_label_res[TRUE_NEGATIVE] += tn
        self.multi_label_res[TOTAL] += total

        #for triple count relation F1 score
        for i in range(self.triple_count):
            tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,triple_count_id[i])
            #self.triple_count_res[i][CORRECT] += tp_and_tn
            self.triple_count_res[i][TRUE_POSITIVE] += tp
            self.triple_count_res[i][FALSE_NEGATIVE] += fn
            self.triple_count_res[i][FALSE_POSITIVE] += fp
            self.triple_count_res[i][TRUE_NEGATIVE] += tn
            self.triple_count_res[i][TOTAL] += total

        #for mention count relation F1 score
        for i in range(self.mention_count):
            tp, fn, fp, tn, total=self.logical_measure(preds_one_hot_without_na,golds_one_hot_without_na,mention_count_id[i])    
            #self.mention_count_res[i][CORRECT] += tp_and_tn
            self.mention_count_res[i][TRUE_POSITIVE] += tp
            self.mention_count_res[i][FALSE_NEGATIVE] += fn
            self.mention_count_res[i][FALSE_POSITIVE] += fp
            self.mention_count_res[i][TRUE_NEGATIVE] += tn
            self.mention_count_res[i][TOTAL] += total
        #else: #no single-label
        #    pass

        #    preds = pred_result_logits.argmax(-1)
        #    labels = torch.tensor([self.rel2id[r] for r in data_list['labels']])
        #    self.re_true_labels.extend(labels.view(-1).detach().cpu().tolist())
        #    self.re_pred_labels.extend(preds.view(-1).detach().cpu().tolist())
        #    self.re_pred_logits.extend(pred_result_logits.detach().cpu().tolist())
        #    re_pred_ranks = 1 + torch.argsort(torch.argsort(pred_result_logits, dim=1, descending=True), dim=1,
        #                                        descending=False)[
        #        torch.arange(labels.shape[0]), labels]
        #    re_pred_ranks = re_pred_ranks.float()
        #   for rel_id in range(len(self.rel2id)):
        #        ranks = re_pred_ranks[labels == rel_id]
        #        for k in range(len(self.rel2id)):
        #            self.hits[rel_id, k + 1] = torch.numel(ranks[ranks <= (k + 1)]) + self.hits[rel_id, k + 1]

    def logical_measure(self,preds_one_hot,golds_one_hot,criteria=None): 
        preds_one_hot=np.array(preds_one_hot).astype(np.float32)
        golds_one_hot=np.array(golds_one_hot).astype(np.float32)

        tp_and_tn = ((preds_one_hot[:,:]==1) & (golds_one_hot[:,:]==1)).astype(np.float32).sum()
        if criteria==None:
            tp = ((preds_one_hot[:,:]==1) & (golds_one_hot[:,:]==1)).astype(np.float32).sum()
            fn = ((preds_one_hot[:,:]!=1) & (golds_one_hot[:,:]==1)).astype(np.float32).sum()
            fp = ((preds_one_hot[:,:]==1) & (golds_one_hot[:,:]!=1)).astype(np.float32).sum()
            #tn = ((preds_one_hot[:,:]!=1) & (golds_one_hot[:,:]!=1)).astype(np.float32).sum()   #将每个正确类别视为正类，其他所有类别视为负类(此处写的有误)
            #total = ((preds_one_hot[:,:]==1)|(golds_one_hot[:,:]==1)).astype(np.float32).sum()
        else:
            criteria=np.array(criteria).astype(np.float32)

            tp = ((preds_one_hot[:,:]==1) & (golds_one_hot[:,:]==1) & (criteria[:,:]==1)).astype(np.float32).sum()
            fn = ((preds_one_hot[:,:]!=1) & (golds_one_hot[:,:]==1) & (criteria[:,:]==1)).astype(np.float32).sum()
            fp = ((preds_one_hot[:,:]==1) & (golds_one_hot[:,:]!=1) & (criteria[:,:]==1)).astype(np.float32).sum()
            #total = (((preds_one_hot[:,:]==1)|(golds_one_hot[:,:]==1)) & (criteria[:,:]==1)).astype(np.float32).sum()
        total=preds_one_hot.shape[0]
        tn = total-(tp+fn+fp)

        return tp, fn, fp, tn, total

    @staticmethod
    def _get_result(res):    #in multi-classification task, accuracy = correct / total of all triples
        acc = float(res[TRUE_POSITIVE]+res[TRUE_NEGATIVE])/ float(res[TOTAL] + 1e-9)   #The Total_ins means all entity_pair
        micro_p = float(res[TRUE_POSITIVE]) / float(res[TRUE_POSITIVE]+res[FALSE_POSITIVE] + 1e-9)
        micro_r = float(res[TRUE_POSITIVE]) / float(res[TRUE_POSITIVE]+res[FALSE_NEGATIVE] + 1e-9)
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)
        return {'micro_f1': round(micro_f1,6), 'micro_p': round(micro_p,6), 'micro_r': round(micro_r,6), 'acc': round(acc,6)}   #round(value, 3)保留三位小数
    
    @staticmethod
    def _get_re_report(true_oh_label,pred_oh_label,logits,label):
        #true = self.re_true_one_hot_labels
        #pred = self.re_pred_one_hot_labels
        #labels = list(self.rel2id.values())
        #label_name = list(self.re_dict.keys())
        true = true_oh_label
        pred = pred_oh_label
        logits = logits
        labels = list(label.values())
        label_name = list(label.keys())

        report = classification_report(y_true=true, y_pred=pred,labels=labels,target_names=label_name, digits=6)
        macro_f1 = f1_score(y_true=true,y_pred=pred,labels=labels,average='macro')
        micro_f1 = f1_score(y_true=true,y_pred=pred,labels=labels,average='micro')
        micro_p,micro_r,fscore,_ = precision_recall_fscore_support(y_true=true,y_pred=pred,labels=labels,average="micro")
        acc = accuracy_score(y_true=true,y_pred=pred)
        #cm= confusion_matrix(y_true=true,y_pred=pred,labels=label_name) 
        topk = F1Metric.calculate_topk(y_true=true, y_pred=logits,labels=labels,target_names=label_name, topk=3,output_format='dict')
        return {'main_indicators':{'micro_f1': round(micro_f1,6), 'micro_p': round(micro_p,6), 'micro_r': round(micro_r,6), 'acc': round(acc,6)},
                #'confusion_matrix':cm,
                'macro_f1':macro_f1,
                'report':report,
                'topk':topk}
    @staticmethod
    def _get_ep_report(true_label, pred_label):
        true = true_label
        pred = pred_label
        micro_p,micro_r,fscore,_ = precision_recall_fscore_support(y_true=true,y_pred=pred,average="binary")
        acc = accuracy_score(y_true=true,y_pred=pred)
        return {'micro_f1': round(fscore,6), 'micro_p': round(micro_p,6), 'micro_r': round(micro_r,6), 'acc': round(acc,6)}
    
    def get_reports(self):
        reports = {} 
        reports['without_na_res'] = F1Metric._get_re_report(self.re_true_one_hot_labels,self.re_pred_one_hot_labels,self.re_pred_logits,self.rel2id)
        res = self.get_result()
        reports['normal'] = res['normal']
        reports["over_lapping"] = res['over_lapping']
        reports["multi_label"] = res["multi_label"]
        reports["triple_res"] = res["triple_res"]
        reports['entity_pair'] = F1Metric._get_ep_report(self.ep_true_labels, self.ep_pred_labels)
        return reports

    # results for overlapping
    def get_result(self):
        res = F1Metric._get_result(self.res)
        res["whl"] = F1Metric._get_result(self.test)
        res["without_na_res"] = F1Metric._get_result(self.without_na_res)
        res["na_res"] = F1Metric._get_result(self.na_res)
        res["without_na_micro_f1"] = res["without_na_res"]["micro_f1"]
        res["normal"] = F1Metric._get_result(self.normal_res)
        res["over_lapping"] = F1Metric._get_result(self.over_lapping_res)
        res["multi_label"] = F1Metric._get_result(self.multi_label_res)

        res["triple_res"] = {}
        for i in range(len(self.triple_count_res)):
            res["triple_res"][str(i)] = F1Metric._get_result(self.triple_count_res[i])

        res["mention_res"] = {}
        for i in range(len(self.mention_count_res)):
            res["mention_res"][str(i)] = F1Metric._get_result(self.mention_count_res[i])
        return res

    def get_predicted_labels(self):
        return self.re_true_labels, self.re_pred_labels, self.re_pred_logits 

    def get_predicted_labels_per_ins(self):
        return self.re_true_labels_per_ins, self.re_pred_labels_per_ins, self.re_pred_logits_per_ins

    def get_predicted_one_hot_labels(self):
        return self.re_true_one_hot_labels, self.re_pred_one_hot_labels 
    
    def reset(self):

        self.res = {CORRECT: 0, TOTAL: 0, TRUE_POSITIVE: 0, FALSE_POSITIVE: 0, FALSE_NEGATIVE:0, TRUE_NEGATIVE:0}

        self.without_na_res = self.res.copy()
        self.na_res = self.res.copy()
        self.normal_res = self.res.copy()
        self.over_lapping_res = self.res.copy()
        self.multi_label_res = self.res.copy()

        self.test = self.res.copy()

        self.triple_count_res = [self.res.copy() for _ in range(self.triple_count)]      # control the result will or not include how many sentence indictors
        self.mention_count_res = [self.res.copy() for _ in range(self.mention_count)]      # control the result will or not include how many mentions indictors
        self.df = pd.DataFrame(columns=["Text", "Subject", "Object", "Gold Label", "Predict"])
        
        #record predicted results
        self.re_true_labels=[] # [1,3,5,8,23,9......]
        self.re_pred_labels=[] # [1,3,6,7,20,8......]
        self.re_pred_logits=[] # [1,3,6,7,20,8......]
        
        self.ep_true_labels=[] #[1,0,0,1.....]
        self.ep_pred_labels=[] #[1,0,0,1.....]

        self.hits = torch.zeros([len(self.rel2id), len(self.rel2id) + 1])
        self.re_true_one_hot_labels=[] #[[1,0,0,0,...,1],[1,0,0,0,1,...,0],...]
        self.re_pred_one_hot_labels=[] #


        self.re_true_labels_per_ins=[]
        self.re_pred_labels_per_ins=[]
        self.re_pred_logits_per_ins=[]

    @staticmethod
    def is_normal(epl):
        entities = set()
        for e in epl:
            entities.add(e[0])
            entities.add(e[1])
        return len(entities) == (len(epl) * 2)

    @staticmethod
    def is_multi_label(epl):
        if F1Metric.is_normal(epl):
            return False
        entities_pair = []
        for i, e in enumerate(epl):
            entities_pair.append(tuple([e[0], e[1]]))
        return len(entities_pair) != len(set(entities_pair))

    @staticmethod
    def is_over_lapping(epl):
        if F1Metric.is_normal(epl):
            return False

        entities_pair = []
        for i, e in enumerate(epl):
            entities_pair.append(tuple([e[0], e[1]]))

        entities_pair = set(entities_pair)
        entities = []
        for pair in entities_pair:
            entities.extend(pair)
        entities = set(entities)
        return len(entities) != (2 * len(entities_pair))
    
    @staticmethod
    def calculate_topk(y_true, y_pred, labels, target_names, topk=5, output_format='dict', range_topk=False):
        from collections import defaultdict
        """
        Calculate top-k metrics for multi-label, multi-class classification.

        Parameters:
            y_true (list of list of int): Ground truth 0-1 binary indicators for each class for each sample.
            y_pred (list of list of float): Predicted probabilities for each class for each sample.
            labels (list of int): List of label indices.
            target_names (list of str): List of label names corresponding to label indices.
            topk (int): Number of top predictions to consider.
            output_format (str): 'dict' for dictionary output, 'string' for formatted string output.
            range_topk (bool): If True, calculate top-1 to top-k accuracies; if False, calculate only top-k.

        Returns:
            dict or str: Top-k analysis results in specified format.

        # Example usage
            y_true = [
                [0, 0, 1, 1],  # Sample 1 ground truth (binary indicators)
                [1, 0, 0, 1],  # Sample 2 ground truth
                [0, 1, 0, 0]   # Sample 3 ground truth
            ]
            y_pred = [
                [0.1, 0.2, 0.9, 0.8],  # Sample 1 predicted probabilities
                [0.9, 0.1, 0.3, 0.8],  # Sample 2 predicted probabilities
                [0.2, 0.6, 0.1, 0.4]   # Sample 3 predicted probabilities
            ]
            labels = [0, 1, 2, 3]
            target_names = ["Class A", "Class B", "Class C", "Class D"]
        """
        if len(labels) != len(target_names):
            raise ValueError("Length of labels must match length of target_names.")

        if not (len(y_true) == len(y_pred)):
            raise ValueError("y_true and y_pred must have the same length.")

        # Initialize counters
        # Initialize counters
        # Initialize counters
        total_samples = len(y_true)
        precision_recall_counts = defaultdict(lambda: [0, 0])  # [True Positives, Predicted Positives]
        classwise_counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [TP, PP] for each class
        recall_counts = defaultdict(int)  # Count of relevant labels per sample

        results = {"overall": {}, "per_class": {}}

        for true_labels, pred_probs in zip(y_true, y_pred):
            true_indices = set(i for i, val in enumerate(true_labels) if val == 1)
            recall_counts[len(true_indices)] += 1

            sorted_indices = np.argsort(pred_probs)[::-1]

            for k in range(1, topk + 1):
                topk_indices = set(sorted_indices[:k])

                true_positives = len(true_indices & topk_indices)
                predicted_positives = len(topk_indices)

                precision_recall_counts[k][0] += true_positives
                precision_recall_counts[k][1] += predicted_positives

                # Class-wise counts
                for idx in topk_indices:
                    if idx in true_indices:
                        classwise_counts[k][idx][0] += 1  # True Positive for this class
                    classwise_counts[k][idx][1] += 1  # Predicted Positive for this class

        # Compute overall metrics for top-1 to top-k
        for k in range(1, topk + 1):
            true_positives = precision_recall_counts[k][0]
            predicted_positives = precision_recall_counts[k][1]

            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = true_positives / total_samples if total_samples > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            results["overall"][f"Top-{k}"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        # Compute per-class metrics
        for k in range(1, topk + 1):
            results["per_class"][f"Top-{k}"] = {}
            for idx, name in zip(labels, target_names):
                true_positives = classwise_counts[k][idx][0]
                predicted_positives = classwise_counts[k][idx][1]

                precision = true_positives / predicted_positives if predicted_positives > 0 else 0
                recall = true_positives / total_samples if total_samples > 0 else 0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

                results["per_class"][f"Top-{k}"][name] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }

        # Format the output
        if output_format == 'string':
            output = []
            for k in range(1, topk + 1):
                overall = results['overall'][f'Top-{k}']
                output.append(f"Top-{k} Overall Metrics: Precision: {overall['precision']:.4f}, Recall: {overall['recall']:.4f}, F1: {overall['f1']:.4f}")
                for name, metrics in results['per_class'][f'Top-{k}'].items():
                    output.append(f"    {name}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            return "\n".join(output)

        return results
