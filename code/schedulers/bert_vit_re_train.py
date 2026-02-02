import torch
import time
from torch import optim
from tqdm import tqdm
from sklearn.metrics import classification_report as re_cls_report
from transformers.optimization import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np

from copy import deepcopy
from thop import profile, clever_format


from utilities.metrics import eval_result, F1Metric


class BertVitReTrainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None, re_dict=None,
                 model=None, process=None,
                 args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.re_dict = re_dict

        self.model = model
        self.process = process
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.best_test_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.final_test_metrics = {'acc': 0.0, 'micro_f1': 0.0, 'micro_r': 0.0, 'micro_p': 0.0}
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.metric = F1Metric(rel2id=re_dict,entity_marker=process.entity_marker)
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args
        self.pbar = None
        self.re_optimizer = None
        self.re_scheduler = None
        self.before_train()
    
    


    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info(f"\n***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            print("this file not exists: {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        if self.args.do_test:
            self.logger.info(f"\n***** Start testing without training *****")
            self.test(0)
            return

        # Display FLOPs and Parameters
        #dummy_input = next(iter(self.train_data))
        #if torch.cuda.is_available():
        #    dummy_input = [x.to(self.args.device) if isinstance(x, torch.Tensor) else x for x in dummy_input]  # Move input data to GPU
        #flops, params = profile(self.model, inputs=(dummy_input[0], dummy_input[1], dummy_input[-1]))  # Pass the correct inputs
        #flops, params = clever_format([flops, params], "%.3f")
        #self.logger.info(f"FLOPs: {flops}, Params: {params}")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            re_avg_loss = 0.0

            for epoch in range(1, self.args.num_epochs + 1):
                import time
                epoch_start = time.time()
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (re_loss, re_logits), labels, _ = self._step(re_batch,
                                                                 mode="train",
                                                                 task='re',
                                                                 epoch=epoch)

                    re_avg_loss += re_loss.detach().cpu().item()
                    re_loss.backward()
                    self.re_optimizer.step()
                    self.re_optimizer.zero_grad()
                    self.re_scheduler.step()

                    if self.step % self.refresh_step == 0:
                        re_avg_loss = float(re_avg_loss) / self.refresh_step
                        print_output = "RE loss:{:<6.5f}".format(re_avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        re_avg_loss = 0

                epoch_time = time.time() - epoch_start
                self.logger.info(f"Epoch {epoch} finished, time: {epoch_time:.2f} seconds.")

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)
                    self.test(epoch)
                self.logger.info("Epoch {} and corresponding loss {:<6.5f}".format(epoch, re_avg_loss))

            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, "
                             "best dev f1 is {}".format(self.best_dev_epoch,
                                                        self.best_dev_metrics['micro_f1'],
                                                        ))
            self.logger.info(
                "Get best test performance at epoch {}, "
                "best test f1 is {}".format(self.best_test_epoch,
                                            self.best_test_metrics['micro_f1'],
                                            ))
            self.logger.info(
                "Get final test performance according to validation results at epoch {}, "
                "final f1 {}, "
                "recall {}, "
                "precision {}, "
                "acc {}".format(
                    self.best_dev_epoch,
                    self.final_test_metrics['micro_f1'],
                    self.final_test_metrics['micro_r'],
                    self.final_test_metrics['micro_p'],
                    self.final_test_metrics['acc']))
            self.logger.info(
                "Get best test performance at epoch {}, "
                "best test f1 {}, "
                "recall {}, "
                "precision {}, "
                "acc {}".format(
                    self.best_test_epoch,
                    self.best_test_metrics['micro_f1'],
                    self.best_test_metrics['micro_r'],
                    self.best_test_metrics['micro_p'],
                    self.best_test_metrics['acc']))



    def evaluate(self, epoch=0):
        self.model.eval()
        self.logger.info(f"\n***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)



        self.metric.reset()
        data_idx = 0
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                step = 0
                total_loss = 0
                #hits = torch.zeros([len(self.re_dict), len(self.re_dict) + 1], device=self.args.device)
                for batch in self.dev_data:
                    step += 1
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels, _= self._step(re_batch,
                                                           mode="dev",
                                                           task='re',
                                                           epoch=epoch,)  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    #construct a metric function
                    batch_start, batch_end=data_idx, data_idx + len(logits)
                    input_ids = batch[0]
                    data_idx += len(logits)
                    self.metric.eval(logits, self.dev_data.dataset.data_dict, input_ids, batch_start, batch_end)
                    pbar.update()
                # evaluate done
                pbar.close()
                
                #re_true_labels,re_pred_labels, _ = self.metric.get_predicted_labels()
                #results_ori = eval_result(re_true_labels, re_pred_labels, self.re_dict, self.logger)
                
                results = self.metric.get_reports()
                micro_result = results['without_na_res']['main_indicators']
                cl_reports = results['without_na_res']['report']
                self.logger.info("%s\n", cl_reports)

                #self.logger.info('Evaluation results_ori: {}.'.format(results_ori))
                self.logger.info('Evaluation results: {}.'.format(micro_result))
                self.logger.info('Normal results: {}, SEP results: {}, EPO results: {}'.format(results['normal'],
                                                                                         results['over_lapping'],
                                                                                         results['multi_label']))
                self.logger.info('Triple results: {}'.format(results["triple_res"]))
                self.logger.info("Entity pair results: {}".format(results['entity_pair']))
                self.logger.info(
                    "Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}." \
                        .format(epoch, self.args.num_epochs, self.best_dev_metrics['micro_f1'],
                                self.best_dev_epoch,
                                micro_result['micro_f1'], ))
                if micro_result['micro_f1'] >= self.best_dev_metrics['micro_f1']:  # this epoch get best performance
                    self.logger.info("Get better dev performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metrics['micro_f1'] = micro_result['micro_f1']  # update best metric
                    self.best_dev_metrics['micro_r'] = micro_result['micro_r']
                    self.best_dev_metrics['micro_p'] = micro_result['micro_p']
                    self.best_dev_metrics['acc'] = micro_result['acc']
                    if self.args.save_path is not None:  # save model
                        torch.save(self.model.state_dict(), self.args.save_path)
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self, epoch=0):
        start_time = time.time()
        self.model.eval()
        self.logger.info(f"\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        profile_model(self.model, self.test_data, self.args, self.logger)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")
        re_true_labels, re_pred_labels, sample_word_lists, sample_image_ids = [], [], [], []
        re_pred_logits = []

        self.metric.reset()
        data_idx = 0
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    re_batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    if self.args.write_path is not None and self.args.do_test:
                        (loss, logits), labels, extend_word_lists, imgids = self._step(re_batch,
                                                                                       mode="test",
                                                                                       task='re',
                                                                                       epoch=epoch,)
                    else:
                        (loss, logits), labels, _ = self._step(re_batch,
                                                               mode="test",
                                                               task='re',
                                                               epoch=epoch,)
                    total_loss += loss.detach().cpu().item()

                    batch_start, batch_end = data_idx, data_idx + len(logits)
                    input_ids = batch[0]
                    data_idx += len(logits)
                    self.metric.eval(logits, self.test_data.dataset.data_dict, input_ids, batch_start, batch_end)

                    if self.args.write_path is not None and self.args.do_test:
                        sample_word_lists.extend([*extend_word_lists])
                        sample_image_ids.extend([*imgids])
                    pbar.update()
                pbar.close()

                if self.args.write_path is not None and self.args.do_test:
                    #re_true_labels, re_pred_labels, re_pred_logits = self.metric.get_predicted_labels()
                    re_true_labels, re_pred_labels, re_pred_logits = self.metric.get_predicted_labels_per_ins()
                    write_file_dict = {'sample_word_lists': sample_word_lists, 'sample_image_ids': sample_image_ids,
                                       'true_labels': re_true_labels, 'pred_labels': re_pred_labels,
                                       'pred_logits': re_pred_logits}
                    df = pd.DataFrame(write_file_dict)
                    # saving the dataframe
                    df.to_csv(self.args.write_path + '_' + 'test.csv')

                results = self.metric.get_reports()
                micro_result = results['without_na_res']['main_indicators']
                cl_reports = results['without_na_res']['report']
                self.logger.info("%s\n", cl_reports)
                self.logger.info('Evaluation results: {}.'.format(micro_result))
                self.logger.info('Normal results: {}, SEP results: {}, EPO results: {}'.format(
                    results['normal'], results['over_lapping'], results['multi_label']))
                self.logger.info('Triple results: {}'.format(results["triple_res"]))
                self.logger.info("Entity pair results: {}".format(results['entity_pair']))
                self.logger.info(
                    "Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, "
                    .format(epoch, self.args.num_epochs,
                            self.best_test_metrics['micro_f1'],
                            self.best_test_epoch,
                            micro_result['micro_f1'], ))

                if epoch == self.best_dev_epoch:
                    if micro_result['micro_f1'] > self.final_test_metrics['micro_f1']:
                        self.final_test_metrics['micro_f1'] = micro_result['micro_f1']
                        self.final_test_metrics['micro_r'] = micro_result['micro_r']
                        self.final_test_metrics['micro_p'] = micro_result['micro_p']
                        self.final_test_metrics['acc'] = micro_result['acc']

                if micro_result['micro_f1'] >= self.best_test_metrics['micro_f1']:
                    self.logger.info("Get better test performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_metrics['micro_f1'] = micro_result['micro_f1']
                    self.best_test_metrics['micro_r'] = micro_result['micro_r']
                    self.best_test_metrics['micro_p'] = micro_result['micro_p']
                    self.best_test_metrics['acc'] = micro_result['acc']
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path+'_test')
                        self.logger.info("Save best <<test>> model at {}".format(self.args.save_path+'_test'))

        elapsed = time.time() - start_time
        self.logger.info(f"Test finished, total time: {elapsed:.2f} seconds.")
        self.model.train()

    def _step(self, batch, mode="train", task='re', epoch=0):
        if self.args.write_path is not None and mode == 'test' and self.args.do_test:
            input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs, extend_word_lists, imgids, matrix_labels = batch
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 labels=labels,
                                 images=images,
                                 aux_imgs=aux_imgs,
                                 rcnn_imgs=rcnn_imgs,
                                 matrix_labels=matrix_labels,
                                 task=task,
                                 epoch=epoch,)
            return outputs, labels, extend_word_lists, imgids
        else:
            re_input_ids, re_token_type_ids, re_attention_mask, re_labels, images, aux_imgs, rcnn_imgs, matrix_labels = batch
            if task == 're':
                input_ids = re_input_ids
                token_type_ids = re_token_type_ids
                attention_mask = re_attention_mask
                labels = re_labels
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 labels=labels,
                                 images=images,
                                 aux_imgs=aux_imgs,
                                 rcnn_imgs=rcnn_imgs,
                                 matrix_labels=matrix_labels,
                                 task=task,
                                 epoch=epoch,)
            return outputs, labels, attention_mask

    def before_train_ori(self):
        optimizer_grouped_parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        for name, param in self.model.named_parameters():
            if 'model' in name or name.startswith('re_classifier'):
                params['params'].append(param)
        optimizer_grouped_parameters.append(params)
        self.re_optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.re_scheduler = get_linear_schedule_with_warmup(optimizer=self.re_optimizer,
                                                            num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                            num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)

    
    def before_train(self):
        optimizer_grouped_parameters = []

        # 参数组 1: 基础模型参数，较低学习率
        base_params = {'lr': self.args.lr, 'weight_decay': 1e-2, 'params': []}
        
        # 参数组 2: 分类器参数，较高学习率
        classifier_params = {'lr': self.args.classifier_lr, 'weight_decay': 1e-2, 'params': []}

        # 遍历模型参数，根据名字分组
        for name, param in self.model.named_parameters():
            if 'model' in name:  # 基础模型参数
                base_params['params'].append(param)
            elif name.startswith('re_classifier'):  # 分类器参数
                classifier_params['params'].append(param)

        # 添加参数组
        optimizer_grouped_parameters.append(base_params)
        optimizer_grouped_parameters.append(classifier_params)

        # 定义优化器
        self.re_optimizer = optim.AdamW(optimizer_grouped_parameters)

        # 调度器
        self.re_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.re_optimizer,
            num_warmup_steps=int(self.args.warmup_ratio * self.train_num_steps),
            num_training_steps=self.train_num_steps
        )

        # 将模型移动到设备
        self.model.to(self.args.device)



def strip_thop_buffers(model):
    """清除 THOP 注入到模块里的 total_ops/total_params 缓存，避免 load_state_dict 报错。"""
    for m in model.modules():
        for k in ("total_ops", "total_params"):
            # THOP 用 register_buffer 注册，优先从 _buffers 移除
            if isinstance(getattr(m, "_buffers", None), dict) and k in m._buffers:
                m._buffers.pop(k, None)
            # 保险起见再删属性
            if hasattr(m, k):
                try:
                    delattr(m, k)
                except Exception:
                    pass



def profile_model(model, dataloader, args, logger):
    """在 deepcopy(model) 上做 FLOPs/Params 统计，不污染原模型。"""
    try:
        sample = next(iter(dataloader))
        if args.write_path is not None and args.do_test:
            input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs, extend_word_lists, imgids, matrix_labels = sample
        else:
            input_ids, token_type_ids, attention_mask, labels, images, aux_imgs, rcnn_imgs, matrix_labels = sample

        def _to_dev(x):
            return x.to(args.device) if isinstance(x, torch.Tensor) else x

        inputs_tuple = (
            _to_dev(input_ids),
            _to_dev(attention_mask),
            _to_dev(token_type_ids),
            _to_dev(labels),
            _to_dev(images),
            _to_dev(aux_imgs),
            _to_dev(rcnn_imgs),
            _to_dev(matrix_labels),
        )

        model_copy = deepcopy(model).to(args.device).eval()
        with torch.no_grad():
            flops, params = profile(model_copy, inputs=inputs_tuple, verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        logger.info(f"[Profile] FLOPs: {flops}, Params: {params}")
    except Exception as e:
        try:
            n_params = sum(p.numel() for p in model.parameters())
            logger.info(f"[Profile] FLOPs: (failed: {e}), Params: {clever_format([n_params], '%.3f')[0]}")
        except Exception as e2:
            logger.info(f"[Profile] failed: {e}; fallback failed: {e2}")
    
