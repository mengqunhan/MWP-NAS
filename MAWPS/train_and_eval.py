import copy
import logging
import math

import torch.nn as nn
from typing import Tuple
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import os
import datetime
from tqdm import tqdm
from utils.tools import set_seed

def train(
        args,
        pretrained_model,
        train_dataloader,
        val_dataloader,
        index2word,
        word2index,
        tokenizer,
        fold):
    set_seed(args.seed)
    device = args.device
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='./save_files/MTree-Solver-5fold.log', level=logging.DEBUG, format=LOG_FORMAT)
    logging.info(f"*********************fold: {fold}*********************")
    # encoder=pretrained_model.from_pretrained(args.bert_model_name,args,return_dict=True).to(device)
    # decoder=Decoder(args).to(device)
    if args.bert_model_name=='bert-base-uncased':
        model = pretrained_model.from_pretrained('./pretrained_model/bert-base-uncased', args).to(device)
    else:
        model = pretrained_model.from_pretrained(args.bert_model_name, args).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * args.epochs)
    # encoder_optimizer, encoder_scheduler = get_optimizers(args.encoder_lr,encoder,t_total)
    # decoder_optimizer, decoder_scheduler = get_optimizers(args.decoder_lr,decoder,t_total)

    optimizer, scheduler = get_optimizers(args.lr, model, t_total)
    model.zero_grad()

    # train
    # encoder.train()
    # decoder.train()

    best_mtree_acc_epoch = 0
    best_mtree_acc = 0
    best_value_acc = 0
    save_path = f"check_points/{args.model_folder}-{fold}"
    os.makedirs(save_path, exist_ok=True)

    # encoder.zero_grad()
    # decoder.zero_grad()
    for epoch in range(args.epochs):
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        model.train()
        with tqdm(enumerate(train_dataloader), desc='training epoch {}'.format(epoch), total=len(train_dataloader),
                  ncols=110) as pbar:
            for step, (feature, max_height, max_width) in pbar:
                # encoder_optimizer.zero_grad()
                # decoder_optimizer.zero_grad()
                optimizer.zero_grad()
                node_data, node_format = get_mtree_node_data(feature.mtree)
                node_data = node_data.to(device)
                node_format = node_format.to(device)

                loss, loss1, loss2 = model(
                    input_ids=feature.input_ids.to(device),
                    attention_mask=feature.attention_mask.to(device),
                    token_type_ids=feature.token_type_ids.to(device),
                    variable_indexs_start=feature.variable_indexs_start.to(device),
                    variable_indexs_end=feature.variable_indexs_end.to(device),
                    num_variables=feature.num_variables.to(device),
                    variable_index_mask=feature.variable_index_mask.to(device),
                    return_dict=True,
                    max_height=max_height,
                    max_width=max_width,
                    mtree_node_data=node_data,
                    mtree_node_format=node_format,
                    word2index=word2index,
                    criterion=criterion
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                pbar.set_postfix({'loss': total_loss / (step + 1)})
        if val_dataloader is not None:
            val_acc_mtree, val_acc_node, val_acc_format, val_acc_value = evaluate_value(args, val_dataloader, model, index2word, word2index, device)
            logging.info('epoch: {}, loss: {:.5f}, val_acc_mtree: {:.5f}, val_acc_node: {:.5f}, val_acc_format: {:.5f}, val_acc_value:{:.5f}'.format(
                epoch, total_loss / (len(train_dataloader)), val_acc_mtree, val_acc_node, val_acc_format, val_acc_value
            )
            )
            if val_acc_mtree > best_mtree_acc:
                best_mtree_acc = val_acc_mtree
                best_value_acc = val_acc_value
                best_mtree_acc_epoch = epoch
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                # torch.save(decoder.state_dict(),f"check_points/{args.model_folder}/decoder.pkl")
            else:
                if epoch - best_mtree_acc_epoch > args.early_stop:
                    logging.info('best epoch: {}, best mtree acc: {}, best value acc:{}'.format(best_mtree_acc_epoch, best_mtree_acc, best_value_acc))
                    logging.info('best model saved in check_points/{}'.format(save_path))
                    break
    return best_value_acc


def evaluate(
        args,
        valid_dataloader,
        model,
        word2index,
        device):
    total_mtree_right = 0
    total_node_right = 0
    total_format_right = 0
    total_case = 0
    total_result = []
    # encoder.eval()
    # decoder.eval()
    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(valid_dataloader), desc='valid', total=len(valid_dataloader), ncols=110) as pbar:
            for step, (feature, max_height, max_width) in pbar:
                batch_size = feature.input_ids.shape[0]

                add_index = word2index['+']
                root_node_data = (torch.ones((batch_size, 1), dtype=torch.long).to(device)) * add_index
                root_node_format = torch.zeros_like(root_node_data)
                node_right, format_right, mtree_right, result_batch = model.eval_forward(
                    input_ids=feature.input_ids.to(device),
                    attention_mask=feature.attention_mask.to(device),
                    token_type_ids=feature.token_type_ids.to(device),
                    variable_indexs_start=feature.variable_indexs_start.to(device),
                    variable_indexs_end=feature.variable_indexs_end.to(device),
                    num_variables=feature.num_variables.to(device),
                    variable_index_mask=feature.variable_index_mask.to(device),
                    return_dict=True,
                    root_node_data=root_node_data,
                    root_node_format=root_node_format,
                    max_height=max_height,
                    max_width=max_width,
                    word2index=word2index,
                    target_mtree=feature.mtree
                )
                total_mtree_right += mtree_right
                total_node_right += node_right
                total_format_right += format_right
                total_result += result_batch
                total_case += batch_size
                pbar.set_postfix(
                    {'mtree_acc': total_mtree_right / (total_case), 'node_acc': total_node_right / (total_case),
                     'format_acc': total_format_right / (total_case)})
    torch.save(total_result, './save_files/valid_result_robertloss1.pt')
    return total_mtree_right / total_case, total_node_right / total_case, total_format_right / total_case


def evaluate_value(
    args,
    valid_dataloader,
    model,
    index2word,
    word2index,
    device):

    total_mtree_right=0
    total_node_right=0
    total_format_right=0
    total_value_right=0
    total_case=0
    total_result=[]

    # for ans evaluate
    inst=valid_dataloader.dataset.insts
    format_index2word=['N','-N','N/','-N/']

    index2value=copy.deepcopy(index2word[:word2index['temp_a']])
    index2value[word2index['PI']]= 3.14 #math.pi
    for i in range(args.num_start,word2index['temp_a'],1):
        index2value[i]=float(index2value[i])

    model.eval()
    with torch.no_grad():
        with tqdm(enumerate(valid_dataloader),desc='valid',total=len(valid_dataloader),ncols=110) as pbar:
            for step,(feature,max_height,max_width) in pbar:
                batch_size=feature.input_ids.shape[0]
                add_index=word2index['+']
                root_node_data=(torch.ones((batch_size,1),dtype=torch.long).to(device))*add_index
                root_node_format=torch.zeros_like(root_node_data)
                node_right,format_right,mtree_right,result_batch,predict_mtrees =model.evalue(
                    input_ids=feature.input_ids.to(device),
                    attention_mask=feature.attention_mask.to(device),
                    token_type_ids=feature.token_type_ids.to(device),
                    variable_indexs_start=feature.variable_indexs_start.to(device),
                    variable_indexs_end=feature.variable_indexs_end.to(device),
                    num_variables=feature.num_variables.to(device),
                    variable_index_mask=feature.variable_index_mask.to(device),
                    return_dict=True,
                    root_node_data=root_node_data,
                    root_node_format=root_node_format,
                    max_height=max_height,
                    max_width=max_width,
                    word2index=word2index,
                    target_mtree=feature.mtree
                )
                for mtree_idx in range(len(predict_mtrees)):
                    cur_index2value=[]
                    cur_index2value.extend(index2value)
                    num_list=[]
                    for n in inst[total_case+mtree_idx]['num_list']:
                        num_list.append(float(n))
                    cur_index2value.extend(num_list)
                    result_batch[mtree_idx]['value_flag']=False
                    result_batch[mtree_idx]['id']=inst[total_case+mtree_idx]['id']
                    result_batch[mtree_idx]['predict_tree']=predict_mtrees[mtree_idx]
                    try:
                        pred_ans=cal_mtree(mtree_index_to_token(predict_mtrees[mtree_idx],cur_index2value,format_index2word))
                        #print(pred_ans)
                        if abs(pred_ans-float(inst[total_case+mtree_idx]['ans']))<1e-5:
                            total_value_right+=1
                            result_batch[mtree_idx]['value_flag']=True
                    except:
                        pass
                total_mtree_right+=mtree_right
                #if total_mtree_right != total_value_right:
                #    print("biubi")
                total_node_right+=node_right
                total_format_right+=format_right
                total_result+=result_batch
                total_case+=batch_size
                pbar.set_postfix({'mtree_acc':total_mtree_right/(total_case),'node_acc':total_node_right/(total_case),'format_acc':total_format_right/(total_case)})
    torch.save(total_result,'./save_files/test_result2.pt')
    return total_mtree_right/total_case,total_node_right/total_case,total_format_right/total_case,total_value_right/total_case


def get_optimizers(learning_rate, model, num_training_steps, weight_decay: float = 0.01,
                   warmup_step: int = -1, eps: float = 1e-8) -> Tuple[
    torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)  # , correct_bias=False)
    # optimizer = AdamW(optimizer_grouped_parameters, eps=eps)  # , correct_bias=False)
    print(f"optimizer: {optimizer}")
    warmup_step = warmup_step if warmup_step >= 0 else int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def get_mtree_node_data(mtree):
    if mtree is None:
        return []
    mtree_node_data = []
    mtree_node_format = []
    get_mtree_node_data_layers(mtree, mtree_node_data, mtree_node_format)
    final_node_data = []
    final_node_format = []
    for i in range(len(mtree_node_data)):
        final_node_data.extend(mtree_node_data[i])
        final_node_format.extend(mtree_node_format[i])
    final_node_data_tensor = torch.stack(final_node_data, dim=0).transpose(0, 1)
    final_node_format_tensor = torch.stack(final_node_format, dim=0).transpose(0, 1)
    return final_node_data_tensor, final_node_format_tensor


def get_mtree_node_data_layers(mtree, data, format, height=1):
    cur_data = [mtree['data']]
    cur_format = [mtree['format']]
    if len(data) < height:
        data.append([])
        format.append([])
        data[height - 1].extend(cur_data)
        format[height - 1].extend(cur_format)
    else:
        data[height - 1].extend(cur_data)
        format[height - 1].extend(cur_format)
    if len(mtree['children']) == 0:
        return
    else:
        for child in mtree['children']:
            get_mtree_node_data_layers(child, data, format, height + 1)

def mtree_index_to_token(mtree,index2word,format_index2word):
    re_mtree={}
    re_mtree['data']=index2word[mtree['data']]
    re_mtree['format']=format_index2word[mtree['format']]
    re_mtree['children']=[]
    for child in mtree['children']:
        re_mtree['children'].append(mtree_index_to_token(child,index2word,format_index2word))
    return re_mtree

def cal_mtree(mtree:dict):
    # result=-1   # -1 can not be anyone of the right answer
    data=mtree['data']
    format=mtree['format']
    if data in ['+','*','*-','+/']:
        if data=='+':
            result=0
            for child in mtree['children']:
                cur_result=cal_mtree(child)
                if not isinstance(cur_result,str):
                    result+=cur_result
            return result
        elif data=='*':
            result=1
            for child in mtree['children']:
                cur_result=cal_mtree(child)
                if not isinstance(cur_result,str):
                    result*=cur_result
            return result
        elif data=='*-':
            result=-1
            for child in mtree['children']:
                cur_result=cal_mtree(child)
                if not isinstance(cur_result,str):
                    result*=cur_result
            return result
        elif data=='+/':
            temp_result=0
            for child in mtree['children']:
                cur_result=cal_mtree(child)
                if not isinstance(cur_result,str):
                    temp_result+=cur_result
            assert temp_result!=0, "when evaluate +/, temp_result can not be 0"
            return 1/temp_result
    else:
        if not isinstance(data,str):
            if format=='N':
                return data
            elif format=='-N':
                return -1*data
            elif format=='N/':
                assert data!=0, "when evaluate N/, data can not be 0"
                return 1/data
            else:
                assert data!=0, "when evaluate -N/, data can not be 0"
                return -1/data
        else:
            return 'None'

