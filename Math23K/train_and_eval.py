import copy
import logging
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
        word2index,
        tokenizer):
    set_seed(args.seed)
    device = args.device
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename='./logs/train.log', level=logging.DEBUG, format=LOG_FORMAT)

    model = pretrained_model.from_pretrained(args.bert_model_name, args).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * args.epochs)

    optimizer, scheduler = get_optimizers(args.lr, model, t_total)
    model.zero_grad()

    best_mtree_acc_epoch = 0
    best_mtree_acc = 0
    cur_mtree_acc = 0

    for epoch in range(args.epochs):
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        model.train()
        with tqdm(enumerate(train_dataloader), desc='training epoch {}'.format(epoch), total=len(train_dataloader),
                  ncols=110) as pbar:
            for step, (feature, max_height, max_width) in pbar:
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
            val_acc_mtree, val_acc_node, val_acc_format = evaluate(args, val_dataloader, model, word2index, device)
            logging.info(
                'epoch: {}, loss: {:.5f}, val_acc_mtree: {:.5f}, val_acc_node: {:.5f}, val_acc_format: {:.5f}'.format(
                    epoch, total_loss / (len(train_dataloader)), val_acc_mtree, val_acc_node, val_acc_format
                )
            )
            if val_acc_mtree > best_mtree_acc:
                best_mtree_acc = val_acc_mtree
                best_mtree_acc_epoch = epoch
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(f"check_points/{args.model_folder}")
                tokenizer.save_pretrained(f"check_points/{args.model_folder}")
                # torch.save(decoder.state_dict(),f"check_points/{args.model_folder}/decoder.pkl")
            else:
                if epoch - best_mtree_acc_epoch > args.early_stop:
                    logging.info('best epoch: {}, best mtree acc: {}'.format(best_mtree_acc_epoch, best_mtree_acc))
                    logging.info('best model saved in check_points/{}'.format(args.model_folder))
                    break

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
    torch.save(total_result, './save_files/valid_result_loss2_1_end_2atten.pt')
    return total_mtree_right / total_case, total_node_right / total_case, total_format_right / total_case

def remove_useless_node(mtree,pad_idx,end_idx):
    if mtree['data']==pad_idx or mtree['data']==end_idx:
        return None
    else:
        if len(mtree['children'])==0:
            return mtree
        else:
            new_children=[]
            for child in mtree['children']:
                new_child=remove_useless_node(child,pad_idx,end_idx)
                if new_child!=None:
                    new_children.append(new_child)
            mtree['children']=new_children
            return mtree

def mtree_to_code(mtree):
    data = mtree['data']
    format = mtree['format']
    code = [{'data': data, 'format': format}]
    if len(mtree['children']) == 0:
        return [code]
    else:
        return_code = []
        for child in mtree['children']:
            child_code = mtree_to_code(child)
            for c in child_code:
                return_code.append(code + c)
        return return_code

def IoU(code1,code2,pad_idx,end_idx):
    code1_str=[]
    code2_str=[]
    for i in code1:
        cur_code1_str=''
        for j in i:
            if j['data']==pad_idx:
                cur_code1_str+='_{}-0'.format(pad_idx)
            elif j['data']==end_idx:
                cur_code1_str+='_{}-0'.format(end_idx)
            else:
                cur_code1_str+='_{}-{}'.format(j['data'],j['format'])
        code1_str.append(cur_code1_str)
    for i in code2:
        cur_code2_str=''
        for j in i:
            if j['data']==pad_idx:
                cur_code2_str+='_{}-0'.format(pad_idx)
            elif j['data']==end_idx:
                cur_code2_str+='_{}-0'.format(end_idx)
            else:
                cur_code2_str+='_{}-{}'.format(j['data'],j['format'])
        code2_str.append(cur_code2_str)
    intersection=[]
    for i in code1_str:
        if i in code2_str:
            intersection.append(i)
            code2_str.remove(i)
    union=code1_str+code2_str
    return len(intersection)/len(union)

def test(
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
    total_iou=0
    total_case=0
    total_result=[]
    result_for_evaluate=[]

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
                node_right,format_right,mtree_right,result_batch,predict_mtrees, target_mtrees =model.evalue(
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
                    cur_index2value.extend(inst[total_case+mtree_idx]['num_list'])
                    result_batch[mtree_idx]['value_flag']=False
                    try:
                        pred_ans=cal_mtree(mtree_index_to_token(predict_mtrees[mtree_idx],cur_index2value,format_index2word))
                        #print(pred_ans)
                        if abs(pred_ans-inst[total_case+mtree_idx]['answer'])<1e-5:
                            total_value_right+=1
                            result_batch[mtree_idx]['value_flag']=True
                    except:
                        pass
                    inst_for_eval={}
                    inst_for_eval['question']=inst[total_case+mtree_idx]['original_text']
                    inst_for_eval['mapped_question']=inst[total_case+mtree_idx]['text']
                    inst_for_eval['answer']=inst[total_case+mtree_idx]['answer']
                    inst_for_eval['equation']=inst[total_case+mtree_idx]['equation']
                    inst_for_eval['mapped_equation']=inst[total_case+mtree_idx]['target_template']
                    
                    pad_idx,end_idx=word2index[args.pad_token],word2index[args.end_token]
                    new_pred_mtree=remove_useless_node(predict_mtrees[mtree_idx],pad_idx,end_idx)
                    new_pred_mtree_str=mtree_index_to_token(new_pred_mtree,cur_index2value,format_index2word)
                    new_target_mtree=remove_useless_node(target_mtrees[mtree_idx],pad_idx,end_idx)
                    new_target_mtree_str=mtree_index_to_token(new_target_mtree,cur_index2value,format_index2word)
                    pred_code=mtree_to_code(new_pred_mtree)
                    target_code=mtree_to_code(new_target_mtree)
                    iou=IoU(pred_code,target_code,pad_idx,end_idx)
                    total_iou+=iou
                    result_for_evaluate.append({
                            'inst':inst_for_eval,
                            'pred_mtree':new_pred_mtree_str,
                            'target_mtree':new_target_mtree_str,
                            'pred_ans':pred_ans,
                            'flags':result_batch[mtree_idx],
                            'iou':iou
                        })
                total_mtree_right+=mtree_right
                total_node_right+=node_right
                total_format_right+=format_right
                total_result+=result_batch
                total_case+=batch_size
                torch.save(result_for_evaluate,'./results/result_for_evaluate.pt')
                pbar.set_postfix({'mtree_acc':total_mtree_right/(total_case),'node_acc':total_node_right/(total_case),'format_acc':total_format_right/(total_case)})
    #torch.save(total_result,'./save_files/valid_result.pt')
    total_case=1000
    return total_mtree_right/total_case,total_node_right/total_case,total_format_right/total_case,total_value_right/total_case,total_iou/total_case




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

# def mtree_index_to_token1(mtree,index2word,format_index2word,num_start):
#     re_mtree={}
#     re_mtree['data']=index2word[mtree['data']]
#     re_mtree['format']=format_index2word[mtree['format']]
#     if mtree['data']>=num_start and mtree['data']<num_start+len(format_index2word):
#         re_mtree['constant']=True
#     else:
#         re_mtree['constant']=False
#     re_mtree['children']=[]
#     for child in mtree['children']:
#         re_mtree['children'].append(mtree_index_to_token1(child,index2word,format_index2word,num_start))
#     return re_mtree

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
