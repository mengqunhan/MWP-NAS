import json
import sys
from tools import prefix_to_postfix, prefix_to_infix
from build_mtree import exp_to_mtree
from tqdm import tqdm
import torch
def read_data(file:str):
	with open(file, "r", encoding='utf-8') as read_file:
		data = json.load(read_file)
	return data

def mtree_to_code(mtree):
    data=mtree.data
    code = [data]
    if len(mtree.children)==0:
        return [code]
    else:
        return_code=[]
        for child in mtree.children:
            child_code=mtree_to_code(child)
            for c in child_code:
                return_code.append(code+c)
        return return_code

def code_to_str(codes):
    codes_str=[]
    for code in codes:
        code_str=''
        for token in code:
            code_str+=('_'+str(token.rstrip('@')))
        codes_str.append(code_str)
    return codes_str

def mtree_and_iou(mtree_code1,mtree_code2):
    # mtree acc
    flag=True
    mtree1,mtree2=mtree_code1,list(mtree_code2)
    for i in range(len(mtree_code1)):
        if mtree_code1[i] in mtree_code2:
            mtree_code2.remove(mtree_code1[i])
        else:
            flag=False
            break
    if len(mtree_code2)!=0:
        flag=False

    #iou
    intersection=[]
    for code in mtree1:
        if code in mtree2:
            intersection.append(code)
            mtree2.remove(code)
    union=mtree1+mtree2
    return flag,len(intersection)/len(union)

def exp_equal(exp1,exp2):
    if len(exp1)!=len(exp2):
        return False
    for token1,token2 in zip(exp1,exp2):
        if token1!=token2:
            return False
    return True

def mtree_to_dict(root):
        mtree_dict={}
        if len(root.children)==0:
            mtree_dict['data']=root.data
            mtree_dict['children']=[]
            return mtree_dict
        else:
            mtree_dict['data']=root.data
            mtree_dict['children']=[]
            for child in root.children:
                mtree_dict['children'].append(mtree_to_dict(child))
        return mtree_dict

def translate_number(mtree,num_list,const_list):
    mtree['const']=False
    if mtree['data'] in const_list:
        mtree['const']=True
        mtree['data']==float(mtree['data'])
    else:
        try:
            mtree['data']=eval(mtree['data'])
        except:
            pass
    for child in mtree['children']:
        translate_number(child,num_list,const_list)
    

def evaluate(data):
    old_idx2word = ['*', '-', '+', '/', '1', 'PI','None','temp_a','temp_b','temp_c',
                               'temp_d','temp_e','temp_f','temp_g','temp_h','temp_i','temp_j','temp_k',
                               'temp_l','temp_m','temp_n','temp_o']
    old_word2idx = {word:idx for idx,word in enumerate(old_idx2word)}
    old_num_start = old_word2idx['1']
    
    total_case=len(data)
    print('Total case: ',total_case)
    target_default_case=0
    pred_default_case=0
    mtree_right=0
    iou_right=0
    exp_right=0
    ans_right=0
    result=[]
    with tqdm(enumerate(data),desc='Evaluate data',total=total_case) as pbar:
        for i,case in pbar:
            target_default_flag=False
            pred_default_flag=False
            mtree_right_flag=False
            iou_right_rate=0
            exp_right_flag=False
            ans_right_flag=False
            # infix_exp=prefix_to_infix(' '.join(case['true_exp_token']))
            infix_exp=case['pred_exp']
            try:
                target_mtree,num_to_code, code_to_num = exp_to_mtree(infix_exp,old_idx2word,old_num_start)
            except:
                target_default_flag=True
            # infix_exp=prefix_to_infix(' '.join(case['prediction_exp_token']))
            infix_exp=case['targ_exp']
            try:
                pred_mtree,num_to_code, code_to_num = exp_to_mtree(infix_exp,old_idx2word,old_num_start)
            except:
                pred_default_flag=True
            target_default_case+=target_default_flag
            pred_default_case+=pred_default_flag
            if target_default_flag:
                result.append({
                    'id':case['id'],
                    'mtree_flag':False,
                    'iou_flag':0,
                    'exp_flag':False,
                    'ans_flag':False,
                })
                continue
            else:
                if pred_default_flag:
                    result.append({
                        'id':case['id'],
                        'mtree_flag':False,
                        'iou_flag':0,
                        'exp_flag':False,
                        'ans_flag':False,
                    })
                    continue
                else:
                    # num_list=case['num_list']
                    # const_list=['1','3.14']
                    target_mtree_for_case=mtree_to_dict(target_mtree)
                    # translate_number(target_mtree_for_case,num_list=num_list,const_list=const_list)
                    pred_mtree_for_case=mtree_to_dict(pred_mtree)
                    # translate_number(pred_mtree_for_case,num_list=num_list,const_list=const_list)
                    target_mtree_code=mtree_to_code(target_mtree)
                    pred_mtree_code=mtree_to_code(pred_mtree)
                    target_code_str=code_to_str(target_mtree_code)
                    pred_code_str=code_to_str(pred_mtree_code)
                    mtree_right_flag,iou_right_rate=mtree_and_iou(target_code_str,pred_code_str)
                    exp_right_flag=exp_equal(case['pred_exp'],case['targ_exp'])
                    ans_right_flag=abs(case["pred_ans"]-case["targ_ans"])<1e-5
                    result.append({
                        'id':case['id'],
                        'mtree_flag':mtree_right_flag,
                        'iou_flag':iou_right_rate,
                        'exp_flag':exp_right_flag,
                        'ans_flag':ans_right_flag,
                    })

                    mtree_right+=mtree_right_flag
                    iou_right+=iou_right_rate
                    exp_right+=exp_right_flag
                    ans_right+=ans_right_flag
    print(f'target default: {target_default_case}')
    print(f'pred default: {pred_default_case}')
    return mtree_right/total_case,iou_right/total_case,exp_right/total_case,ans_right/total_case,result

def write_data(data,file):
    with open(file,'w') as f:
        json_str=json.dumps(data,separators=(',', ': '),ensure_ascii=False,indent=4)
        f.write(json_str)

if __name__=='__main__':
    data=read_data('examples.json')
    mtree_acc,iou_acc,exp_acc,value_acc,result=evaluate(data)
    print('--------------------------------')
    print(f'mtree_acc: {mtree_acc}')
    print(f'IOU_acc: {iou_acc}')
    print(f'exp_acc: {exp_acc}')
    print(f'value_acc: {value_acc}')
    print('--------------------------------')
    write_data(result,'result.json')