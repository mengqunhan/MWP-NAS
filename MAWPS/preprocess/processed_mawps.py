import json
import os
source_path='./datasets/mawps/mawps_5fold/source/'
processed_path='./datasets/mawps/mawps_5fold/processed/'
special=list("γδεζηθικλμνορςστυ")
temp=list("abcdefghijklmnopq")
special2temp={}
for s,t in zip(special,temp):
    special2temp[s]='temp_'+t
for i in range(5):
    with open(source_path+'Fold_'+str(i)+'/train_mawps_new_mwpss_fold_'+str(i)+'.json','r') as f:
        train=json.load(f)
    with open(source_path+'Fold_'+str(i)+'/test_mawps_new_mwpss_fold_'+str(i)+'.json','r') as f:
        test=json.load(f)

    train_processed=[]
    for item in train:
        num_list=[v for v in item['T_number_map'].values()][2:]
        masked_question=item['T_question_2'][8:]
        masked_equation=item['T_equation']
        for s in special:
            masked_question=masked_question.replace(s,special2temp[s])
            masked_equation=masked_equation.replace(s,special2temp[s])
        masked_question=masked_question.replace('α','1')
        masked_equation=masked_equation.replace('α','1')
        masked_question=masked_question.replace('β','PI')
        masked_equation=masked_equation.replace('β','PI')

        new_item={}
        new_item['id']=item['id']
        new_item['original_text']=item['original_text']
        new_item['segmented_text']=item['segmented_text']
        new_item['processed_text']=masked_question
        new_item['equation']=item['equation']
        new_item['ans']=float(item['ans'])
        new_item['processed_equation']=masked_equation
        new_item['num_list']=num_list

        train_processed.append(new_item)
    
    test_processed=[]
    for item in test:
        num_list=[v for v in item['T_number_map'].values()][2:]
        masked_question=item['T_question_2'][8:]
        masked_equation=item['T_equation']
        for s in special:
            masked_question=masked_question.replace(s,special2temp[s])
            masked_equation=masked_equation.replace(s,special2temp[s])
        masked_question=masked_question.replace('α','1')
        masked_equation=masked_equation.replace('α','1')
        masked_question=masked_question.replace('β','PI')
        masked_equation=masked_equation.replace('β','PI')

        new_item={}
        new_item['id']=item['id']
        new_item['original_text']=item['original_text']
        new_item['segmented_text']=item['segmented_text']
        new_item['processed_text']=masked_question.lower()
        new_item['equation']=item['equation']
        new_item['ans']=item['ans']
        new_item['processed_equation']=masked_equation
        new_item['num_list']=num_list

        test_processed.append(new_item)

    if not os.path.exists(processed_path+'Fold_'+str(i)):
        os.makedirs(processed_path+'Fold_'+str(i))
    with open(processed_path+'Fold_'+str(i)+'/train_mawps_new_mwpss_fold_'+str(i)+'.json','w') as f:
        jsonstr=json.dumps(train_processed,indent=4,separators=(',',':'),ensure_ascii=False)
        f.write(jsonstr)
    with open(processed_path+'Fold_'+str(i)+'/test_mawps_new_mwpss_fold_'+str(i)+'.json','w') as f:
        jsonstr=json.dumps(test_processed,indent=4,separators=(',',':'),ensure_ascii=False)
        f.write(jsonstr)


print('hello')