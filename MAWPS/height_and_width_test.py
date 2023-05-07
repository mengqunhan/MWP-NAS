# import torch
# from transformers import BertTokenizerFast
# from preprocess.MTreeDataset_mawps import MTreeDataset
# from utils.args3 import get_args
# from torch.utils.data import DataLoader
# from train_and_eval3 import train, evaluate
# from models.model2 import UniversalModel
# args=get_args()
# i=0
# train_file=f"./datasets/mawps/mawps_5fold/processed/Fold_{i}_copy/dataset.json"
# val_file=f"./datasets/mawps/mawps_5fold/processed/Fold_{i}/test_mawps_new_mwpss_fold_{i}.json"
# tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
# train_dataset=MTreeDataset(args,tokenizer,train_file,args.train_number,'train')
# train_dataset=torch.load(f'./save_files/mawps_5fold/train_dataset_fold_0.pt')
# train_dataloader=DataLoader(
#     dataset=train_dataset,
#     batch_size=16,
#     shuffle=True,
#     num_workers=0,
#     collate_fn=train_dataset.collate_function
# )
# val_dataset=MTreeDataset(args,tokenizer,val_file,args.val_number,'val')
# torch.save(val_dataset,f'./save_files/mawps_5fold/test_dataset_fold_{i}.pt')
# val_dataset=torch.load(f'./save_files/mawps_5fold/test_dataset_fold_0.pt')
# val_dataloader=DataLoader(
#     dataset=val_dataset,
#     batch_size=16,
#     shuffle=False,
#     num_workers=0,
#     collate_fn=val_dataset.collate_function
# )

import torch
height_and_width=torch.load('./save_files/height_and_width.pt')
max_height=0
max_width=0
for item in height_and_width:
    if item['height']>max_height:
        max_height=item['height']
    if item['width']>max_width:
        max_width=item['width']
height=[0 for _ in range(max_height)]
width=[0 for _ in range(max_width)]
for item in height_and_width:
    height[item['height']-1]+=1
    width[item['width']-1]+=1
print("----------------------------------height----------------------------------")
util_count=0
for i in range(max_height):
    util_count+=height[i]
    print("height: {:<10} count: {:<10} scale: {:<10} until_scale:{:<10}".format(i+1,height[i],round(height[i]/len(height_and_width),4),round(util_count/len(height_and_width),4)))
print("----------------------------------width----------------------------------")
util_count=0
for i in range(max_width):
    util_count+=width[i]
    print("width: {:<10} count: {:<10} scale: {:<10} until_scale:{:<10}".format(i+1,width[i],round(width[i]/len(height_and_width),4),round(util_count/len(height_and_width),4)))
print(max_height,max_width)