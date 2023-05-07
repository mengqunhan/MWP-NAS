from transformers import BertTokenizerFast
from preprocess.MTreeDataset_mawps import MTreeDataset
from utils.args import get_args
from torch.utils.data import DataLoader
from train_and_eval import train, evaluate
from models.model import UniversalModel
import torch
def main(i):
    args=get_args()
    train_file=f"./datasets/mawps/mawps_5fold/processed/Fold_{i}/train_mawps_new_mwpss_fold_{i}.json"
    val_file=f"./datasets/mawps/mawps_5fold/processed/Fold_{i}/test_mawps_new_mwpss_fold_{i}.json"
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
    if args.mode=='train':
        train_dataset=MTreeDataset(args,tokenizer,train_file,args.train_number,'train')
        # torch.save(train_dataset,f'./save_files/mawps_5fold/train_dataset_fold_{i}.pt')
        # train_dataset=torch.load(f'./save_files/mawps_5fold/train_dataset_fold_{i}.pt')
        train_dataloader=DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=train_dataset.collate_function
        )
        val_dataset=MTreeDataset(args,tokenizer,val_file,args.val_number,'val')
        # torch.save(val_dataset,f'./save_files/mawps_5fold/test_dataset_fold_{i}.pt')
        # val_dataset=torch.load(f'./save_files/mawps_5fold/test_dataset_fold_{i}.pt')
        val_dataloader=DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_dataset.collate_function
        )
        best_value_acc=train(
            args,
            UniversalModel,
            train_dataloader,
            val_dataloader,
            train_dataset.new_idx2word,
            train_dataset.new_word2idx,
            tokenizer,
            fold=i)
        return best_value_acc

if __name__ == '__main__':
    mtree_acc=0
    for i in range(5):
        mtree_acc+=main(i)
    print(f"average value acc:{mtree_acc/5}")
