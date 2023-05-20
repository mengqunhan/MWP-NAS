from transformers import BertTokenizerFast
from preprocess.MTreeDataset_mawps import MTreeDataset
from utils.args import get_args
from torch.utils.data import DataLoader
from train_and_eval import train, evaluate_value
from models.model import UniversalModel
import torch

def test(fold):
    args=get_args()
    val_file=f"./datasets/mawps/mawps_5fold/processed/Fold_{fold}/test_mawps_new_mwpss_fold_{fold}.json"
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
    val_dataset=MTreeDataset(args,tokenizer,val_file,args.val_number,'val')
    val_dataloader=DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_dataset.collate_function
        )
    load_path=f"check_points/MTree_Solver_robert-{fold}"
    model = UniversalModel.from_pretrained(load_path,args).to(torch.device(args.device))
    acc_mtree, acc_node, acc_format, value = evaluate_value(args, val_dataloader, model, val_dataset.new_idx2word, val_dataset.new_word2idx, torch.device(args.device))
    print(f"mtree acc:{acc_mtree},node acc:{acc_node},format acc:{acc_format},value acc:{value}")
    return acc_mtree, value

if __name__ == '__main__':
    total_mtree=0
    total_value=0
    for i in range(5):
        mtree_acc,value_acc=test(i)
        total_mtree+=mtree_acc
        total_value+=value_acc
    print(f"average mtree acc:{total_mtree/5}")
    print(f"average value acc:{total_value/5}")
