from utils.args import get_args
from utils.tools import set_seed
from transformers import BertTokenizerFast
from preprocess.MTreeDataset import MTreeDataset
from torch.utils.data import DataLoader
from models.models import UniversalModel #
from train_and_eval import train, evaluate, test #
import torch

def main():
    args=get_args()
    set_seed(args.seed)
    tokenizer=BertTokenizerFast.from_pretrained(args.bert_model_name)

    if args.mode=='train':
        train_dataset=MTreeDataset(args,tokenizer,args.train_file,args.train_number,'train')
        train_dataloader=DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_dataset.collate_function
        )

        val_dataset=MTreeDataset(args,tokenizer,args.val_file,args.val_number,'val')
        val_dataloader=DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_dataset.collate_function
        )

        model=train(
            args,
            UniversalModel,
            train_dataloader,
            val_dataloader,
            train_dataset.new_word2idx,
            tokenizer)
    else:
        test_dataset = MTreeDataset(args, tokenizer, args.test_file, args.test_number, 'test')
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=test_dataset.collate_function
        )
        model = UniversalModel.from_pretrained(
            f"check_points/{args.model_folder}",
            args
        ).to(torch.device(args.device))
        if test_dataloader is not None:
            acc_mtree, acc_node, acc_format, acc_value, iou = test(args, test_dataloader, model,
                                                                        test_dataset.new_idx2word,
                                                                        test_dataset.new_word2idx,
                                                                        torch.device(args.device))

            print(f"mtree acc:{acc_mtree},node acc:{acc_node},format acc:{acc_format}, value acc:{acc_value}, iou:{iou}")

if __name__ == '__main__':
    main()
