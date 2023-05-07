from argparse import ArgumentParser

def get_args():
    parser=ArgumentParser(description='MTree Solver LOSS1 05 end 2atten')

    #Runtime parameter
    parser.add_argument('--mode',type=str,dest='mode',default='train',help='train or test')
    parser.add_argument('--device',type=str,dest='device',default='cuda:0',help='device')
    parser.add_argument('--early_stop',type=int,dest='early_stop',default=500,help='early stop')

    parser.add_argument('--train_number',type=int,dest='train_number',default=-1,help='train number')
    parser.add_argument('--val_number',type=int,dest='val_number',default=-1,help='val number')
    parser.add_argument('--test_number',type=int,dest='test_number',default=-1,help='test number')

    #Hyper-parameter
    parser.add_argument('--seed',type=int,dest='seed',default=3407,help='random seed')
    parser.add_argument('--epochs',type=int,dest='epochs',default=1000,help='train epochs')
    parser.add_argument('--batch_size',type=int,dest='batch_size',default=8,help='batch size')
    parser.add_argument('--hidden_size',type=int,dest='hidden_size',default=768,help='hidden size')

    parser.add_argument('--lr', type=float, dest='lr', default=2e-5, help='learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")

    #Dataset parameter
    parser.add_argument('--num_workers',type=int,dest='num_workers',default=0,help='num workers')
    parser.add_argument('--num_constant',type=int,dest='num_constant',default=4,help='num constant')
    parser.add_argument('--num_start',type=int,dest='num_start',default=6,help='num start')
    parser.add_argument('--train_file',type=str,dest='train_file',default='./datasets/train23k_processed_nodup.json',help='train file')
    parser.add_argument('--val_file',type=str,dest='val_file',default='./datasets/valid23k_processed_nodup.json',help='val file')
    parser.add_argument('--test_file',type=str,dest='test_file',default='./datasets/test23k_processed_nodup.json',help='test file')
    parser.add_argument('--pad_token',type=str,dest='pad_token',default='PAD',help='pad token')
    parser.add_argument('--end_token', type=str, dest='end_token', default='EOS', help='end token')

    #Data cut
    parser.add_argument('--cut_height',type=int,dest='cut_height',default=5,help='cut height')
    parser.add_argument('--cut_width',type=int,dest='cut_width',default=8,help='cut width')
    parser.add_argument('--eval_cut_height',type=int,dest='eval_cut_height',default=5,help='eval cut height')
    parser.add_argument('--eval_cut_width',type=int,dest='eval_cut_width',default=8,help='eval cut width')

    #Encoder parameter
    parser.add_argument('--bert_model_name',type=str,dest='bert_model_name',default='hfl/chinese-roberta-wwm-ext',help='tokenizer')  #chinese-roberta-wwm-ext chinese-bert-wwm-ext

    #Decoder parameter
    parser.add_argument('--num_attn_heads',type=int,dest='num_attn_heads',default=8,help='num attn heads')
    parser.add_argument('--hidden_dropout',type=float,dest='hidden_dropout',default=0.1,help='hidden dropout')
    parser.add_argument('--layer_norm_eps',type=float,dest='layer_norm_eps',default=1e-12,help='layer norm eps')
    parser.add_argument('--leaky_relu_slope', type=float, dest='leaky_relu_slope', default=0.1)
    parser.add_argument('--num_format',type=int,dest='num_format',default=4,help='num format')

    #Position_embedding parameter
    parser.add_argument('--max_pos_num',type=int,dest='max_pos_num',default=32,help='max pos num')
    parser.add_argument('--pos_dropout',type=float,dest='pos_dropout',default=0.1,help='pos dropout')

    #Pointer network parameter
    parser.add_argument('--ptr_dropout',type=float,dest='ptr_dropout',default=0.5,help='ptr dropout')

    #Train and eval parameter
    parser.add_argument('--model_folder',type=str,dest='model_folder',default='robert-loss2_1-end-2atten',help='model folder') #robert-loss1_05-end-2atten robert-loss2_1-end-2atten

    args=parser.parse_args()
    return args