from torch.utils.data import Dataset
import json
from tqdm import tqdm
from preprocess.build_mtree import exp_to_mtree
import copy
import numpy as np
from torch.utils.data._utils.collate import default_collate
import collections

UniFeature = collections.namedtuple(
    'uniFeature',
    'input_ids attention_mask token_type_ids variable_indexs_start variable_indexs_end num_variables variable_index_mask mtree'
)
UniFeature.__new__.__defaults__ = (None,) * len(UniFeature._fields)

class MTreeDataset(Dataset):
    def __init__(self, args, tokenizer, file, number, mode):
        super(MTreeDataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        #old dictionary
        self.old_idx2word = ['*', '-', '+', '/', '^', '1', 'PI', 'None', 'temp_a', 'temp_b', 'temp_c',
                             'temp_d', 'temp_e', 'temp_f', 'temp_g', 'temp_h', 'temp_i', 'temp_j', 'temp_k',
                             'temp_l', 'temp_m', 'temp_n', 'temp_o']
        self.old_word2idx = {word: idx for idx, word in enumerate(self.old_idx2word)}
        self.old_num_start = self.old_word2idx['1']

        # new dictionary
        self.new_idx2word = ['+', '*', '*-', '+/', 'PAD', 'EOS', '0', '1', '2', 'PI', 'temp_a', 'temp_b', 'temp_c',
                             'temp_d', 'temp_e', 'temp_f', 'temp_g', 'temp_h', 'temp_i', 'temp_j', 'temp_k',
                             'temp_l', 'temp_m', 'temp_o', 'temp_p']
        self.new_word2idx = {word: idx for idx, word in enumerate(self.new_idx2word)}
        self.new_constant_values = ['0', '1', '2', 'PI']
        self.quant_list = ['<', 'q', '##uan', '##t', '>']
        self.number = number
        self.mode = mode
        self.read_math23k_file(file)

    def read_math23k_file(self, file):
        data = read_data(file)
        self._features = []
        self.insts = []
        default_pair = 0
        found_duplication_inst_num = 0
        if self.number > 0:
            data = data[:self.number]
        for obj in tqdm(data, desc='Tokenization', total=len(data), ncols=80):
            try:
                mtree, num_to_code, code_to_num = exp_to_mtree(''.join(obj['target_template'][2:]), self.old_idx2word, self.old_num_start)
                self.pseudo_order(mtree)
                mtree_dict = self.mtree_to_dict(mtree)
                height, width = self.get_mtree_height_and_width(mtree_dict)
                if self.mode == 'train':
                    cut_height = self.args.cut_height
                    cut_width = self.args.cut_width
                else:
                    cut_height = self.args.eval_cut_height
                    cut_width = self.args.eval_cut_width
                if height > cut_height or width > cut_width:
                    default_pair += 1
                    continue
            except:
                default_pair += 1
                continue
            mapped_text = obj["text"]
            sent_len = len(mapped_text.split())
            ## replace the variable with <quant>
            for k in range(ord('a'), ord('a') + 26):  
                mapped_text = mapped_text.replace(f"temp_{chr(k)}", " <quant> ")
            mapped_text = mapped_text.split()
            input_text = ""
            for idx, word in enumerate(mapped_text):
                if word.strip() == "<quant>":
                    input_text += " <quant> "
                elif word == "," or word == "，":
                    input_text += word + " "
                else:
                    input_text += word
            res = self.tokenizer.encode_plus(" " + input_text, add_special_tokens=True, return_attention_mask=True)
            input_ids = res["input_ids"]
            token_type_ids = res["token_type_ids"]
            attention_mask = res["attention_mask"]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            var_starts = []
            var_ends = []
            quant_num = len(self.quant_list)
            # quants = ['<', 'q', '##uan', '##t', '>'] if not is_roberta_tokenizer else ['Ġ<', 'quant', '>']
            # obtain the start and end position of "<quant>" token
            for k, token in enumerate(tokens):
                if (token == self.quant_list[0]) and tokens[k:k + quant_num] == self.quant_list:
                    var_starts.append(k)
                    var_ends.append(k + quant_num - 1)

            assert len(input_ids) < 512
            num_variable = len(var_starts)
            assert len(var_starts) == len(obj["num_list"])
            var_mask = [1] * num_variable

            if "nodup" in file:
                eq_set = set()
                for equation in obj["equation_layer"]:
                    eq_set.add(' '.join(equation))
                try:
                    assert len(eq_set) == len(obj["equation_layer"])
                except:
                    found_duplication_inst_num += 1

            self._features.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'variable_indexs_start': var_starts,
                'variable_indexs_end': var_ends,
                'num_variables': num_variable,
                'variable_index_mask': var_mask,
                'mtree': mtree_dict
            })
            self.insts.append(obj)

        print(f"Found {found_duplication_inst_num} instances with duplicated equations")
        print(f"Found {default_pair} instances with default pair")

    def __len__(self):
        return len(self._features)

    def __getitem__(self, index: int):
        return self._features[index]

    def collate_function(self, batch):
        max_wordpiece_length = max([len(feature['input_ids']) for feature in batch])
        max_num_variable = max([feature['num_variables'] for feature in batch])
        max_height, max_width = 0, 0

        if self.mode=='train' or self.mode=='val':
            for feature in batch:
                height,width=self.get_mtree_height_and_width(feature['mtree'])
                max_height=max(max_height,height)
                max_width=max(max_width,width)
        else:
            max_height=self.args.eval_cut_height
            max_width=self.args.eval_cut_width

        return_batch = []

        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature['input_ids'])
            input_ids = feature['input_ids'] + [self.tokenizer.pad_token_id] * padding_length
            attn_mask = feature['attention_mask'] + [0] * padding_length
            token_type_ids = feature['token_type_ids'] + [0] * padding_length
            padded_variable_idx_len = max_num_variable - feature['num_variables']
            var_starts = feature['variable_indexs_start'] + [0] * padded_variable_idx_len
            var_ends = feature['variable_indexs_end'] + [0] * padded_variable_idx_len
            variable_index_mask = feature['variable_index_mask'] + [0] * padded_variable_idx_len

            mtree = copy.deepcopy(feature['mtree'])
            self.pad_mtree(mtree, max_height, max_width)
            mtree_idx = self.mtree_word2idx(mtree)

            return_batch.append(UniFeature(
                input_ids=np.asarray(input_ids),
                attention_mask=np.asarray(attn_mask),
                token_type_ids=np.asarray(token_type_ids),
                variable_indexs_start=np.asarray(var_starts),
                variable_indexs_end=np.asarray(var_ends),
                num_variables=feature['num_variables'],
                variable_index_mask=np.asarray(variable_index_mask),
                mtree=mtree_idx))

        results = UniFeature(*(default_collate(samples) for samples in zip(*return_batch)))

        return results, max_height, max_width

    def mtree_word2idx(self, root):
        new_root = {}
        if 'temp' in root['data'] or root['data'].rstrip('@').rstrip('/').lstrip('-') in self.new_constant_values:
            new_root['data'] = self.new_word2idx[root['data'].rstrip('@').rstrip('/').lstrip('-')]
            if '/' in root['data'] and '-' in root['data']:
                new_root['format'] = 3
            elif '/' in root['data']:
                new_root['format'] = 2
            elif '-' in root['data']:
                new_root['format'] = 1
            else:
                new_root['format'] = 0
        else:
            new_root['data'] = self.new_word2idx[root['data'].rstrip('@')]
            new_root['format'] = 0

        new_root['children'] = []

        if len(root['children']) != 0:
            for child in root['children']:
                new_root['children'].append(self.mtree_word2idx(child))
        return new_root

    def pad_mtree(self, mtree, max_height, max_width):
        pad_token = self.args.pad_token
        end_token = self.args.end_token
        if max_height == 1:
            return
        else:
            if len(mtree['children']) == 0:
                mtree['children'] = [{'data': pad_token, 'children': []} for _ in range(max_width)]
            else:
                if len(mtree['children']) < max_width:
                    mtree['children'].extend([{'data': end_token, 'children': []}])
                mtree['children'].extend(
                    [{'data': pad_token, 'children': []} for _ in range(max_width - len(mtree['children']))])
            for child in mtree['children']:
                self.pad_mtree(child, max_height - 1, max_width)

    def pseudo_order(self, root):
        if len(root.children) == 0:
            return
        children_index = []
        for child in root.children:
            children_index.append(self.new_word2idx[child.data.rstrip('@').rstrip('/').lstrip('-')])
        sorted_id = sorted(range(len(children_index)), key=lambda k: children_index[k], reverse=False)
        new_children = []
        for idx in sorted_id:
            new_children.append(copy.deepcopy(root.children[idx]))
        root.children = new_children
        for child in root.children:
            self.pseudo_order(child)

    def mtree_to_dict(self, root):
        mtree_dict = {}
        if len(root.children) == 0:
            mtree_dict['data'] = root.data
            mtree_dict['children'] = []
            return mtree_dict
        else:
            mtree_dict['data'] = root.data
            mtree_dict['children'] = []
            for child in root.children:
                mtree_dict['children'].append(self.mtree_to_dict(child))
        return mtree_dict

    def get_mtree_height_and_width(self, mtree):
        if len(mtree['children']) == 0:
            return 1, 1
        else:
            height = 1
            width = len(mtree['children'])
            max_child_height = 0
            max_child_width = 0
            for child in mtree['children']:
                child_height, child_width = self.get_mtree_height_and_width(child)
                max_child_height = max(max_child_height, child_height)
                max_child_width = max(max_child_width, child_width)
            height += max_child_height
            width = max(width, max_child_width)
            return height, width


def read_data(file: str):
    with open(file, "r", encoding='utf-8') as read_file:
        data = json.load(read_file)
    return data
