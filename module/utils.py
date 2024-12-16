import io, random, argparse, subprocess, re
from copy import deepcopy
from pprint import PrettyPrinter
import numpy as np

import torch

import os

def bleu_score(label, pred_file):
    # bleu_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multi-bleu.perl')

    # with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
    #     bleu_out = subprocess.check_output(
    #         [bleu_script, labels_file],
    #         stdin=predictions_file,
    #         stderr=subprocess.STDOUT)
    #     bleu_out = bleu_out.decode("utf-8")
    #     bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
    #     print(bleu_score)
        # return float(bleu_score)
    dir_path = os.path.dirname(os.path.realpath(__file__))

    cmd_string = "perl " + dir_path + "/multi-bleu.perl -lc " + './data/data_release/{}/test.summary'.format(label) \
        + " < " + pred_file + " > " + pred_file.replace("txt", "bleu")

    os.system(cmd_string)

    try:
        bleu_info = open(pred_file.replace("txt", "bleu"), 'r').readlines()[0].strip()
    except:
        bleu_info = -1

    return bleu_info
    return None


def eval_bleu(pred_file, dataset, folder_data='./data/web_nlg/pos'):

    dir_path = os.path.dirname(os.path.realpath(__file__))

    cmd_string = "perl " + dir_path + "/multi-bleu.perl -lc " + folder_data + "/" + dataset + ".target_eval " \
                  + folder_data + "/" + dataset + ".target2_eval " + folder_data + "/" + dataset + ".target3_eval < " \
                  + pred_file + " > " + pred_file.replace("txt", "bleu")

    os.system(cmd_string)

    try:
        bleu_info = open(pred_file.replace("txt", "bleu"), 'r').readlines()[0].strip()
    except:
        bleu_info = -1

    return bleu_info

def set_deterministic_training(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_config(config):
    pp = PrettyPrinter(indent=4)
    print('\n[CONFIGURATION]')
    config_dic = deepcopy(vars(config))
    unshown_words = ['port', 'local_rank', 'seed']
    for word in unshown_words:
        config_dic.pop(word)
    pp.pprint(config_dic)

def pad_token(vector, max_len):
    padding = max_len - len(vector)
    vector.extend([1] * padding)
    return vector

def pad_attn(vector, max_len):
    padding = max_len - len(vector)
    vector.extend([0] * padding)
    return vector

def encode_line_s(tokenizer, line, prefix, max_length, return_tensors="pt"):
    triples = line
    token = [0]
    attn = [1]
    tree = [0]
    role = [0]
    if prefix != '':
        tokenized = tokenizer(prefix)
        token_tmp = tokenized.input_ids[1:-1]
        attention_mask = tokenized.attention_mask[1:-1]
        len_token = len(token_tmp)
        token += token_tmp
        attn += attention_mask
        role += [0] * len_token
        tree += [0] * len_token

    for h,r,t, tree_index in triples:
        tokenized = tokenizer(h)
        token_tmp = tokenized.input_ids[1:-1]
        attention_mask = tokenized.attention_mask[1:-1]
        len_token = len(token_tmp)

        token += token_tmp
        attn += attention_mask
        role += [3] * len_token
        tree += [tree_index] * len_token

        tokenized = tokenizer(r)
        token_tmp = tokenized.input_ids[1:-1]
        attention_mask = tokenized.attention_mask[1:-1]
        len_token = len(token_tmp)

        token += token_tmp
        attn += attention_mask
        role += [4] * len_token
        tree += [tree_index] * len_token

        tokenized = tokenizer(t)
        token_tmp = tokenized.input_ids[1:-1]
        attention_mask = tokenized.attention_mask[1:-1]
        len_token = len(token_tmp)

        token += token_tmp
        attn += attention_mask
        role += [5] * len_token
        tree += [tree_index] * len_token

    token = token[:max_length-1]    
    role = role[:max_length-1]    
    tree = tree[:max_length-1]    
    attn = attn[:max_length-1]    

    token.append(1)
    role.append(2)
    tree.append(2)
    attn.append(1)
    return {
        "input_ids": torch.LongTensor(token),
        "attention_mask": torch.LongTensor(attn),
        "src_role": torch.LongTensor(role),
        "src_tree": torch.LongTensor(tree)}

def compare_config(current_config:argparse.Namespace, prev_config:argparse.Namespace, replace=False) -> argparse.Namespace:
    current_config_dic = vars(current_config)
    prev_config_dic = vars(prev_config)
    remove_keys = ['verbose', 'rule_based_date_parsing', 'print_example', 'save_path', 'load_path']
    keys = list(current_config_dic.keys())
    if current_config_dic['load_path'] != prev_config_dic['save_path']:
        print(' [Config warning] current load path {} != previous save path {}'.format(current_config_dic['load_path'], prev_config_dic['save_path']))
    for r in remove_keys:
        keys.remove(r)
    for key in keys:
        try:
            if current_config_dic[key] != prev_config_dic[key] and not replace:
                print(' [Config warning] current config {}:{} != previous config {}:{}'.format(
                    key, current_config_dic[key],
                    key, prev_config_dic[key]
                    ))
            elif current_config_dic[key] != prev_config_dic[key] and replace:
                print(' [Config warning] current config {}:{} => previous config {}:{} for consistency'.format(
                    key, current_config_dic[key],
                    key, prev_config_dic[key]
                    ))
                current_config_dic[key] = prev_config_dic[key]
        except:
            print(' [Config warning] {} in current configuration does not exist in previous one'.format(
                key
                )) 

    new_config = argparse.Namespace(**current_config_dic)
    return new_config

def field_preprocess(train_data_path, valid_data_path):
    for d in [train_data_path, valid_data_path]:
        new_fn = '(field)' + d
        with open(d, 'r')  as f:
            corpus = f.readlines()

        if d == 'Training':
            field_vocab = set()

        for line in corpus:
            dic = eval(line)

            slot_values = dic['input'].split(' <SEP> ')
            trunc_slot_values = []
            for slot_value in slot_values:
                slot, value = slot_value.split(' : ')
                
                if d == 'Training':
                    field_vocab.add(slot)   

                trunc_slot_values.append({'field': slot, 'value': value})
            dic['input'] = trunc_slot_values

            with open(new_fn, 'a') as f:
                f.write(str(dic) + '\n')

    field_vocab_list = list(field_vocab)
    vocab = {}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    for idx, f in enumerate(field_vocab_list):
        vocab['{}'.format(f)] = idx + 2

    with open('./field_vocab.json', 'w') as f:
        f.write(str(vocab))

def rename_path(path):
    path_split = path.split('/')
    path_split[-1] = '(field)' + path_split[-1]
    path = '/'.join(path_split)
    return path

def get_positional_encoding(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table