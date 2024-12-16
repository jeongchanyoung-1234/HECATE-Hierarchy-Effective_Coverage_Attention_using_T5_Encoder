import re
import os
import time
from copy import deepcopy

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from module.utils import rename_path, field_preprocess, encode_line_s


class ComplaintDataset(Dataset):
    def __init__(self,
                 path,
                 config,
                 tokenizer,
                 is_train=True,
                 n_data=1e8):
        self.tokenizer = tokenizer
        self.counter = {'s':0, 'u':0, 'v':0, 'else':0}
        self.prostitution_only = config.prostitution_only
        
        start = time.time()

        with open(path, 'r') as f:
            lines = f.readlines()

        input_ids, labels = [], []

        t = lines[:int(n_data)] if len(lines) > n_data else lines
        for line in t:
            # if config.verbose > 0:
            #     t.set_description('Loading {} samples... ({:.4f}sec)'.format('training' if is_train else 'validtion', time.time() - start))
            dic = eval(line)
            try:
                if dic['input'].strip() == '' or dic['output'].strip() == '':
                    continue
            except:
                pass

            domain = dic['input'].split('<SEP>')[0]
            if self.prostitution_only and '성매매' not in domain:
                continue

            if '성매매' in domain:
                self.counter['s'] += 1
            elif '중고' in domain:
                self.counter['u'] += 1
            elif '보이스' in domain:
                self.counter['v'] += 1
            else:
                self.counter['else'] += 1

            input_ids.append(dic['input'])
            labels.append(dic['output'])
            

        self.data = list(zip(input_ids, labels))

    def __getitem__(self, idx):
        return self.data[idx]    

    def __len__(self):
        return len(self.data)

    def print_data_statistics(self):
        print('Total: {}'.format(sum([v for v in self.counter.values()])))
        print('S:{s} U:{u} V:{v}'.format(**self.counter))

class ComplaintDataLoader:
    def __init__(self,
                 config,
                 tokenizer,
                 mode='complaint'):
        if mode == 'complaint':
            self.dataset = ComplaintDataset
        elif mode == 'paraphrase':
            self.dataset = ParaphraseDataset
        elif mode == 'infill':
            self.dataset = InfillDataset
        else:
            raise NotImplementedError

        self.config = config
        self.train_path = config.train_data_path
        self.test_path = config.valid_data_path
        self.batch_size = config.batch_size
        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length
        self.tokenizer = tokenizer
        self.prefetch_n_workers = config.prefetch_n_workers
        self.verbose = config.verbose
        self.train_n_data = config.train_n_data
        if mode == 'paraphrase':
            self.prefix = 'paraphrase'
        else:
            if config.prefix == 'none':
                self.prefix = ''
            else:
                self.prefix = config.prefix + ":"

    def collate_fn(self, batch):
        input_ids, labels = [], []

        for data in batch:
            inp, outp = data
            inp = self.tokenizer(self.prefix + inp, max_length=self.max_input_length, truncation=True, padding='longest')
            # try:
            with self.tokenizer.as_target_tokenizer():
                outp = self.tokenizer(str(outp), max_length=self.max_target_length, truncation=True, padding='longest')
            # except:
            #     print(outp)
            #     break
            input_ids.append(torch.tensor(inp['input_ids']))
            labels.append(torch.tensor(outp['input_ids']))

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(dtype=torch.long)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True).to(dtype=torch.long)

        return input_ids, labels


    def dual_attention_collate_fn(self, batch):
        input_ids, labels = [], []
        fields = []
        lpos, rpos = [] ,[] 

        for data in batch:
            inp, outp = data
            content, field = [], []
            l, r = [], []

            for slot, value in inp.split('<SEP>'):
                value_encoded = self.tokenizer.encode(value, add_special_tokens=False)
                i = 0
                for _ in range(len(value_encoded)):
                    field += self.tokenizer.encode(slot, add_special_tokens=False)
                    l += [i]
                    r += [len(value_encoded) - i]
                content += value_encoded

            print(content)
            print(len(content))
            print(field)
            print(len(field))

            with self.tokenizer.as_target_tokenizer():
                outp = self.tokenizer(outp, max_length=self.max_target_length, truncation=True, padding='longest')

            input_ids.append(torch.tensor(content[:self.max_input_length - 1] + [1]))
            fields.append(torch.tensor(fields[:self.max_input_length - 1] + [1]))
            labels.append(torch.tensor(outp['input_ids']))

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(dtype=torch.long)
        fields = nn.utils.rnn.pad_sequence(fields, batch_first=True).to(dtype=torch.long)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True).to(dtype=torch.long)

        return (input_ids, fields), labels


    def get_dataloader(self, is_train=True):
        test_dataset = self.dataset(self.test_path, self.config, self.tokenizer, is_train=False)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=self.batch_size,
                                 shuffle=False,  
                                 collate_fn=self.collate_fn, 
                                 num_workers=self.prefetch_n_workers, 
                                 persistent_workers=True)
        if not is_train:
            return test_loader

        train_dataset = self.dataset(self.train_path, self.config, self.tokenizer, n_data=self.train_n_data)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True, 
                                  collate_fn=self.collate_fn, 
                                  num_workers=self.prefetch_n_workers, 
                                  persistent_workers=True)

        if self.verbose > 0:
            print('[DATASET]')
            print('- Training')
            train_dataset.print_data_statistics()
            print('- Validation')
            test_dataset.print_data_statistics()

        return train_loader, test_loader


class InfillDataset(ComplaintDataset):
    def __init__(self,
                 path,
                 config,
                 tokenizer,
                 is_train=True):
        super(InfillDataset, self).__init__(path, config, tokenizer, is_train)

    def __getitem__(self, idx):
        return self.data[idx][0]    

    def __len__(self):
        return len(self.data)

    def print_data_statistics(self):
        super(InfillDataset, self).print_data_statistics()

class InfillDataLoader(ComplaintDataLoader):
    def __init__(self, config, tokenizer):
        super(InfillDataLoader, self).__init__(config, tokenizer, 'infill')
        self.input_unit = config.input_unit
        self.num_masked_slot = config.num_masked_slot if self.input_unit == 'paragraph' else None

    def collate_fn(self, batch):
        input_ids, labels = [], []
        for input_id in batch:
            if self.input_unit == 'paragraph':
                while True:
                    rows = input_id.split('<SEP>')
                    idxs = np.random.randint(0, len(rows), size=self.num_masked_slot)
                    label = ''
                    for cnt, idx in enumerate(idxs):
                        if ':' not in rows[idx]:
                            print('retry: {}'.format(rows[idx])) 
                            continue
                        label += '<extra_id_{}>'.format(cnt) + rows[idx].split(':')[0]
                    break

                for cnt, idx in enumerate(idxs):
                    _, value = rows[idx].split(' : ')
                    rows[idx] = ' : '.join(('<extra_id_{}>'.format(cnt),value))
                input_id = '<SEP>'.join(rows)
            elif self.input_unit == 'sentence':
                while True:
                    rows = input_id.split('<SEP>')
                    idx = np.random.randint(0, len(rows), size=self.num_masked_slot)
                    if ':' not in rows[idx]:
                        print('retry: {}'.format(rows[idx])) 
                        continue
                    label = '<extra_id_0>' + rows[idx].split(':')[0]
                    print(label)
                    break

                _, value = rows[idx].split(' : ')
                rows[idx] = ' : '.join(('<extra_id_0>', value))
                input_id = rows[idx]
            else:
                raise NotImplementedError('Please verify a valid input unit')

            inp = self.tokenizer(self.prefix + input_id, max_length=self.max_input_length, truncation=True, padding='longest')
            outp = self.tokenizer(label, max_length=self.max_target_length, truncation=True, padding='longest')

            input_ids.append(torch.tensor(inp['input_ids']))
            labels.append(torch.tensor(outp['input_ids']))

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(dtype=torch.long)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True).to(dtype=torch.long)

        return input_ids, labels

    def get_dataloader(self, is_train=True):
        return super(InfillDataLoader, self).get_dataloader(is_train)

class ParaphraseDataset(Dataset):
    def __init__(self,
                 path,
                 config,
                 tokenizer,
                 is_train=True):
        self.tokenizer = tokenizer
        start = time.time()

        with open(path, 'r') as f:
            lines = f.readlines()

        input_ids, labels = [], []

        t = tqdm(lines, miniters=3) if config.verbose > 0 else lines
        for line in t:
            if config.verbose > 0:
                t.set_description('Loading {} data... ({:.4f}sec)'.format('training' if is_train else 'validation', time.time() - start))

            sent1, sent2 = line.split('\t')[1:-1]
            if sent1.strip() == '' or sent2 == ''.strip():
                continue
                    
            input_ids.append(sent1)
            labels.append(sent2)

        self.data = list(zip(input_ids, labels))

    def __getitem__(self, idx):
        return self.data[idx]    

    def __len__(self):
        return len(self.data)

    def print_data_statistics(self):
        print('Total: {}'.format(self.__len__()))
        

class ParaphraseDataloader(ComplaintDataLoader):
    def __init__(self, config, tokenizer):
        super(ParaphraseDataloader, self).__init__(config, tokenizer, 'paraphrase')

    def collate_fn(self, batch):
        return super(ParaphraseDataloader, self).collate_fn(batch)

    def get_dataloader(self, is_train=True):
        return super(ParaphraseDataloader, self).get_dataloader(is_train)