

class FieldDataset(Dataset):
    def __init__(self,
                 path,
                 config,
                 tokenizer,
                 is_train=True):
        self.tokenizer = tokenizer
        
        start = time.time()
        with open(path, 'r') as f:
            lines = f.readlines()

        input_ids, fields, labels = [], [] ,[]

        t = lines
        for line in t:
            # if config.verbose > 0:
            #     t.set_description('Loading {} samples... ({:.4f}sec)'.format('training' if is_train else 'validtion', time.time() - start))

            dic = eval(line)
            if dic['output'].strip() == '':
                continue
                
            field, input_id = '' ,''
            for d in dic['input']:
                field += d['field'] + '\t'
                input_id += d['value'] + '\t'
            field, input_id = field.rstrip(), input_id.rstrip()

            fields.append(field)
            input_ids.append(input_id)
            labels.append(dic['output'])

        self.data = list(zip(input_ids, fields, labels))

    def __getitem__(self, idx):
        return self.data[idx]    

    def __len__(self):
        return len(self.data)

class FieldDataLoader:
    def __init__(self,
                 config,
                 tokenizer):
        self.config = config
        self.dataset = FieldDataset
        self.train_path = rename_path(config.train_data_path)
        self.test_path = rename_path(config.valid_data_path)
        if not os.path.isfile(self.train_path):
            field_preprocess(config.train_data_path, config.valid_data_path)
        self.batch_size = config.batch_size
        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length
        self.tokenizer = tokenizer
        self.prefetch_n_workers = config.prefetch_n_workers
        self.verbose = config.verbose
        with open('./data/field_vocab.json') as f:
            self.field2id = eval(f.read())

        if config.prefix == 'none':
            self.prefix = ''
        else:
            self.prefix = config.prefix + ":" 

    def collate_fn(self, batch):
        input_ids, fields, labels = [], [], []
        lpos, rpos = [], []

        for data in batch:
            input, field, output = data
            input_split = input.split('\t')
            field_split = field.split('\t')

            _input_ids, _fields = [] ,[] 
            l, r = [], []
            for idx, inp in enumerate(input_split):
                f = field_split[idx].strip()
                inp_encoded= self.tokenizer.encode(inp, add_special_tokens=False)
                f_encoded = [self.field2id[f] for _ in range(len(inp_encoded))]
                l += [i + 1 for i in range(len(inp_encoded))]
                r += [len(inp_encoded) - i for i in range(len(inp_encoded))]
                _input_ids += inp_encoded
                _fields += f_encoded

            with self.tokenizer.as_target_tokenizer():
                output = self.tokenizer(output, max_length=self.max_target_length, truncation=True, padding='longest')

            input_ids.append(torch.tensor(_input_ids[:self.max_input_length]))
            fields.append(torch.tensor(_fields[:self.max_input_length]))
            lpos.append(torch.tensor(l[:self.max_input_length]))
            rpos.append(torch.tensor(r[:self.max_input_length]))
            labels.append(torch.tensor(output['input_ids']))

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(dtype=torch.long)
        fields = nn.utils.rnn.pad_sequence(fields, batch_first=True).to(dtype=torch.long)
        lpos = nn.utils.rnn.pad_sequence(lpos, batch_first=True).to(dtype=torch.long)
        rpos = nn.utils.rnn.pad_sequence(rpos, batch_first=True).to(dtype=torch.long)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True).to(dtype=torch.long)

        return (input_ids, fields, lpos, rpos), labels


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

        train_dataset = self.dataset(self.train_path, self.config, self.tokenizer)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.batch_size, 
                                  shuffle=True, 
                                  collate_fn=self.collate_fn, 
                                  num_workers=self.prefetch_n_workers, 
                                  persistent_workers=True)

        return train_loader, test_loader



class WebNLGDataset(Dataset):
    def __init__(self,
                 src_path,
                 tgt_path,
                 tokenizer,
                 is_train=True):
        self.tokenizer = tokenizer
        start = time.time()

        with open(src_path, 'r') as f:
            src_lines = f.readlines()
        with open(tgt_path, 'r') as f:
            tgt_lines = f.readlines()   
        
        input_ids, labels = [], []

        t = list(zip(src_lines, tgt_lines))
        
        for src_line, tgt_line in t:
            src_line, tgt_line = eval(src_line), tgt_line

            input_ids.append(src_line)
            labels.append(tgt_line)

        self.data = list(zip(input_ids, labels))

    def __getitem__(self, idx):
        return self.data[idx]    

    def __len__(self):
        return len(self.data)

    def print_data_statistics(self):
        print('Total: {}'.format(self.__len__()))

class WebNLGDataLoader(DataLoader):
    def __init__(self,
                 config,
                 tokenizer):
        self.dataset = WebNLGDataset
        
        self.config = config
        self.train_path = config.train_data_path
        self.test_path = config.valid_data_path
        self.batch_size = config.batch_size
        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length
        self.tokenizer = tokenizer
        self.prefetch_n_workers = config.prefetch_n_workers
        self.verbose = config.verbose
        if config.prefix == 'none':
            self.prefix = ''
        else:
            self.prefix = config.prefix.replace('_', ' ') + ":" 

    def collate_fn(self, batch):
        input_ids, labels = [], []
        role, tree, attn_mask = [], [], []

        for data in batch:
            inp, outp = data
            inp_ret = encode_line_s(self.tokenizer, inp, self.prefix, self.max_input_length)
            with self.tokenizer.as_target_tokenizer():
                outp_ret = self.tokenizer(outp, max_length=self.max_target_length, truncation=True, return_tensors='pt')
            
            input_ids.append(inp_ret['input_ids'])
            labels.append(outp_ret['input_ids'].squeeze(0))
            role.append(inp_ret['src_role'])
            tree.append(inp_ret['src_tree'])
            attn_mask.append(inp_ret['attention_mask'])

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(dtype=torch.long)
        labels = nn.utils.rnn.pad_sequence(labels, batch_first=True).to(dtype=torch.long)
        role = nn.utils.rnn.pad_sequence(role, batch_first=True, padding_value=1.0).to(dtype=torch.long)
        tree = nn.utils.rnn.pad_sequence(tree, batch_first=True, padding_value=1.0).to(dtype=torch.long)
        attn_mask = nn.utils.rnn.pad_sequence(attn_mask, batch_first=True).to(dtype=torch.long)

        return (input_ids, role, tree, attn_mask), labels

    def get_dataloader(self, is_train=True):
        test_src_path = self.test_path + '.source'
        test_tgt_path = self.test_path + '.target'
        test_dataset = self.dataset(test_src_path, test_tgt_path, self.tokenizer, is_train=False)
        test_loader = DataLoader(test_dataset, 
                                 batch_size=self.batch_size,
                                 shuffle=False,  
                                 collate_fn=self.collate_fn, 
                                 num_workers=self.prefetch_n_workers, 
                                 persistent_workers=True)
        if not is_train:
            return test_loader
        train_src_path = self.train_path + '.source'
        train_tgt_path = self.train_path + '.target'
        train_dataset = self.dataset(train_src_path, train_tgt_path, self.tokenizer)
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


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.peephole = config.peephole
        self.hidden_dim = config.hidden_dim
        with open('./data/field_vocab.json', 'r') as f:
            dic = eval(f.read())
            self.field_vocab_size = len(dic.keys())
        self.embed = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_dim, padding_idx=0)
        self.f_embed = nn.Embedding(num_embeddings=self.field_vocab_size, embedding_dim=config.field_emb_size, padding_idx=0)
        self.l_embed = nn.Embedding(num_embeddings=config.max_input_length, embedding_dim=config.pos_emb_size, padding_idx=0)
        self.r_embed = nn.Embedding(num_embeddings=config.max_input_length, embedding_dim=config.pos_emb_size, padding_idx=0)
        self.W_h = nn.Linear(in_features=config.hidden_dim, out_features=config.hidden_dim * 4, bias=False)
        self.W_x = nn.Linear(in_features=config.hidden_dim, out_features=config.hidden_dim * 4)
        self.W_f = nn.Linear(in_features=config.field_emb_size + 2 * config.pos_emb_size, out_features=config.hidden_dim * 2)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None, f=None, l=None, r=None):
        device = x.device
        x = self.embed(x)
        if f is not None:
            f = torch.concat([self.f_embed(f), self.l_embed(l), self.r_embed(r)], dim=-1) 

        B, L, _ = x.size()
        hidden_seq, field_seq = [], []
        if init_states is None:
            h_t, c_t = (torch.zeros(B, self.hidden_dim).to(device),
                        torch.zeros(B, self.hidden_dim).to(device))
        else:
            h_t, c_t = init_states

        for t in range(L):
            x_t = x[:, t, :]

            if f is not None:
                f_t = f[:, t, :]
                f_gates = self.W_f(f_t)

                l_t, z_t = (torch.sigmoid(f_gates[:, :self.hidden_dim]),
                            torch.tanh(f_gates[:, self.hidden_dim:])
                            )

            if self.peephole:
                gates = self.W_x(x_t) + self.W_h(c_t)
            else:
                gates = self.W_x(x_t) + self.W_h(h_t)
                g_t = torch.tanh(gates[:, self.hidden_dim * 2:self.hidden_dim * 3])

            i_t, f_t, o_t = (torch.sigmoid(gates[:, :self.hidden_dim]),
                             torch.sigmoid(gates[:, self.hidden_dim:self.hidden_dim * 2]),
                             torch.sigmoid(gates[:, self.hidden_dim * 3:])
                             )
            
            if self.peephole:
                c_t = f_t * c_t + i_t * self.W_x(x_t)[:, self.hidden_dim * 2:self.hidden_dim * 3] + l_t * z_t
                h_t = torch.tanh(o_t * c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                if f is not None:
                    c_t += l_t * z_t
                h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, f