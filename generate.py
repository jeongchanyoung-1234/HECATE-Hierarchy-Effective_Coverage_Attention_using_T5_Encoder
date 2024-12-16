import argparse

import torch
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from module.trainer import Trainer
from module.utils import set_deterministic_training, print_config, compare_config
from module.dataloader import ComplaintDataLoader
from ontology import special_tokens_dict
from train import define_argparser

def get_optimizer(model, config):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.lr,
    )
    return optimizer

def load(save_path, local_rank):
    ckpt = torch.load(save_path, map_location = lambda storage, loc: storage.cuda(local_rank))
    return ckpt
    
if __name__ == '__main__':
    config = define_argparser()
    ckpt = load(config.load_path, config.local_rank)
    try:
        config = compare_config(config, ckpt['config'], replace=False)
    except: 
        pass
    set_deterministic_training(config)

    device = torch.device(('cuda:{}'.format(config.local_rank)) if torch.cuda.is_available() else torch.device('cpu')) if not config.distributed_training else None
    print('[Device] {}'.format(device if not config.distributed_training else 'Distributed training'))

    model = AutoModelForSeq2SeqLM.from_pretrained(config.huggingface_model_name)
    optim = get_optimizer(model, config)
    model.load_state_dict(ckpt["model"])

    tokenizer = AutoTokenizer.from_pretrained(config.huggingface_model_name)
    tokenizer.add_special_tokens(special_tokens_dict)

    dataloader = ComplaintDataLoader(config, tokenizer)
    train_loader, test_loader = dataloader.get_dataloader()
    scheduler = get_linear_schedule_with_warmup(optim, config.warmup_stage, len(train_loader) * config.epochs) if config.warmup_stage > 0 else None
    trainer = Trainer(model, tokenizer, optim, scheduler, train_loader, test_loader, config, device)

    print('load from {} (epoch {})'.format(config.load_path, ckpt['epoch']))

    trainer.generate(is_train=False)