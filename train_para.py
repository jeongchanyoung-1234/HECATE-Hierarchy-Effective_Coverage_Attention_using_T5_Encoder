import argparse

import torch
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from train import define_argparser, get_optimizer
from module.trainer import Trainer
from module.module import Model
from module.utils import set_deterministic_training, print_config
from module.dataloader import ParaphraseDataloader
from ontology import special_tokens_dict

def load(model, optimizer, save_path, local_rank):
    checkpoint = torch.load(save_path, map_location = lambda storage, loc: storage.cuda(local_rank))
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"], checkpoint["score"]
    
if __name__ == '__main__':
    config = define_argparser()

    set_deterministic_training(config)

    device = torch.device(('cuda:{}'.format(config.local_rank)) if torch.cuda.is_available() else torch.device('cpu'))
    print('[Device] {}'.format(device))
    print_config(config)
    
    model = Model(config)
    model.init()

    tokenizer = AutoTokenizer.from_pretrained(config.huggingface_model_name)
    # tokenizer.add_special_tokens(special_tokens_dict)

    dataloader = ParaphraseDataloader(config, tokenizer)
    train_loader, test_loader = dataloader.get_dataloader()
    optim = get_optimizer(model, config)
    scheduler = get_linear_schedule_with_warmup(optim, config.warmup_stage, len(train_loader) * config.epochs) if config.linear_warmup and config.warmup_stage > 0 else None
    trainer = Trainer(model, tokenizer, optim, scheduler, train_loader, test_loader, config, device)

    global_epoch = 0
    try:
        global_epoch, score = load(model, config, optim, config.load_path, config.local_rank)
        print('load from {} (epoch {})'.format(config.load_path, global_epoch))
        trainer.best_loss = score
    except:
        pass

    trainer.train(global_epoch)