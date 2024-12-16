import torch
from transformers import AutoTokenizer

from train import define_argparser, get_optimizer, load
from module.trainer import Trainer
from module.utils import print_config, compare_config
from module.dataloader import ComplaintDataLoader
from module.module import Model
from ontology import special_tokens_dict


    
if __name__ == '__main__':
    config = define_argparser()

    device = torch.device(('cuda:{}'.format(config.local_rank)) if torch.cuda.is_available() else torch.device('cpu'))
    print('[Device] {}'.format(device))
    print_config(config)
    
    model = Model(config)
    model.init()

    tokenizer = AutoTokenizer.from_pretrained(config.huggingface_model_name)
    # tokenizer.add_special_tokens(special_tokens_dict)
    if 'web' in config.train_data_path or 'web' in config.valid_data_path:
        tokens = ['S|', 'P|', 'O|']
        new_tokens_vocab = {}
        new_tokens_vocab['additional_special_tokens'] = []	
        for  t in tokens:	
            new_tokens_vocab['additional_special_tokens'].append(t)	
        tokenizer.add_special_tokens(new_tokens_vocab)
        model.generator.resize_token_embeddings(len(tokenizer))
        config.vocab_size = len(tokenizer)

    dataloader = ComplaintDataLoader(config, tokenizer)
    train_loader, test_loader = dataloader.get_dataloader()
    optim = get_optimizer(model, config)
    scheduler = None

    _, _, prev_config = load(model, config, optim, config.load_path, config.local_rank, return_prev_config=True)
    config = compare_config(config, prev_config)

    global_epoch, score = load(model, config, optim, config.load_path, config.local_rank)
    trainer = Trainer(model, tokenizer, optim, scheduler, train_loader, test_loader, config, device)

    trainer.validate(global_epoch, is_test=True)