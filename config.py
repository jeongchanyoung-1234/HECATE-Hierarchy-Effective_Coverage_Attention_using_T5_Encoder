import argparse

class Config:
    parser = argparse.ArgumentParser()
    
    # legal dataset only
    parser.add_argument('--rule_based_date_parsing', action='store_true')
    parser.add_argument('--prostitution_only', action='store_true')

    # path
    parser.add_argument("--save_path", default="./save/copy_t5_summ.pt", type=str)
    parser.add_argument('--load_path', default=None)
    parser.add_argument("--train_data_path", default="./data/Training_set_prostitution_only.json", type=str)
    parser.add_argument("--valid_data_path", default="./data/Valid_set_prostitution_only.json", type=str)

    # hyperparameters
    parser.add_argument('--huggingface_model_name', default='KETI-AIR-Downstream/long-ke-t5-base-summarization', type=str, help='KETI-AIR/long-ke-t5-small, KETI-AIR/ke-t5-large')
    parser.add_argument('--max_input_length', default=4096, type=int)
    parser.add_argument('--max_target_length', default=512, type=int)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--prefix', default='none', type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--warmup_stage', default=0, type=int)
    parser.add_argument('--linear_warmup', action='store_true')
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument('--lr_decay_factor', default=2, type=int)
    parser.add_argument('--early_stopping_round', default=3, type=int)
    parser.add_argument('--early_stopping_criterion', default='bleu', type=str, help='bleu')
    parser.add_argument('--num_beams', default=1, type=int)

    # copy mechanism
    parser.add_argument('--use_copy', default=True, type=bool)
    parser.add_argument('--gen_loss_lambda', type=float, default=.7)
    parser.add_argument('--coverage', default=True, type=bool)
    parser.add_argument('--coverage_version', type=int, default=3)

    # distributed training  
    parser.add_argument("--prefetch_n_workers", default=1, type=int)

    # etc
    parser.add_argument('--print_example', action='store_true', help='print decoded result')
    parser.add_argument('--print_test_result', action='store_true', help='for benchmark dataset')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--train_n_data', default=1e8, type=int)

     # do not touch
    parser.add_argument('--port', default=10006, type=int)
    parser.add_argument('--seed', default=42, type=int) #42
    parser.add_argument('--local_rank', default=0, type=int)
