import logging
from threading import Thread

import re
import torch
from flask import Flask, request
from transformers import AutoTokenizer

from train import  get_optimizer, load
from module.module import Model
from module.dataloader import ComplaintDataLoader
from module.trainer import Trainer
from module.utils import compare_config
from config import Config

config = Config()
parser = config.parser
config,unknown = parser.parse_known_args()

device = torch.device(('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu'))

model = Model(config)
model.init()
tokenizer = AutoTokenizer.from_pretrained(config.huggingface_model_name)
# dataloader = ComplaintDataLoader(config, tokenizer)
# train_loader, test_loader = dataloader.get_dataloader()
train_loader, test_loader = None, None
optim = get_optimizer(model, config)
scheduler = None

config.load_path = './save/copy_t5_summ.pt'
config.huggingface_model_name = 'KETI-AIR-Downstream/long-ke-t5-base-summarization'
_, _, prev_config = load(model, config, optim, config.load_path, config.local_rank, return_prev_config=True)
global_epoch, score = load(model, prev_config, optim, prev_config.save_path, config.local_rank)

trainer = Trainer(model, tokenizer, optim, scheduler, train_loader, test_loader, prev_config, device)

# Setting for Flask communication
logger = logging.getLogger("Dialog")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

logger.info("Model is loaded.")

app = Flask(__name__)

@app.route("/response", methods=["GET", "POST"])
def process():
    try:
        in_request = request.json
        print(in_request)
    except:
        return "invalid input: {}".format(in_request)
    
    tmp=[]
    for key, value in in_request.items():
        tmp.append(key)
        tmp.append(":")
        tmp.append(value)
        if key !='다른 민형사':
            tmp.append("<SEP>")
    tmp=" ".join(tmp)
    prompt= config.prefix + tmp

    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    input_ids = input_ids.to('cuda:0')
    output = trainer.generate(input_ids)

    tmp=[]
    for k in output.split():
        if k !="<pad>" and k!= "</s>":
            tmp.append(k)
    print('tmp', tmp)

    line = ' '.join(tmp)
    line = re.sub("</s>","",line)

    return line

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.port)