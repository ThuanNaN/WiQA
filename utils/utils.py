from models import viMRC, Albert, phoBERT
from typing import Dict
import re
import subprocess
import yaml
import os


CONFIG_PATH = f"{os.getcwd()}/configs/"
MODEL_ZOO = ['vimrc', 'phobert', 'albert']

def load_config(config_name="base.yaml") -> dict:
    """
    Function to load YAML configuration file, default: "base.yaml"
    """
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

CONFIG = load_config()

def load_model():
    assert CONFIG['model']['name'] in MODEL_ZOO, \
        ValueError(f"{CONFIG['model']['name']} is not in {MODEL_ZOO}")

    model_ckpt = CONFIG['model']['model_ckpt']
    tokenizer_ckpt = CONFIG['model']['tokenizer_ckpt']
    device = CONFIG['model']['device']


    if CONFIG['model']['name'] == "vimrc":
        model = viMRC(model_pretrained=model_ckpt,
                      tokenizer_pretrained=tokenizer_ckpt,
                      device=device)

    elif CONFIG['model']['name'] == 'phobert':
        model = phoBERT(model_pretrained=model_ckpt,
                        tokenizer_pretrained=tokenizer_ckpt,
                        device=device)

    elif CONFIG['model']['name'] == 'albert':
        model = Albert(model_pretrained=model_ckpt,
                       tokenizer_pretrained=tokenizer_ckpt,
                       device=device)
    else:
        raise NotImplementedError()

    return model

def clean_text(text) -> str:
    text = re.sub('\n',' ', text)
    text = re.sub('\t','', text)
    text = re.sub('=*', '', text)
    text = re.sub('\[[0-9].\]', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub('BULLET:*.', ' ', text)
    return text.strip()
    
def execute(cmd):
    """
    This function is for displaying `Lucene Indexer` command output
    """
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def get_title(doc_id: str, titles_list: list) -> str:
    for rec in titles_list:
        if rec['id'] == doc_id:
            return rec['title']