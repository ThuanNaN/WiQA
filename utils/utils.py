from qa_models.nguyenvulebinh_qa.model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer
import re
import subprocess
import yaml
import os


CONFIG_PATH = f"{os.getcwd()}/configs/"

def load_config(config_name="base.yaml") -> dict:
    """
    Function to load YAML configuration file, default: "base.yaml"
    """
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

CONFIG = load_config()

def load_model(CONFIG):
    if CONFIG['model']['name'] == "nguyenvulebinh":
        model_chpt = CONFIG['model']['model_ckpt']
        tokenizer_ckpt = CONFIG['model']['tokenizer_ckpt']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt)
        model = MRCQuestionAnswering.from_pretrained(model_chpt)
        model.to(CONFIG['model']['device'])
        return {
            "model": model,
            "tokenizer": tokenizer,
            "device":CONFIG['model']['device']
        } 

def clean_text(text):
    text = text.lower()
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('=*', '', text)
    text = re.sub('(BULLET:*.) | (BULLET:*[0-9].)', '', text)

    return text
    
def execute(cmd):
    """
    This function is for displaying `Lucene Indexer` output
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