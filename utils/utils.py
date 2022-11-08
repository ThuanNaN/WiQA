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

def load_model(model_name, device):
    if model_name == "nguyenvulebinh":
        model_checkpoint = "nguyenvulebinh/vi-mrc-large"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = MRCQuestionAnswering.from_pretrained(model_checkpoint)
        model.to(device)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "device":device
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