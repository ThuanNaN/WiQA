from models.nguyenvulebinh_qa.model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer

import re



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
    

    