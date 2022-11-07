from models.nguyenvulebinh_qa.model.mrc_model import MRCQuestionAnswering
from models.mailong_qa.reader import Reader
from transformers import AutoTokenizer



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
    elif model_name == "mailong":
        model = Reader(device=device)
        return {
            "model": model,
            "tokenizer": None,
            "device": device
        }
    

    