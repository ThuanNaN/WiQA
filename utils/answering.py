import torch
from utils import CONFIG
from models.QA.viMRC import data_collator_2device, extract_answer, tokenize_function_2
import nltk

from typing import Dict

nltk.download('punkt')

def answer(model: Dict, question: str, context: str):
    def viMRC_qa(model: Dict, question: str, context: str):
        """
            Answering function of vi-MRC model
        """
        QA_input = {
            "question": question,
            "context": context
        }
        tokenizer = model["tokenizer"]
        device = model["device"]

        inputs = [tokenize_function_2(QA_input, tokenizer)]
        inputs_ids = data_collator_2device(inputs, tokenizer, device)

        outputs = model["model"](**inputs_ids)
        answer = extract_answer(inputs, outputs, tokenizer)

        return answer[0]['answer']

    def phoBERT_qa(model: Dict, question: str, context: str):
        _question = [question]
        _context = [context]

        tokenizer = model["tokenizer"]
        device = model["device"]

        encodings = tokenizer(_context, _question, 
                              truncation=True, padding=True)

        with torch.no_grad():
            input_ids = torch.tensor(encodings['input_ids']).to(device)
            attention_mask = torch.tensor(encodings['attention_mask']).to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            start_idx = torch.argmax(outputs['start_logits']).item()
            end_idx = torch.argmax(outputs['end_logits']).item()

            answer = context[start_idx:end_idx]

        return answer
        
    if CONFIG['model']['name'] == "vimrc":
        return viMRC_qa(model, question, context)
    
    else:
        return phoBERT_qa(model, question, context)