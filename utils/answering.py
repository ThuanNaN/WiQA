import torch
from utils import CONFIG
from models.QA.viMRC import data_collator_2device, extract_answer, tokenize_function_2
import nltk

from typing import Dict

from qa_models.nguyenvulebinh_qa.infer import data_collator_2device, extract_answer, tokenize_function_2, tokenize_function
import nltk
import numpy as np
nltk.download('punkt')

#---------------------------
#### mailong model

def mailong_qa(model, question, context):  
    answers, score = model["model"].getPredictions(question, [context])[0]
    return answers
#---------------------------


#---------------------------
##### nguyenvulebinh model

def nguyenvulebinh_qa(model, question, context):


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

  return answer[0]["answer"]

def nguyenvulebinh_qa_v2(model, question, context):

    lst_context = context.split(".")
    answers = []

    for i in range(0, int(len(lst_context)/2)):
        QA_input = {
            "question": question,
            "context": lst_context[i*2] + lst_context[i*2+1]
        }
        tokenizer = model["tokenizer"]
        device = model["device"]

        inputs = [tokenize_function_2(QA_input, tokenizer)]
        inputs_ids = data_collator_2device(inputs, tokenizer, device)
        outputs = model["model"](**inputs_ids)

        answer = extract_answer(inputs, outputs, tokenizer)
        answers.append(answer[0]['answer'])

    answer_counting = np.unique(answers, return_counts=True)
    return answer_counting[0][np.argmax(answer_counting[1])]

nltk.download('punkt')

def answer(model: Dict, question: str, context: str):
    def viMRC_qa():
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

    def phoBERT_qa():
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
        return viMRC_qa()
    
    else:
        return phoBERT_qa()
