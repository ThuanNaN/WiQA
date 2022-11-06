from models.nguyenvulebinh_qa.infer import data_collator_2device, extract_answer, tokenize_function_2, data_collator
from models.nguyenvulebinh_qa.model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer
import nltk

import time
nltk.download('punkt')

device = 'cuda'

#### mailong model
from models.mailong_qa.reader import Reader
reader = Reader()

def mailong_qa(question, context):  
    answers, score = reader.getPredictions(question, [context])[0]
    return answers

#####
model_checkpoint = "nguyenvulebinh/vi-mrc-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = MRCQuestionAnswering.from_pretrained(model_checkpoint)
model.to(device)

def nguyenvulebinh_qa(question, context):

  QA_input = {
      "question": question,
      "context": context
  }


  inputs = [tokenize_function_2(QA_input, tokenizer)]

  #inputs_ids = data_collator(inputs, tokenizer)

  inputs_ids = data_collator_2device(inputs, tokenizer, device)
  
  outputs = model(**inputs_ids)

  answer = extract_answer(inputs, outputs, tokenizer)
  return answer[0]['answer']