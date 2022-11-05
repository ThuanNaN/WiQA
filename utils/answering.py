from models.nguyenvulebinh_qa.infer import tokenize_function, data_collator, extract_answer
from models.nguyenvulebinh_qa.model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer

import nltk
nltk.download('punkt')

model_checkpoint = "nguyenvulebinh/vi-mrc-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = MRCQuestionAnswering.from_pretrained(model_checkpoint)

def QA(question, context):
  QA_input = {
      "question": question,
      "context": context
  }
  
  inputs = [tokenize_function(QA_input, tokenizer)]
  inputs_ids = data_collator(inputs, tokenizer)
  outputs = model(**inputs_ids)
  answer = extract_answer(inputs, outputs, tokenizer)

  return answer[0]['answer']