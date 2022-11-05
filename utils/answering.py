from models.nguyenvulebinh_qa.infer import data_collator, extract_answer, tokenize_function_2
from models.nguyenvulebinh_qa.model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer
import nltk
nltk.download('punkt')



####
from models.mailong_qa.reader import Reader
reader = Reader()

def mailong_qa(question, context):
    answers, score = reader.getPredictions(question, [context])[0]
    return answers



#####
model_checkpoint = "nguyenvulebinh/vi-mrc-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = MRCQuestionAnswering.from_pretrained(model_checkpoint)

def nguyenvulebinh_qa(question, context):
  QA_input = {
      "question": question,
      "context": context
  }
  
  inputs = [tokenize_function_2(QA_input, tokenizer)]
  inputs_ids = data_collator(inputs, tokenizer)
  outputs = model(**inputs_ids)
  answer = extract_answer(inputs, outputs, tokenizer)

  return answer[0]['answer']