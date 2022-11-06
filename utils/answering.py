from models.nguyenvulebinh_qa.infer import data_collator_2device, extract_answer, tokenize_function_2

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
  return answer[0]['answer']



