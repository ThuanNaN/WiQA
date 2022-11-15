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

    lst_context = context.split(".")
    answers = []

    for i in range(0, int(len(lst_context)/2)):
        QA_input = {
            "question": question,
            "context": lst_context[i*2] + lst_context[i*2+1]
        }
        tokenizer = model["tokenizer"]
        device = model["device"]

        inputs = [tokenize_function(QA_input, tokenizer)]
        inputs_ids = data_collator_2device(inputs, tokenizer, device)
        outputs = model["model"](**inputs_ids)

        answer = extract_answer(inputs, outputs, tokenizer)
        answers.append(answer[0]['answer'])

    answer_counting = np.unique(answers, return_counts=True)
    return answer_counting[0][np.argmax(answer_counting[1])]


