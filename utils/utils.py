from nltk import word_tokenize

def tokenize_function(example, tokenizer):
    question_word = word_tokenize(example["question"])
    context_word = word_tokenize(example["context"])

    question_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in question_word]
    context_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in context_word]
  
    valid = True

    max_len_question = 100
    max_len_context = 400

    question_token_len  = len([j for i in question_sub_words_ids  for j in i])
    if question_token_len > max_len_question:
      question_sub_words_ids = question_sub_words_ids[:max_len_question]
    question_token_len  = len([j for i in question_sub_words_ids  for j in i])
    while question_token_len > max_len_question:
      question_sub_words_ids = question_sub_words_ids[: len(question_sub_words_ids)-1]
      question_token_len  = len([j for i in question_sub_words_ids  for j in i])

    context_token_len = len([j for i in context_sub_words_ids for j in i])
    if context_token_len > max_len_context:
      context_sub_words_ids = context_sub_words_ids[:max_len_context]
    
    context_token_len = len([j for i in context_sub_words_ids for j in i])
    while context_token_len > max_len_context:
      context_sub_words_ids = context_sub_words_ids[: len(context_sub_words_ids) -1 ]
      context_token_len = len([j for i in context_sub_words_ids for j in i])


    question_sub_words_ids = [[tokenizer.bos_token_id]] + question_sub_words_ids + [[tokenizer.eos_token_id]]
    context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]


    input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]
    if len(input_ids) > tokenizer.max_len_single_sentence + 2:
        valid = False

    words_lengths = [len(item) for item in question_sub_words_ids + context_sub_words_ids]

    return {
        "input_ids": input_ids,
        "words_lengths": words_lengths,
        "valid": valid
    }
