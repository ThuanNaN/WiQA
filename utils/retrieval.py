import re

from rank_bm25 import BM25Okapi

def clean_text(text):
  text = text.lower()
  text = re.sub('BULLET::::- ', '', text)
  text = re.sub('\w*\d\w*', '', text)
  text = re.sub('\n', ' ',text)
  text = re.sub('\t', ' ',text)
  text = re.sub(r'http\S+', '', text)
  #text = re.sub('[^a-z]',' ',text)
  return text

def train_context_retrieval(corpus):
  tokenized_corpus = [clean_text(doc).split(" ") for doc in corpus]
  bm25 = BM25Okapi(tokenized_corpus)

  return bm25

def context_retrieval(question, corpus, bm25, top_k=1):
  query = clean_text(question)
  tokenized_query = query.split(" ")
  rank_lst = bm25.get_top_n(tokenized_query, corpus, n=top_k)

  return rank_lst[0]