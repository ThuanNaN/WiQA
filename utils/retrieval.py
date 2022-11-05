import re
import pickle

from rank_bm25 import BM25Okapi

def clean_text(text):
    text = text.lower()
    text = re.sub('\n',' ', text)
    text = re.sub('\t',' ', text)
    text = re.sub('=*', '', text)
    text = re.sub('(BULLET:*.) | (BULLET:*[0-9].)', '', text)

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

def main():
  corpus_pkl_path = './dataset/corpus.pkl'
  save_path = '../dataset/bm25.pkl'
  with open(corpus_pkl_path, 'rb') as f:
    dataset = pickle.load(f)

  corpus = [
      record['text'] for record in dataset
  ]

  bm25 = train_context_retrieval(corpus)

  with open(save_path, 'wb') as f:
    pickle.dump(bm25, f)

if __name__ == '__main__':
  main()