import re
import pickle

from rank_bm25 import BM25Okapi
from utils import clean_text


def train_rank_bm25(corpus):
    tokenized_corpus = [clean_text(doc).split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


def rank_bm25_context_retrieval(question, corpus, bm25, top_k=1):
    query = clean_text(question)
    tokenized_query = query.split(" ")
    rank_lst = bm25.get_top_n(tokenized_query, corpus, n=top_k)

    return rank_lst

def context_retrieval(question, lucene_searcher, top_k=1) -> list:
    assert top_k > 0 and top_k < 11, ValueError()

    hits = lucene_searcher.search(question)
    if top_k == 1:
        d_id = hits[0].docid
        doc_content = lucene_searcher.doc(d_id).contents()
        return [{
            "id": d_id,
            "text": doc_content
        }]
    else:
        rs = []
        for d in hits[:top_k]:
            d_id = d.docid
            doc_content = lucene_searcher.doc(d_id).contents()
            obj = {
                "id": d_id,
                "text": doc_content
            }
            rs.append(obj)
        return rs


# def main():
#     corpus_pkl_path = './dataset/corpus.pkl'
#     save_path = './dataset/bm25.pkl'
#     with open(corpus_pkl_path, 'rb') as f:
#         dataset = pickle.load(f)

#     corpus = [
#         record['text'] for record in dataset
#     ]

#     bm25 = train_context_retrieval(corpus)

#     with open(save_path, 'wb') as f:
#         pickle.dump(bm25, f)


# if __name__ == '__main__':
#     main()
