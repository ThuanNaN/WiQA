import os
import pickle
import json
import argparse


from utils import rank_bm25_context_retrieval, train_rank_bm25, context_retrieval
from utils import answer
from utils import CONFIG, load_model, get_title

from tqdm import tqdm

from pyserini.search.lucene import LuceneSearcher

def main():
    searcher = LuceneSearcher(CONFIG['lucene']['index'])
    searcher.set_language(CONFIG['lucene']['language'])

    model = load_model(CONFIG)

    with open(CONFIG['title_id_file'], 'rb') as f:
        titles_list = pickle.load(f)

    public_test_path = CONFIG['question_path']
    records = []

    with open(public_test_path, "r", encoding="utf8") as f:
        data = f.read()
        public_test_dict = json.loads(data)
        public_test_samples = public_test_dict['data']

    for test_sample in tqdm(public_test_samples):
        id = test_sample["id"]
        question = test_sample["question"]
        # relevant_doc = context_retrieval(question, corpus, bm25)

        docs = context_retrieval(question, searcher, top_k=5)

        record = {
            "id": id,
            "question": question,
            "candidate_answers": []
        }

        for doc in docs:
            doc_id = doc['id']
            title = get_title(doc_id, titles_list)
            relevant_doc = doc['text']

            sentences = relevant_doc.split('.')
            sub_bm25 = train_rank_bm25(sentences)
            relevant_sentences = rank_bm25_context_retrieval(question, sentences, sub_bm25, top_k=10)

            for rel_sent in relevant_sentences:
                _answer = model.answer()
                if _answer is not None:
                    record['candidate_answers'].append(
                        {
                            "doc_id": doc_id,
                            "title": title,
                            "answer": _answer
                        }
                    )
                    break

        records.append(record)

    submission_dict = {
        'data': records
    }

    os.makedirs("./outputs", exist_ok=True)
    with open(CONFIG['raw_candidate_answers'], "w+", encoding="utf8") as f:
        json.dump(submission_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()

