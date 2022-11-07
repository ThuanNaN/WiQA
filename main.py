import pickle
import json
import argparse

from utils.retrieval import context_retrieval
from utils.answering import nguyenvulebinh_qa
from tqdm import tqdm

from utils import load_model, load_config, get_title
from pyserini.search.lucene import LuceneSearcher

def main():
    parser = argparse.ArgumentParser(description='Some arguments for submission file')

    parser.add_argument('--test_path', type=str, default='./dataset/zac2022_testa_only_question.json',
                        help='The test dataset json file')
    # parser.add_argument('--corpus_pkl_path', type=str, default='./dataset/corpus.pkl',
    #                     help='Path to created corpus')
    # parser.add_argument('--bm25_pkl_path', type=str, default='./dataset/bm25.pkl',
    #                     help='Path to created bm25')
    parser.add_argument('--title_ids', type=str, default='./dataset/titles_id.pkl',
                        help='Path to created bm25')
    parser.add_argument('--submit_filename', type=str, default='submission.json',
                        help='Filename of final submission json file')
    parser.add_argument('--lucene_index', type=str, default='./indexes/test_subprocess',
                        help='Lucene index directory')
    
    args = parser.parse_args()

    # with open(args.corpus_pkl_path, 'rb') as f:
    #     dataset = pickle.load(f)

    # corpus = [
    #     record['text'] for record in dataset
    # ]

    # with open(args.bm25_pkl_path, 'rb') as f:
    #     bm25 = pickle.load(f)


    searcher = LuceneSearcher(args.lucene_index)
    searcher.set_language('vi')

    with open(args.title_ids, 'rb') as f:
        titles_list = pickle.load(f)

    public_test_path = args.test_path
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

            answer = nguyenvulebinh_qa(question, relevant_doc)

            record['candidate_answers'].append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "answer": answer
                }
            )

        records.append(record)

    submission_dict = {
        'data': records
    }

    submission_filename = args.submit_filename
    with open(submission_filename, "w+", encoding="utf8") as f:
        json.dump(submission_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':

    main()

