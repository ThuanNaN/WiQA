import json
import pickle
import argparse

from utils.retrieval import train_context_retrieval

def main():
    parser = argparse.ArgumentParser(description='Some arguments for creating corpus')

    parser.add_argument('--corpus_path', type=str, default='./dataset/corpus_v2.json',
                        help='The corpus json file')
    parser.add_argument('--save_path', type=str, default='./dataset/corpus.pkl',
                        help='Path to created corpus')
    
    args = parser.parse_args()

    with open(args.corpus_path, 'r', encoding='utf8') as f:
        data = f.read()
        dataset = json.loads(data)

    corpus = [
        record['text'] for record in dataset
    ]

    train_context_retrieval(corpus)

    with open(args.save_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()
