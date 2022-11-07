import json
import os
import requests
from utils import CONFIG

def main():
    with open(CONFIG['raw_candidate_answers'], encoding='utf8') as f:
        data = json.load(f)

    raw_answers = data['data']
    del data

    complete_answer = []

    for raw_answer in raw_answers:
        obj = {}
        obj['id'] = raw_answer['id']
        obj['question'] = raw_answer['question']

        for ans in raw_answer['candidate_answers']:
            if len(ans['answer']) > 1:
                url = "_".join([tk for tk in ans['answer'].split()])
                r = requests.head(f'https://vi.wikipedia.org/wiki/{url}')
                if r.status_code == 200:
                    obj['answer'] = f"wiki/{url}"
                else:
                    obj['answer'] = ans['answer']
                break

            else:
                obj['answer'] = None       

        complete_answer.append(obj)

    submission_dict = {
            'data': complete_answer
        }

    os.makedirs("./outputs", exist_ok=True)
    with open(CONFIG['submission_name'], "w+", encoding="utf8") as f:
        json.dump(submission_dict, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()