import json
import re
import os
from tqdm import tqdm
from utils import CONFIG

from pyserini.search.lucene import LuceneSearcher

try:
    from thefuzz import fuzz
    from thefuzz import process
except:
    from pip._internal import main as pip
    pip(['install', '-q', 'thefuzz'])
    from thefuzz import fuzz
    from thefuzz import process

searcher = LuceneSearcher("./indexes/viWiki_title")
searcher.set_language("vi")


def is_date(answer):
    if re.match(r"ngày.[0-9]+.tháng.[0-9]+.năm.[0-9]*", answer):
        return True
    elif re.match(r"tháng.[0-9]+.năm.[0-9]*", answer):
        return True
    elif re.match(r"năm.[0-9]*", answer):
        return True
    else:
        return False

def matching_title(ans, threshold=43.0):
    hits_list = []
    hits = searcher.search(ans)

    for hit in hits:
        t_id = hit.docid
        t_score = hit.score

        content = searcher.doc(t_id).contents()
        matching_score = fuzz.partial_ratio(ans, content)

        norm_score =  (t_score + matching_score) / 2
        hits_list.append((content, t_score, matching_score, norm_score))

    sorted_hits_list = sorted(hits_list, key=lambda t: t[3], reverse=True)
    if len(sorted_hits_list) > 0:
        best_hit = sorted_hits_list[0]
        if best_hit[3] > threshold:
            return best_hit[0]
        else:
            return None
    else:
        return None

def main():
    with open(CONFIG['raw_candidate_answers'], encoding='utf8') as f:
        raw_answers = json.load(f)

    raw_answers = raw_answers['data']

    complete_answer = []

    for raw_answer in tqdm(raw_answers):
        obj = {}
        obj['id'] = raw_answer['id']
        obj['question'] = raw_answer['question']

        for candidate_answer in raw_answer['candidate_answers']:
            cand_ans = candidate_answer['answer']
            if cand_ans != "":
                if is_date(cand_ans):
                    obj['answer'] = cand_ans.lower()
                    break

                elif cand_ans.isdigit():
                    if len(cand_ans) > 2:
                        obj['answer'] = f"năm {cand_ans}".lower()
                    else:
                        obj['answer'] = cand_ans.lower()
                    break
                else:
                    title = matching_title(cand_ans)
                    if title is not None and len(title.split()) > 0:
                        url = "_".join([tk for tk in title.split()])
                        obj['answer'] = f"wiki/{url}"
                    else:
                        obj['answer'] = title
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