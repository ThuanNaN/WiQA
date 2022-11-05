from utils.utils import mailong_qa, nguyenvulebinh_qa


def QA(question, context):
  answer = mailong_qa(question, context)
  return answer   