
from utils.answering import nguyenvulebinh_qa, mailong_qa
from utils.utils import load_model
import time


question = "Người giàu nhất Việt Nam"

relevant_doc = " Ông Phạm Nhật Vượng tiếp tục được công nhận là người giàu nhất Việt Nam với tài sản định giá 6,6 tỷ USD, đứng thứ 239 thế giới, tăng 2,3 tỷ USD so với năm ngoái."

model_name = "mailong"
device = "cpu"
model= load_model(model_name, device)


start = time.time()
answer = mailong_qa(model, question, relevant_doc)
finish = time.time()- start

print(answer)
print("Total time: {}s".format(finish))