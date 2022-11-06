
from utils.answering import nguyenvulebinh_qa
import time


question = "Người giàu nhất Việt Nam"

relevant_doc = " Ông Phạm Nhật Vượng tiếp tục được công nhận là người giàu nhất Việt Nam với tài sản định giá 6,6 tỷ USD, đứng thứ 239 thế giới, tăng 2,3 tỷ USD so với năm ngoái."

start = time.time()
answer = nguyenvulebinh_qa(question, relevant_doc)
finish = time.time()- start

print(answer)
print("Total time: {}s".format(finish))