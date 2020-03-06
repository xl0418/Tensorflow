user_dic = ['三味','牛肉丸','肉质','Q弹','清甜','爽口','请','各位','慢用',',','。']
sentence = "三味牛肉丸,肉质Q弹,清甜爽口,请各位慢用。"
import numpy as np

dict_add = 'c:/Liang/jieba-master/extra_dict/dict.txt.big'
user_dic = np.loadtxt(dict_add, encoding="utf8")
list_of_lists = []
with open(dict_add, 'r', encoding="utf8") as f:
    for line in f:
        inner_list = [elt.strip() for elt in line.split(' ')]
        # in alternative, if you need to use the file content as numbers
        # inner_list = [int(elt.strip()) for elt in line.split(',')]
        list_of_lists.append(inner_list[0])

user_dic = list_of_lists
class mmsegLiang():
    def __init__(self):
        self.i = 0
        self.max_size = 22
        self.fenci = []

    def simpleseg(self,sentence,user_dic):
        while self.i < len(sentence):
            match = True
            match_character = sentence[self.i]
            while match == True:
                if match_character in user_dic:
                    self.fenci.append(match_character)
                    match = False
                    self.i += 1
                else:
                    if self.i < self.max_size:
                        self.i += 1
                        match_character = match_character+sentence[self.i]
                    else:
                        match = False
        return self.fenci

testmm = mmsegLiang()
test_fenci = testmm.simpleseg(sentence,user_dic)

