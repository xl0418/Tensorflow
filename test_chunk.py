import json
dict_add = 'c:/Liang/jieba-master/extra_dict/dict.txt.big'
list_of_lists = {}
with open(dict_add, 'r',encoding="utf8") as f:
    for index,words in enumerate(f):
        inner_list = [elt.strip() for elt in words.split(' ')]
        # in alternative, if you need to use the file content as numbers
        # inner_list = [int(elt.strip()) for elt in line.split(',')]
        list_of_lists[inner_list[0]]=inner_list[1]

user_dic = list_of_lists

fenci = []
sentence = "三味牛肉丸,肉质Q弹,清甜爽口,请各位慢用。"

dag = {}
max_size = 100
for index, word in enumerate(sentence):
    for j in range(index, len(sentence)):
        cur_word = u"".join(sentence[index:j + 1])
        if cur_word in user_dic:
            if index in dag:
                dag[index].append(j)
            else:
                dag[index] = [j]
        else:
            #是否break决定着最大或者最小切分
            # break
            if (j - index) >= max_size: break



chunks = []
chunk = []
begin_index = 0
text_length = len(sentence)
if begin_index > text_length:
    raise Exception("begin index out of sentcen length!!!")
for i in dag.get(begin_index, [begin_index]):  # 有的话就是词词典中的匹配，否则就是单字本身
    if (i + 1) > text_length - 1:  # 到了sentence的结尾,后面无可用字或词
        chunks.append([(begin_index, i)])
        break
    for j in dag.get(i + 1, [i + 1]):
        if (j + 1) > text_length - 1:  # 到了sentence的结尾,后面无可用字或词
            chunks.append([(begin_index, i), (i + 1, j)])
            break
        for k in dag.get(j + 1, [j + 1]):
            chunk.append((begin_index, i))
            chunk.append((i + 1, j))
            chunk.append((j + 1, k))
            chunks.append(chunk)
            chunk = []
