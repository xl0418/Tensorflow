class mmsegLiang:
    def __init__(self):

    def divison_index_by_dic(self,sentence,user_dic):

        index_range = {}
        max_size = 100
        for index, word in enumerate(sentence):
            for j in range(index, len(sentence)):
                cur_word = u"".join(sentence[index:j + 1])
                if cur_word in user_dic:
                    if index in index_range:
                        index_range[index].append(j)
                    else:
                        index_range[index] = [j]
                else:
                    if (j - index) >= max_size: break
        return index_range

    def get_chunks(self,DAG,sentence):
        chunks = []
        chunk = []
        text_length = len(sentence)
        begin_index = 0
        if begin_index > text_length:
            raise Exception("begin index out of sentcen length!!!")
        for i in DAG.get(begin_index, [begin_index]):  # 有的话就是词词典中的匹配，否则就是单字本身
            if (i + 1) > text_length - 1:  # 到了sentence的结尾,后面无可用字或词
                chunks.append([(begin_index, i)])
                break
            for j in DAG.get(i + 1, [i + 1]):
                if (j + 1) > text_length - 1:  # 到了sentence的结尾,后面无可用字或词
                    chunks.append([(begin_index, i), (i + 1, j)])
                    break
                for k in DAG.get(j + 1, [j + 1]):
                    chunk.append((begin_index, i))
                    chunk.append((i + 1, j))
                    chunk.append((j + 1, k))
                    chunks.append(chunk)
                    chunk = []
        chunks = self.chunksFilter(chunks,sentence)
        if self.show_chunks_flag:
            self.show_chunks(sentence,chunks)
        return chunks

    def
