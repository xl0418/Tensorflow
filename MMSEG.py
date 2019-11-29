#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import unicode_literals
import json
import math
import os, sys


import re




class Mmseg(object):

    def __init__(self):
        self.MIN_FLOAT = -3.14e100
        self.dic_file = "c:/Liang/jieba-master/extra_dict/dict.txt.big"
        list_of_lists = {}
        with open(self.dic_file, 'r', encoding="utf8") as f:
            for index, words in enumerate(f):
                inner_list = [elt.strip() for elt in words.split(' ')]
                # in alternative, if you need to use the file content as numbers
                # inner_list = [int(elt.strip()) for elt in line.split(',')]
                list_of_lists[inner_list[0]] = inner_list[1]
        self.user_dic = list_of_lists
        self.maxlenth = 100  # 最大匹配长度
        self.re_eng = re.compile('[a-zA-Z0-9]', re.U)  # 英文数字的正则
        self.re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)", re.U)  # 汉字的正则
        self.re_skip_default = re.compile("(\r\n|\s)", re.U)  # 符号的正则
        self.re_han_cut_all = re.compile("([\u4E00-\u9FD5]+)", re.U)
        self.re_skip_cut_all = re.compile("[^a-zA-Z0-9+#\n]", re.U)
        self.show_chunks_flag=False

    def get_DAG(self,sentence):
        """
        构建有向无环图,使用词典中的词进行最大匹配
        :param sentence:
        :return:
        """
        dag = {}
        for index, word in enumerate(sentence):
            for j in range(index, len(sentence)):
                cur_word = u"".join(sentence[index:j + 1])
                if cur_word in self.user_dic:
                    if index in dag:
                        dag[index].append(j)
                    else:
                        dag[index] = [j]
                else:
                    #是否break决定着最大或者最小切分
                    # break
                    if (j - index) >= self.maxlenth: break
        return dag


    def maxLengthFilter(self,chunks):
        """
        规则1：取最大长度匹配的chunk,即chunk中各词的长度之和取最大的chunk
        :param chunks:
        :return:
        """
        chunks_dict = {}
        for chunk in chunks:
            words_length = sum([(end - begin + 1) for begin, end in chunk])  # 词的长度
            if words_length in chunks_dict:
                chunks_dict[words_length].append(chunk)
            else:
                chunks_dict[words_length] = [chunk]
        chunks_dict_sorted = sorted(chunks_dict.items(), key=lambda x: x[0], reverse=True)
        filted_chunks = chunks_dict_sorted[0][1]  # 取最长的chunks
        return filted_chunks


    def averageMaxLengthFilter(self,chunks):
        """
        规则2：取平均长度最大的,即chunk中(词的长度/chunk的切分个数)
        :param chunks:
        :return:
        """
        chunks_dict = {}
        for chunk in chunks:
            chunk_size = float(len(chunk))  # chunk的切分个数
            words_length = float(sum([(end - begin + 1) for begin, end in chunk]))
            average_length = float(words_length / chunk_size)
            if average_length in chunks_dict:
                chunks_dict[average_length].append(chunk)
            else:
                chunks_dict[average_length] = [chunk]
        chunks_dict_sorted = sorted(chunks_dict.items(), key=lambda x: x[0], reverse=True)
        filted_chunks = chunks_dict_sorted[0][1]  # 取最长的chunks
        return filted_chunks


    def standardDeviationFilter(self,chunks):
        """
        规则3:取标准差平方最小的
        标准差平方:[(x1-x)^2+(x2-x)^2 + ...+(xn-x)^2]/n ,其中x为平均值
        :param chunks:
        :return:
        """
        chunks_dict = {}
        for chunk in chunks:
            chunk_size = float(len(chunk))  # chunk的切分个数
            average_words_size = float(sum([(end - begin + 1) for begin, end in chunk]) / chunk_size)
            chunk_sdq = float(0)  # chunk的标准差平方
            for (begin, end) in chunk:
                cur_size = float(end - begin + 1)
                chunk_sdq += float((cur_size - average_words_size) ** 2)
            chunk_sdq = float(chunk_sdq / chunk_size)
            if chunk_sdq in chunks_dict:
                chunks_dict[chunk_sdq].append(chunk)
            else:
                chunks_dict[chunk_sdq] = [chunk]
        chunks_dict_sorted = sorted(chunks_dict.items(), key=lambda x: x[0], reverse=False)
        filted_chunks = chunks_dict_sorted[0][1]  # 取最长的chunks
        return filted_chunks


    def logFreqFilterSingle(self,chunks,sentence):
        """
        规则4:取单字自由度之和最大的chunk
        自由度：log(frequency)，即词频取自然对数
        :param chunks:
        :return:
        """
        chunks_dict = {}
        for chunk in chunks:
            log_freq = 0
            for (begin,end) in chunk:
                if begin==end:
                    log_freq += math.log(float(self.user_dic.get(sentence[begin:end+1],1)))
            if log_freq in chunks_dict:
                chunks_dict[log_freq].append(chunk)
            else:
                chunks_dict[log_freq] = [chunk]
        chunks_dict_sorted = sorted(chunks_dict.items(), key=lambda x: x[0], reverse=True)
        filted_chunks = chunks_dict_sorted[0][1]  # 取最长的chunks
        return filted_chunks


    def logFreqFilterMutiWord(self, chunks, sentence):
        """
        规则5:取非单字的词语自由度之和最大的chunk
        自由度：log(frequency)，即词频取自然对数
        :param chunks:
        :return:
        """
        chunks_dict = {}
        for chunk in chunks:
            log_freq = 0
            for (begin, end) in chunk:
                if begin != end:
                    log_freq += math.log(float(self.user_dic.get(sentence[begin:end + 1], 1)))
            if log_freq in chunks_dict:
                chunks_dict[log_freq].append(chunk)
            else:
                chunks_dict[log_freq] = [chunk]
        chunks_dict_sorted = sorted(chunks_dict.items(), key=lambda x: x[0], reverse=True)
        filted_chunks = chunks_dict_sorted[0][1]  # 取最长的chunks
        return filted_chunks


    def chunksFilter(self,chunks,sentence):
        """
        过滤规则
        :param chunks:
        :param sentence:
        :return:
        """
        if len(chunks) == 1: return chunks
        # 1、取words长度之后最大的chunks
        new_chunks = self.maxLengthFilter(chunks)
        if len(new_chunks) == 1: return new_chunks

        # 2、取平均word长度最大的chunks
        new_chunks = self.averageMaxLengthFilter(new_chunks)
        if len(new_chunks) == 1: return new_chunks

        # 3、取平均方差最小的chunks
        new_chunks = self.standardDeviationFilter(new_chunks)
        if len(new_chunks) == 1: return new_chunks

        # 4、取单字自由度组合最大的chunk
        new_chunks = self.logFreqFilterSingle(new_chunks,sentence)
        if len(new_chunks) == 1:return new_chunks

        # 5、取非单字的所有词的组合自由语素之和最大的
        new_chunks = self.logFreqFilterMutiWord(new_chunks, sentence)

        if len(new_chunks) != 1:
            print(sentence,"=====")
            raise Exception("chunk划分最终的结果不唯一")
        return new_chunks


    def get_chunks(self,DAG, begin_index, text_length,sentence):
        """
        给出所有的词和开始的位置,拿到所有的chunk
        :param DAG:一句话的在字典中的最大匹配的位置
        :param begin_index:从sentence中的某个起始位置开始构建chunk
        :param text_length:文本的最大长度
        :return:返回该位置生成的三个chunk组成chunks的所有组合
        """
        chunks = []
        chunk = []
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


    def cut_DAG(self,sentence):
        """
        对于划分后的句子，进行切分然后分词
        :param sentence:
        :return:
        """
        self.sentence = sentence
        text_length = len(sentence)
        DAG = self.get_DAG(sentence)
        index = 0
        buff = ""
        while True:
            if index>text_length-1:break
            chunks = self.get_chunks(DAG,index,text_length,sentence)
            begin,end = chunks[0][0]
            word = "".join(sentence[begin:end+1])
            if self.re_eng.match(word) and len(word)==1: #对英文和数字进行处理
                buff += word
            else:
                if buff:
                    yield buff
                    buff = ""

                yield word
            index = end+1
        if buff: #末尾为数字或英文
            yield buff
            buff = ""

    def cut_re(self,sentence,cut_all = False):
        """
        文本的分词切分
        :param sentence:输入进来的文本
        :param cut_all: 全切分(暂不支持)
        :return:返回切分的词
        """
        if cut_all:
            re_han = self.re_han_cut_all
            re_skip = self.re_skip_cut_all
        else:
            re_han = self.re_han_default
            re_skip = self.re_skip_default
        blocks = self.re_han_default.split(sentence)
        for blk in blocks: #按照符号划分然后在针对划分后的分词
            if not blk:
                continue
            if re_han.match(blk):
                for word in self.cut_DAG(blk):
                    yield word
            else:
                tmp = re_skip.split(blk)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        for xx in x:
                            yield xx
                    else:
                        yield x

    def cut(self,sentence):

        sentence = self.strdecode(sentence)
        for word in self.cut_DAG(sentence):
            yield word

    def show_chunks(self,sentence, chunks,word=None):
        """
        打印使用，调试手段
        :param sentence:
        :param chunks:
        :return:
        """

        for chunk in chunks:
            chunk_words = []
            for (begin, end) in chunk:
                word = "".join(sentence[begin:end + 1])
                chunk_words.append(word)
            print("\t".join(chunk_words))
            if word!=None:print(word)
        print("=======================")

    def strdecode(self,sentence,text_type=str):
        if not isinstance(sentence, text_type):
            try:
                sentence = sentence.decode('utf-8')
            except UnicodeDecodeError:
                sentence = sentence.decode('gbk', 'ignore')
        return sentence


if __name__ == "__main__":
    testmmseg = Mmseg()
    sen = '三味牛肉丸,肉质Q弹,清甜爽口,请各位慢用。'
    list(testmmseg.cut(sen))
    print("/".join(list(testmmseg.cut(sen))))
