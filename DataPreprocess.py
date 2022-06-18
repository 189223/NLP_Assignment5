#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author Xu
# @date 2022/6/17
# @file DataPreprocess.py

import re
import torch
from vocabulary import Vocab
from torch.utils.data import Dataset

data_path = "./data/天龙八部.txt"
test_path="./data/verify.txt"
rabbish=['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '『', '』', '（', '）', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b']

def getCorpus():
    with open(data_path, "r", encoding="GB18030") as fp:
        text = "".join(fp.readlines())
    for i in rabbish:
        text=text.replace(i,'')
    for i in ["\t","\n","\u3000","\u0020","\u00A0"," "]:
        text = text.replace(i,"")
    for i in ["？","！","……"]:
        text_raw = text.replace(i,"。")
    corpus1 = text_raw.split("。")
    with open(test_path, "r", encoding="utf-8") as fp:
        text = "".join(fp.readlines())
    for i in rabbish:
        text=text.replace(i,'')
    for i in ["\t","\n","\u3000","\u0020","\u00A0"," "]:
        text = text.replace(i,"")
    for i in ["？","！","……"]:
        text_raw = text.replace(i,"。")
    corpus2 = text_raw.split("。")
    return corpus1,corpus2

def build(corpus_train,corpus_test):
    vocab = Vocab()
    vocab.count_file(corpus_train)
    vocab.build_vocab()
    vocab.encode_file(corpus_train,'corpus_train')
    vocab.encode_file(corpus_test,'corpus_test')
    pass

class LoadData(Dataset):
    def __init__(self,corpus_encoding):
        self.corpus_temp= []
        self.max_len = 132
        for corpus_onehot in corpus_encoding:
            if corpus_onehot.strip() == "":
                continue
            data = [int(i) for i in corpus_onehot.strip().split(",")]
            if len(data)>132:
                continue
            self.corpus_temp.append(data)

    def __getitem__(self,index):
        inputs = self.corpus_temp[index]
        targets = self.corpus_temp[index+1]
        # "1": "<PAD>",
        # "2": "<BOS>",
        # "3": "<EOS>",
        inputs = inputs + [2] * (self.max_len-len(inputs))
        targets =targets + [2]*(self.max_len-len(targets))

        inputs = torch.tensor(inputs)
        targets = torch.tensor(targets)
        return (inputs, targets)

    def __len__(self):
        return len(self.corpus_temp) - 1

if __name__ == '__main__':
    corpus_train,corpus_test= getCorpus()
    build(corpus_train,corpus_test)

    # with open('./data/corpus_encode.txt', "r", encoding="utf-8") as fp:
    #     textL = fp.readlines()
    #
    # data=LoadData(textL)
    # data.__getitem__(1)
