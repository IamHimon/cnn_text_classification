# -*- coding: utf-8 -*-：
import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    :param string:
    :return:
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_labels():
    """
    loads MR polarity data from files,splits the data into words and generates labels.
    returns split sentences and labels.
    :return:list[[ps1,ps2,ps3,,,,,ns1,ns2,ns3,,,],[pl1,pl2,pl3,,,,,nl1,nl2,nl3,,,]]
            ps1：splited positive sentences list,[w1,w2,w3,,,]
            pl1:positive label([0,1])
            ns1：splited negative sentences list,[w1,w2,w3,,,]
            nl1:negative label([1,0])
    """
    # Load data from files, convert content per line to  list[string]
    positive_examples = list(open("/home/himon/Jobs/corpus/rt-polaritydata/rt-polarity.pos", "r", encoding="ISO-8859-1")
                             .readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("/home/himon/Jobs/corpus/rt-polaritydata/rt-polarity.neg", "r", encoding="ISO-8859-1")
                             .readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels, all element is [0,1] label positive case or [1,0] label negative case.
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)  # join a sequence of arrays
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有yield说明不是一个普通函数，是一个Generator.
    函数效果：对data，一共分成num_epochs个阶段（epoch），在每个epoch内，如果shuffle=True，就将data重新洗牌，
    批量生成(yield)一批一批的重洗过的data，每批大小是batch_size，一共生成int(len(data)/batch_size)+1批。
    Generate a  batch iterator for a dataset.
    :param data:
    :param batch_size:每批data的size
    :param num_epochs:阶段数目
    :param shuffle:洗牌
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batch_per_epoch = int(len(data)/batch_size) + 1  # 每段的batch数目
    for epoch in range(num_epochs):
        if shuffle:
            # np.random.permutation(),得到一个重新排列的序列(Array)
            # np.arrange(),得到一个均匀间隔的array.
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]    # 重新洗牌的data
        else:
            shuffle_data = data
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]   # all elements index between start_index and end_index

