from download import download
from pathlib import Path
from tqdm import tqdm
import os
import re
import torch
import numpy as np
from torch import tensor
from torch import nn

from transformer import *

# pip install download nltk -i https://pypi.tuna.tsinghua.edu.cn/simple


url = "https://modelscope.cn/api/v1/datasets/SelinaRR/Multi30K/repo?Revision=master&FilePath=Multi30K.zip"

# download(url, './', kind='zip', replace=True)

datasets_path = './datasets/'
train_path = datasets_path + 'train/'
valid_path = datasets_path + 'valid/'
test_path = datasets_path + 'test/'


def print_data(data_file_path, print_n=5):
    print("=" * 40 + "datasets in {}".format(data_file_path) + "=" * 40)
    with open(data_file_path, 'r', encoding='utf-8') as en_file:
        en = en_file.readlines()[:print_n]
        for index, seq in enumerate(en):
            print(index, seq.replace('\n', ''))


# print_data(train_path + 'train.de')
# print_data(train_path + 'train.en')


class Multi30K():
    """Multi30K数据集加载器，加载Multi30K数据集并处理为一个Python迭代对象"""
    def __init__(self, path):
        self.data = self._load(path)

    def _load(self, path):
        def tokenize(text):
            text = text.rstrip()
            return [tok.lower() for tok in re.findall(r'\w+|[^\w\s]', text)]
        
        def read_data(data_file_path):
            with open(data_file_path, 'r', encoding='utf-8') as data_file:
                data = data_file.readlines()[:-1]
                return [tokenize(i) for i in data]

        members = {i.split('.')[-1]: path + i for i in os.listdir(path)}
        ret = [read_data(members['de']), read_data(members['en'])]
        return list(zip(*ret))
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)


train_dataset, valid_dataset, test_dataset = Multi30K(train_path), Multi30K(valid_path), Multi30K(test_path)

# for de, en in test_dataset:
#     print(f'de = {de}')
#     print(f'en = {en}')
#     break

class Vocab:
    """通过词频字典，构建词典"""
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']

    def __init__(self, word_count_dict, min_freq=1):
        self.word2idx = {}
        for idx, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = idx

        filted_dict = {w: c for w, c in word_count_dict.items() if c >= min_freq}
        for w, _ in filted_dict.items():
            self.word2idx[w] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        self.bos_idx = self.word2idx['<bos>']
        self.eos_idx = self.word2idx['<eos>']
        self.pad_idx = self.word2idx['<pad>']
        self.unk_idx = self.word2idx['<unk>']

    def _word2idx(self, word):
        """单词映射至数字索引"""
        if word not in self.word2idx:
            return self.unk_idx
        return self.word2idx[word]

    def _idx2word(self, idx):
        """数字索引映射至单词"""
        if idx not in self.idx2word:
            raise ValueError('input index is not in vocabulary.')
        return self.idx2word[idx]

    def encode(self, word_or_list):
        """将单个单词或单词数组映射至单个数字索引或数字索引数组"""
        if isinstance(word_or_list, list):
            return [self._word2idx(i) for i in word_or_list]
        return self._word2idx(word_or_list)

    def decode(self, idx_or_list):
        """将单个数字索引或数字索引数组映射至单个单词或单词数组"""
        if isinstance(idx_or_list, list):
            return [self._idx2word(i) for i in idx_or_list]
        return self._idx2word(idx_or_list)

    def __len__(self):
        return len(self.word2idx)


from collections import Counter, OrderedDict

def build_vocab(dataset):
    de_words, en_words = [], []
    for de, en in dataset:
        de_words.extend(de)
        en_words.extend(en)
    
    de_count_dict = OrderedDict(sorted(Counter(de_words).items(), key=lambda t: t[1], reverse=True))
    en_count_dict = OrderedDict(sorted(Counter(en_words).items(), key=lambda t: t[1], reverse=True))

    return Vocab(de_count_dict, min_freq=2), Vocab(en_count_dict, min_freq=2)

de_vocab, en_vocab = build_vocab(train_dataset)
# print('Unique tokens in de vocabulary:{} and en vocabulary:{}\n'.format(len(de_vocab), len(en_vocab)))

# str_seq_en = ['two', 'young', ',', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes', '.']
# print("word:{}\nindex:{}\n".format(str_seq_en, en_vocab.encode(str_seq_en)))

# index = [5, 6, 7, 8, 9, 10]
# print("index:{}\nword:{}".format(index, en_vocab.decode(index)))



class Iterator():
    """创建数据迭代器"""
    def __init__(self, dataset, de_vocab, en_vocab, batch_size, max_len=32, drop_reminder=False):
        self.dataset = dataset
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab

        self.batch_size = batch_size
        self.max_len = max_len
        self.drop_reminder = drop_reminder

        length = len(self.dataset) // batch_size
        self.len = length if drop_reminder else length + 1  # 批量数量

    def __call__(self):
        def pad(idx_list, vocab, max_len):
            """统一序列长度，并记录有效长度"""
            idx_pad_list, idx_len = [], []
            for i in idx_list:
                if len(i) > max_len - 2:
                    idx_pad_list.append([vocab.bos_idx] + i[:max_len-2] + [vocab.eos_idx])
                    idx_len.append(max_len)
                else:
                    idx_pad_list.append([vocab.bos_idx] + i + [vocab.eos_idx] + [vocab.pad_idx] * (max_len - len(i) - 2))
                    idx_len.append(len(i) + 2)
            return idx_pad_list, idx_len

        def sort_by_length(src, trg):
            """根据src的字段长度进行排序"""
            data = zip(src, trg)
            data = sorted(data, key=lambda t: len(t[0]), reverse=True)
            return zip(*list(data))

        def encode_and_pad(batch_data, max_len):
            """将批量中的文本数据转换为数字索引，并统一每个序列的长度"""
            src_data, trg_data = zip(*batch_data)
            src_idx = [self.de_vocab.encode(i) for i in src_data]
            trg_idx = [self.en_vocab.encode(i) for i in trg_data]

            src_idx, trg_idx = sort_by_length(src_idx, trg_idx)
            src_idx_pad, src_len = pad(src_idx, de_vocab, max_len)
            trg_idx_pad, _ = pad(trg_idx, en_vocab, max_len)

            return src_idx_pad, src_len, trg_idx_pad

        for i in range(self.len):
            if i == self.len - 1 and not self.drop_reminder:
                batch_data = self.dataset[i * self.batch_size:]
            else:
                batch_data = self.dataset[i * self.batch_size: (i+1) * self.batch_size]

            src_idx, src_len, trg_idx = encode_and_pad(batch_data, self.max_len)
            yield tensor(src_idx), tensor(src_len), tensor(trg_idx)


    def __len__(self):
        return self.len

train_iterator = Iterator(train_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=True)
valid_iterator = Iterator(valid_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=False)
test_iterator = Iterator(test_dataset, de_vocab, en_vocab, batch_size=1, max_len=32, drop_reminder=False)

# for src_idx, src_len, trg_idx in train_iterator():
#     print(f'src_idx.shape:{src_idx.shape}\n{src_idx}\nsrc_len.shape:{src_len.shape}\n{src_len}\ntrg_idx.shape:{trg_idx.shape}\n{trg_idx}')
#     break

# word_index = tensor([21, 28, 49, 12, 275, 119, 49, 23, 54, 32])
# src_emb = nn.Embedding(len(de_vocab), 4)
# enc_outputs = src_emb(word_index)
# print(enc_outputs)


src_vocab_size = len(de_vocab)
tgt_vocab_size = len(en_vocab)
print("src_vocab_size: ", src_vocab_size)
print("tgt_vocab_size: ", tgt_vocab_size)



