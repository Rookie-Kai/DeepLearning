import collections
import re


# 读入文本
def read_time_machine():
    with open("D:/cloudFile/timemachine.txt", 'r', encoding='UTF-8') as f:
        lines = [re.sub('[^a-z]+', ' ', line.strip().lower()) for line in f]
        # 将文件每一行的前后缀空白字符去掉，大写字母换为小写，然后把由非小写英文字符构成的子串替换为空格，得到lines列表，列表元素就是文件每一行处理后得到的结果。
    return lines


lines = read_time_machine()
print('# sentences %d' % len(lines))


# 分词
def tokenize(sentences, token='word'):
    if token == 'word':
        return [sentence.split(' ') for sentence in sentences]
    elif token == 'char':
        return [list(sentence) for sentence in sentences]
    else:
        print('Error: Unknow token type '+token)


tokens = tokenize(lines)
print(tokens[0:2])


# 建立字典
class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        # 统计词频
        counter = count_corpus(tokens)  #<key, value>：<词，词频>
        self.token_freqs = list(counter.items())
        # 建立用来维护字典的空列表
        self.idx_to_token = []
        # 如果use_special_tokens为真，建立如下token
        if use_special_tokens:
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            self.idx_to_token += ['<pad>', '<bos>', '<eos>', '<unk>']
        # 如果为假，使用unk这个token
        else:
            self.unk = 0
            self.idx_to_token += ['<unk>']
        # 把满足条件的token添加到idx_to_token中
        # 因为index本就是一个列表，天然满足索引到token的映射，用下标当作索引即可。
        self.idx_to_token += [token for token, freq in self.token_freqs
                              if freq >= min_freq and token not in self.idx_to_token]
        self.token_to_idx = dict()
        # 通过enumerate枚举每个词的下标和词并添加到idx_to_token中
        for idx, token in enumerate(self.idx_to_token):
            self.token_to_idx[token] = idx

    # 返回列表长度
    def __len__(self):
        return len(self.idx_to_token)

    # 定义Vocab类的索引，实现词到索引的映射
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    # 根据给定索引返回对应的词
    def to_tokens(self, indices):
        if not isinstance(tokens, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def count_corpus(sentences):
    # 二维列表sentences转换为一维列表
    tokens = [tk for st in sentences for tk in st]
    # 返回一个字典，记录每个词的出现次数
    return collections.Counter(tokens)


vocab = Vocab(tokens)
# 打印前十个键值对
print(list(vocab.token_to_idx.items())[0:10])


# 从单词序列转换为索引序列
for i in range(8, 10):
    print('words: ', tokens[i])
    print('indices: ', vocab[tokens[i]])
