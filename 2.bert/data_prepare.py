from bert import *


# 数据预处理
def make_batch():
    batch = [] # list
    positive = negative = 0  # 计数器 为了记录NSP任务中的正样本和负样本的个数，比例最好是在一个batch中接近1：1
    while positive != batch_size/2 or negative != batch_size/2:
        #  抽出来两句话 先随机sample两个index 再通过index找出样本
        tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences))  # 比如tokens_a_index=3，tokens_b_index=1；从整个样本中抽取对应的样本；
        tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]## 根据索引获取对应样本：tokens_a=[5, 23, 26, 20, 9, 13, 18] tokens_b=[27, 11, 23, 8, 17, 28, 12, 22, 16, 25]
        # 拼接
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']] ## 加上特殊符号，CLS符号是1，sep符号是2：[1, 5, 23, 26, 20, 9, 13, 18, 2, 27, 11, 23, 8, 17, 28, 12, 22, 16, 25, 2]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)##分割句子符号：[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # MASK LM
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))  # n_pred=3；整个句子的15%的字符可以被mask掉，这里取和max_pred中的最小值，确保每次计算损失的时候没有那么多字符以及信息充足，有15%做控制就够了；其实可以不用加这个，单个句子少了，就要加上足够的训练样本
        # 不让特殊字符参与mask
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word_dict['[CLS]'] and token != word_dict['[SEP]']] ## cand_maked_pos=[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]；整个句子input_ids中可以被mask的符号必须是非cls和sep符号的，要不然没意义
        shuffle(cand_maked_pos)## 打乱顺序：cand_maked_pos=[6, 5, 17, 3, 1, 13, 16, 10, 12, 2, 9, 7, 11, 18, 4, 14, 15]  其实取mask对应的位置有很多方法，这里只是一种使用shuffle的方式
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:  # 取其中的三个；masked_pos=[6, 5, 17] 注意这里对应的是position信息；masked_tokens=[13, 9, 16] 注意这里是被mask的元素之前对应的原始单字数字；
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])  # 回到ppt看一下
            if random() < 0.8:  # 80%
                input_ids[pos] = word_dict['[MASK]']  # make mask
            elif random() < 0.5:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                input_ids[pos] = word_dict[number_dict[index]] # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)##maxlen=30；n_pad=10
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)# 这里有一个问题，0和之前的重了

        # Zero Padding (100% - 15%) tokens 是为了计算一个batch中句子的mlm损失的时候可以组成一个有效矩阵放进去；不然第一个句子预测5个字符，第二句子预测7个字符，第三个句子预测8个字符，组不成一个有效的矩阵；
        ## 这里非常重要，为什么是对masked_tokens是补零，而不是补其他的字符？？？？我补1可不可以？？
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)##  masked_tokens= [13, 9, 16, 0, 0] masked_tokens 对应的是被mask的元素的原始真实标签是啥，也就是groundtruth
            masked_pos.extend([0] * n_pad)## masked_pos= [6, 5, 17，0，0] masked_pos是记录哪些位置被mask了

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch



text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet\n'
    'Thanks you Romeo'
)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word_dict[w] = i + 4
number_dict = {i: w for i, w in enumerate(word_dict)}
vocab_size = len(word_dict)

# 把文本转化成数字
token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)

batch = make_batch()  # 最重要的一部分  预训练任务的数据构建部分
input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))# map把函数依次作用在list中的每一个元素上，得到一个新的list并返回。注意，map不改变原list，而是返回一个新list。
