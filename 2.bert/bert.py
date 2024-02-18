"""
max_len = 30                        # max_len: 输入序列的最大长度.
max_vocab = 50                      # max_vocab: 字典的最大大小.
max_pred = 5                        # max_pred: Mask时最大的Mask数量.

d_k = d_v = 64                      # d_k, d_v: 自注意力中K和V的维度, Q的维度直接用K的维度代替, 因为这二者必须始终相等.
d_model = 768  # n_heads * d_k      # d_model: Embedding的大小.
d_ff = d_model * 4                  # d_ff: 前馈神经网络的隐藏层大小, 一般是d_model的四倍.

n_heads = 12                        # n_heads: 多头注意力的头数.
n_layers = 6                        # n_layers: Encoder的堆叠层数.
n_segs = 2                          # n_segs: 输入BERT的句子段数. 用于制作Segment Embedding.

p_dropout = 0.1                     # p_dropout: BERT中所有dropout的概率.
# BERT propability defined          # p_mask, p_replace, p_do_nothing:
p_mask = 0.8                        # 80%的概率将被选中的单词替换为[MASK].
p_replace = 0.1                     # 10%的概率将被选中的单词替换为随机词.
p_do_nothing = 1 - p_mask - p_replace # 10%的概率对被选中的单词不进行替换.

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

"""



"""
orginal from ：
https://github.com/graykode/nlp-tutorial/tree/master/5-2.BERT
"""
import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# BERT Parameters
maxlen = 30 # 句子的最大长度
batch_size = 6 # 每一组有多少个句子一起送进去模型
vocab_size = 50
max_pred = 5  # max tokens of prediction
n_layers = 6 # number of Encoder of Encoder Layer
n_heads = 12 # number of heads in Multi-Head Attention
d_model = 768 # Embedding Size
d_ff = 3072  # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2


# 符号矩阵
def get_attn_pad_mask(seq_q, seq_k): # 在自注意力层q k是一致的
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # eq(0)表示和0相等的返回True，不相等返回False。
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # 重复了len_q次  batch_size x len_q x len_k 不懂可以看一下例子


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


#Embedding层
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)
    def forward(self, input_ids, segment_ids):# x对应input_ids, seg对应segment_ids
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(input_ids)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(input_ids) + self.pos_embed(pos) + self.seg_embed(segment_ids)
        return self.norm(embedding)


# 注意力打分函数
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_pad):
        ## 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]  V: [batch_size x n_heads x len_k x d_v]
        ##首先经过matmul函数得到的scores形状是 : [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        ## 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_pad，把被pad的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_pad, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


#多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        ## 输入进来的QKV是相等的，使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_pad):
        ## 这个多头分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        ## 输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        ##下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以这里都是dk
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
        ## 输入进行的attn_pad形状是 batch_size x len_q x len_k，然后经过下面这个代码得到 新的attn_pad : [batch_size x n_heads x len_q x len_k]，就是把pad信息重复了n个头上
        attn_pad = attn_pad.unsqueeze(1).repeat(1, n_heads, 1, 1)    # repeat 对张量重复扩充
        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_pad)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]


#基于位置的前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self): # 对每个字的增强语义向量再做两次线性变换，以增强整个模型的表达能力。
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))

#Encoder
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_pad):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_pad) # enc_inputs to same Q,K,V enc_self_attn_mask是pad符号矩阵
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


## 1. BERT模型整体架构
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding() ## 词向量层，构建词表矩阵
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)]) ## 把N个encoder堆叠起来，具体encoder实现一会看
        self.fc = nn.Linear(d_model, d_model) ## 前馈神经网络-cls
        self.activ1 = nn.Tanh() ## 激活函数-cls
        self.linear = nn.Linear(d_model, d_model)#-mlm
        self.activ2 = gelu ## 激活函数--mlm
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)## cls 这是一个分类层，维度是从d_model到2，对应我们架构图中就是这种：

        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        input = self.embedding(input_ids, segment_ids) # 将input_ids，segment_ids，pos_embed加和

        ##get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响，去看一下这个函数 4.
        enc_self_attn_pad = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(input, enc_self_attn_pad) ## enc_self_attn这里是QK转置相乘之后softmax之后的矩阵值，代表的是每个单词和其他单词相关性；
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model] cls 对应的位置 可以看一下例子
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]  其中一个 masked_pos= [6, 5, 17，0，0]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) #在output取出一维对应masked_pos数据 masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf