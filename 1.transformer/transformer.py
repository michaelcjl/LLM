from torch import tensor
import torch 
from torch import nn
from torch import optim
from torch.utils import data as Data
import numpy as np

d_model = 512 # embedding size 
max_len = 1024 # max length of sequence
d_ff = 2048 # feedforward nerual network  dimension
d_k = d_v = 64 # dimension of k(same as q) and v
n_layers = 6 # number of encoder and decoder layers
n_heads = 8 # number of heads in multihead attention
p_drop = 0.1 # propability of dropout

# d_model: Embedding的大小.
# max_len: 输入序列的最长大小.
# d_ff: 前馈神经网络的隐藏层大小, 一般是d_model的四倍.
# d_k, d_v: 自注意力中K和V的维度, Q的维度直接用K的维度代替, 因为这二者必须始终相等.
# n_layers: Encoder和Decoder的层数.
# n_heads: 自注意力多头的头数.
# p_drop: Dropout的概率.
# 一般为了平衡计算成本 我们会取 d_v = d_k = d_model/n_heads


if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_attn_pad_mask(seq_q, seq_k):
    '''
    Padding, because of unequal in source_len and target_len.

    parameters:
    seq_q: [batch, seq_len]
    seq_k: [batch, seq_len]

    return:
    mask: [batch, len_q, len_k]

    '''
    batch, len_q = seq_q.size()
    batch, len_k = seq_k.size()
    # we define index of PAD is 0, if tensor equals (zero) PAD tokens
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch, 1, len_k]

    return pad_attn_mask.expand(batch, len_q, len_k) # [batch, len_q, len_k]


def get_attn_subsequent_mask(seq):
  '''
  Build attention mask matrix for decoder when it autoregressing.

  parameters:
  seq: [batch, target_len]

  return:
  subsequent_mask: [batch, target_len, target_len] 
  '''
  attn_shape = [seq.size(0), seq.size(1), seq.size(1)] # [batch, target_len, target_len]
  subsequent_mask = np.triu(np.ones(attn_shape), k=1) # [batch, target_len, target_len] 
  subsequent_mask = torch.from_numpy(subsequent_mask)

  return subsequent_mask # [batch, target_len, target_len] 



class ScaledDotProductAttention(nn.Module):
  def __init__(self):
    super(ScaledDotProductAttention, self).__init__()

  def forward(self, Q, K, V, attn_mask):
    '''
    Q: [batch, n_heads, len_q, d_k]
    K: [batch, n_heads, len_k, d_k]
    V: [batch, n_heads, len_v, d_v]
    attn_mask: [batch, n_heads, seq_len, seq_len]
    '''
    d_k = K.size(3)
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # [batch, n_heads, len_q, len_k]
    if attn_mask is not None:
        scores.masked_fill_(attn_mask, -1e9)

    attn = nn.Softmax(dim=-1)(scores) # [batch, n_heads, len_q, len_k]
    prob = torch.matmul(attn, V) # [batch, n_heads, len_q, d_v]
    return prob, attn


class MultiHeadAttention(nn.Module):

  def __init__(self, d_model=512 , n_heads=8, dropout=0.0):
    super(MultiHeadAttention, self).__init__()
    # do not use more instance to implement multihead attention
    # it can be complete in one matrix
    self.n_heads = n_heads
    d_k = d_v = d_model// n_heads

    # we can't use bias because there is no bias term in formular
    self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
    self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
    self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
    self.fc = nn.Linear(d_v * n_heads, d_model, bias=False)
    # self.layer_norm = nn.LayerNorm(d_model).to(device)
    self.dropout = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, input_Q, input_K, input_V, attn_mask):
    '''
    To make sure multihead attention can be used both in encoder and decoder, 
    we use Q, K, V respectively.
    input_Q: [batch, len_q, d_model]
    input_K: [batch, len_k, d_model]
    input_V: [batch, len_v, d_model]
    '''
    residual, batch = input_Q, input_Q.size(0)

    # [batch, len_q, d_model] -- matmul W_Q --> [batch, len_q, d_q * n_heads] -- view --> 
    # [batch, len_q, n_heads, d_k,] -- transpose --> [batch, n_heads, len_q, d_k]

    Q = self.W_Q(input_Q).view(batch, -1, n_heads, d_k).transpose(1, 2) # [batch, n_heads, len_q, d_k]
    K = self.W_K(input_K).view(batch, -1, n_heads, d_k).transpose(1, 2) # [batch, n_heads, len_k, d_k]
    V = self.W_V(input_V).view(batch, -1, n_heads, d_v).transpose(1, 2) # [batch, n_heads, len_v, d_v]

    attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # [batch, n_heads, seq_len, seq_len]

    # prob: [batch, n_heads, len_q, d_v] attn: [batch, n_heads, len_q, len_k]
    prob, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

    prob = prob.transpose(1, 2).contiguous() # [batch, len_q, n_heads, d_v]
    prob = prob.view(batch, -1, n_heads * d_v).contiguous() # [batch, len_q, n_heads * d_v]

    output = self.fc(prob) # [batch, len_q, d_model]
    output = self.dropout(output)

    return self.layer_norm(residual + output), attn


class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=0.1, max_len=1024):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    positional_encoding = torch.zeros(max_len, d_model) # [max_len, d_model]
    position = torch.arange(0, max_len).float().unsqueeze(1) # [max_len, 1]

    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                         (-torch.log(torch.Tensor([10000])) / d_model)) # [max_len / 2]

    positional_encoding[:, 0::2] = torch.sin(position * div_term) # even
    positional_encoding[:, 1::2] = torch.cos(position * div_term) # odd

    # [max_len, d_model] -> [1, max_len, d_model] -> [max_len, 1, d_model]
    positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

    # register pe to buffer and require no grads
    self.register_buffer('pe', positional_encoding)

  def forward(self, x):
    # x: [seq_len, batch, d_model]
    # we can add positional encoding to x directly, and ignore other dimension
    x = x + self.pe[:x.size(0), :]

    return self.dropout(x)


class FeedForwardNetwork(nn.Module):
  '''
  Using nn.Conv1d replace nn.Linear to implements FFN.
  '''
  def __init__(self, d_model=512 , d_ff=2048 ,dropout=0.0):
    super(FeedForwardNetwork, self).__init__()
    # self.ff1 = nn.Linear(d_model, d_ff)
    # self.ff2 = nn.Linear(d_ff, d_model)
    self.ff1 = nn.Conv1d(d_model, d_ff, 1)
    self.ff2 = nn.Conv1d(d_ff, d_model, 1)
    self.relu = nn.ReLU()

    self.dropout = nn.Dropout(dropout)
    # self.layer_norm = nn.LayerNorm(d_model).to(device)
    self.layer_norm = nn.LayerNorm(d_model)

  def forward(self, x):
    # x: [batch, seq_len, d_model]
    residual = x
    x = x.transpose(1, 2) # [batch, d_model, seq_len]
    x = self.ff1(x)
    x = self.relu(x)
    x = self.ff2(x)
    x = x.transpose(1, 2) # [batch, seq_len, d_model]

    return self.layer_norm(residual + x)


class EncoderLayer(nn.Module):

  def __init__(self, d_model=512 , n_heads=8, d_ff=2048 ,dropout=0.0):
    super(EncoderLayer, self).__init__()
    self.encoder_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
    self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

  def forward(self, encoder_input, encoder_pad_mask):
    '''
    encoder_input: [batch, source_len, d_model]
    encoder_pad_mask: [batch, n_heads, source_len, source_len]

    encoder_output: [batch, source_len, d_model]
    attn: [batch, n_heads, source_len, source_len]
    '''
    encoder_output, attn = self.encoder_self_attn(encoder_input, encoder_input, encoder_input, encoder_pad_mask)
    encoder_output = self.ffn(encoder_output) # [batch, source_len, d_model]

    return encoder_output, attn


class Encoder(nn.Module):

  def __init__(self, source_vocab_size, max_seq_len, n_layers=6, d_model=512, n_heads=8, d_ff=2048, dropout=0.0):
    super(Encoder, self).__init__()
    self.source_embedding = nn.Embedding(source_vocab_size, d_model)
    self.positional_embedding = PositionalEncoding(d_model, max_len=max_seq_len)
    self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for layer in range(n_layers)])

  def forward(self, encoder_input):
    # encoder_input: [batch, source_len]
    encoder_output = self.source_embedding(encoder_input) # [batch, source_len, d_model]
    encoder_output = self.positional_embedding(encoder_output.transpose(0, 1)).transpose(0, 1) # [batch, source_len, d_model]

    encoder_self_attn_mask = get_attn_pad_mask(encoder_input, encoder_input) # [batch, source_len, source_len]
    encoder_self_attns = list()
    for layer in self.layers:
      # encoder_output: [batch, source_len, d_model]
      # encoder_self_attn: [batch, n_heads, source_len, source_len]
      encoder_output, encoder_self_attn = layer(encoder_output, encoder_self_attn_mask)
      encoder_self_attns.append(encoder_self_attn)

    return encoder_output, encoder_self_attns


class DecoderLayer(nn.Module):

  def __init__(self, d_model=512 , n_heads=8, d_ff=2048 ,dropout=0.0):
    super(DecoderLayer, self).__init__()
    self.decoder_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
    self.encoder_decoder_attn = MultiHeadAttention(d_model , n_heads, dropout)
    self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

  def forward(self, decoder_input, encoder_output, decoder_self_mask, decoder_encoder_mask):
    '''
    decoder_input: [batch, target_len, d_mdoel]
    encoder_output: [batch, source_len, d_model]
    decoder_self_mask: [batch, target_len, target_len]
    decoder_encoder_mask: [batch, target_len, source_len]
    '''
    # masked mutlihead attention
    # Q, K, V all from decoder it self
    # decoder_output: [batch, target_len, d_model]
    # decoder_self_attn: [batch, n_heads, target_len, target_len]
    decoder_output, decoder_self_attn = self.decoder_self_attn(decoder_input, decoder_input, decoder_input, decoder_self_mask)

    # Q from decoder, K, V from encoder
    # decoder_output: [batch, target_len, d_model]
    # decoder_encoder_attn: [batch, n_heads, target_len, source_len]
    decoder_output, decoder_encoder_attn = self.encoder_decoder_attn(decoder_output, encoder_output, encoder_output, decoder_encoder_mask)
    decoder_output = self.ffn(decoder_output) # [batch, target_len, d_model]

    return decoder_output, decoder_self_attn, decoder_encoder_attn


class Decoder(nn.Module):

  def __init__(self, target_vocab_size, max_seq_len, n_layers=6, d_model=512, n_heads=8, d_ff=2048, dropout=0.0):
    super(Decoder, self).__init__()
    self.target_embedding = nn.Embedding(target_vocab_size, d_model)
    self.positional_embedding = PositionalEncoding(d_model, max_len=max_seq_len)
    self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for layer in range(n_layers)])

  def forward(self, decoder_input, encoder_input, encoder_output):
    '''
    decoder_input: [batch, target_len]
    encoder_input: [batch, source_len]
    encoder_output: [batch, source_len, d_model]
    '''
    decoder_output = self.target_embedding(decoder_input) # [batch, target_len, d_model]
    decoder_output = self.positional_embedding(decoder_output.transpose(0, 1)).transpose(0, 1).to(device) # [batch, target_len, d_model]
    decoder_self_attn_mask = get_attn_pad_mask(decoder_input, decoder_input).to(device) # [batch, target_len, target_len]
    decoder_subsequent_mask = get_attn_subsequent_mask(decoder_input).to(device) # [batch, target_len, target_len]

    decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input) # [batch, target_len, source_len]

    decoder_self_mask = torch.gt(decoder_self_attn_mask + decoder_subsequent_mask, 0).to(device)
    decoder_self_attns, decoder_encoder_attns = [], []

    for layer in self.layers:
      # decoder_output: [batch, target_len, d_model]
      # decoder_self_attn: [batch, n_heads, target_len, target_len]
      # decoder_encoder_attn: [batch, n_heads, target_len, source_len]
      decoder_output, decoder_self_attn, decoder_encoder_attn = layer(decoder_output, encoder_output, decoder_self_mask, decoder_encoder_attn_mask)
      decoder_self_attns.append(decoder_self_attn)
      decoder_encoder_attns.append(decoder_encoder_attn)

    return decoder_output, decoder_self_attns, decoder_encoder_attns


class Transformer(nn.Module):

  def __init__(self, src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len, n_layers=6, d_model=512, n_heads=8, d_ff=2048, dropout=0.2):
    super(Transformer, self).__init__()

    self.encoder = Encoder(src_vocab_size, src_max_len, n_layers, d_model, n_heads, d_ff, dropout).to(device)
    self.decoder = Decoder(tgt_vocab_size, tgt_max_len, n_layers, d_model, n_heads, d_ff, dropout).to(device)
    self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)
    # self.softmax = nn.Softmax(dim=2)

  def forward(self, encoder_input, decoder_input):
    '''
    encoder_input: [batch, source_len]
    decoder_input: [batch, target_len]
    '''
    # encoder_output: [batch, source_len, d_model]
    # encoder_attns: [n_layers, batch, n_heads, source_len, source_len]
    encoder_output, encoder_attns = self.encoder(encoder_input)
    # decoder_output: [batch, target_len, d_model]
    # decoder_self_attns: [n_layers, batch, n_heads, target_len, target_len]
    # decoder_encoder_attns: [n_layers, batch, n_heads, target_len, source_len]
    decoder_output, decoder_self_attns, decoder_encoder_attns = self.decoder(decoder_input, encoder_input, encoder_output)
    decoder_logits = self.projection(decoder_output) # [batch, target_len, target_vocab_size]

    # decoder_logits: [batch * target_len, target_vocab_size]
    # output = self.softmax(decoder_logits)
    # output = self.softmax(decoder_logits.view(-1, decoder_logits.size(-1)))
    return decoder_logits.view(-1, decoder_logits.size(-1)), encoder_attns, decoder_self_attns, decoder_encoder_attns
    # return output, encoder_attns, decoder_self_attns, decoder_encoder_attns


##########################################################################
## test get_attn_pad_mask

# q = k = tensor([[1, 1, 0, 0]])

# mask = get_attn_pad_mask(q, k)
# print(mask)
# print(q.shape, mask.shape)



##########################################################################
## test MultiHeadAttention, ScaledDotProductAttention

# d_model, d_k, n_heads = 512, 64, 8
# d_k, d_v = 64, 64
# # q = k = v = np.ones((1, 64, 512))
# q = k = v = tensor(np.ones((1, 64, 512) , np.float32))
# # print("q.size(0):", q.size(0))
# attn_mask = tensor([False]).broadcast_to((1, 64, 64))
# multi_head_attn = MultiHeadAttention(n_heads)
# output, attn = multi_head_attn(q, k, v, attn_mask)
# print(output.shape, attn.shape)


##########################################################################
## test PositionalEncoding

# x = torch.zeros((1, 2, 4))
# pe = PositionalEncoding(d_model=4, max_len=100)
# print(pe(x))


##########################################################################
## test FeedForwardNetwork
# d_ff = 16
# d_model = 4
# p_drop = 0.1
# x = torch.ones((1, 2, 4))
# ffn = FeedForwardNetwork()
# print(ffn(x).shape)


##########################################################################
## test EncoderLayer
# d_model, n_heads, d_ff = 8, 4, 16
# d_k = 2
# d_v = 2
# p_drop = 0.1

# x = torch.ones((1, 2, 8))
# mask = tensor([False]).broadcast_to((1, 2, 2))
# encoder_layer = EncoderLayer()
# output, attn = encoder_layer(x, mask)
# print(output.shape, attn.shape)


##########################################################################
## test get_attn_subsequent_mask
# seq = torch.ones((1, 4))
# mask = get_attn_subsequent_mask(seq)
# print(mask)

##########################################################################
## test DecoderLayer

# d_ff = 16
# d_model = 4
# p_drop = 0.1
# n_heads = 1
# d_k = 4
# d_v = 4

# self.decoder_self_attn = MultiHeadAttention(n_heads = 1)
# x = y = torch.ones((1, 2, 4))
# mask1 = mask2 = tensor([False]).broadcast_to((1, 2, 2))
# decoder_layer = DecoderLayer()
# output, attn1, attn2 = decoder_layer(x, y, mask1, mask2)
# print(output.shape, attn1.shape, attn2.shape)