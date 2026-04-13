import torch
import torch.nn as nn
import math
from torch.nn.modules import ModuleList

## Embedding
class inputEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    def forward(self, x):
        x = self.embedding(x)
        return x

## Positional Encoding
class positionalEncoding(nn.Module):
    def __init__(self, seq_len, d_k):
        super().__init__()
        self.seq_len = seq_len
        self.d_k = d_k
        pe = torch.zeros(seq_len, d_k)
        div = torch.exp(torch.arange(0, self.d_k, 2).float() * - (math.log(10000.0) / self.d_k))
        positions = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].requires_grad_(False)

## Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
        self.fnn = nn.Sequential(
            nn.Linear(self.d_k, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.d_k)
        )
    def forward(self, x):
        return self.fnn(x)

## layer norm
class layerNorm(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.ones(embedding_dim))
        self.eps = 0.001
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = self.alpha * (x - mean) / (self.eps + std) + self.beta
        return x

## Multihead Self Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, heads, head_dim, device):
        super().__init__()
        self.d_k = d_k
        assert d_k == heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.Q = nn.Linear(self.d_k, self.d_k)
        self.K = nn.Linear(self.d_k, self.d_k)
        self.V = nn.Linear(self.d_k, self.d_k)
        self.device = device
        self.out = nn.Linear(self.d_k, self.d_k)
    def forward(self, x, mask):
        B, S, E = x.size()
        q = self.Q(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.K(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = self.V(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = k.transpose(-2, -1)
        attention_scores = ((q @ k) / math.sqrt(self.d_k))
        if mask:
            mat = torch.tril(torch.ones(S, S)).to(self.device)
            mat = mat.unsqueeze(0).unsqueeze(0)
            attention_scores = attention_scores.masked_fill(mat == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1) @ v
        attention_score = attention_scores.transpose(1, 2).contiguous().view(B, S, E)
        return self.out(attention_score)

## cross attention
class CrossAttention(nn.Module):
    def __init__(self, d_k, heads, head_dim):
        super().__init__()
        self.d_k = d_k
        assert d_k == heads * head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.Q = nn.Linear(self.d_k, self.d_k)
        self.out = nn.Linear(self.d_k, self.d_k)
    def forward(self, x, k, v):
        B, S, E = x.size()
        B_enc, S_enc, E_enc = k.size()
        q = self.Q(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B_enc, S_enc, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B_enc, S_enc, self.heads, self.head_dim).transpose(1, 2)
        k = k.transpose(-2, -1)
        attention_scores = ((q @ k) / math.sqrt(self.d_k))
        attention_scores = torch.softmax(attention_scores, dim=-1) @ v
        attention_score = attention_scores.transpose(1, 2).contiguous().view(B, S, E)
        return self.out(attention_score)

class EncoderBlock(nn.Module):
    def __init__(self, attention: MultiHeadAttention, fnn: FeedForward, d_k):
        super().__init__()
        self.attention = attention
        self.fnn = fnn
        self.norm = layerNorm(d_k)
    def forward(self, x):
        x = self.norm(x + self.attention(x, 0))
        x = self.norm(x + self.fnn(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, attention: MultiHeadAttention, fnn: FeedForward, cr: CrossAttention, d_k):
        super().__init__()
        self.attention = attention
        self.fnn = fnn
        self.norm = layerNorm(d_k)
        self.cr = cr
    def forward(self, x, k, v):
        x = self.norm(x + self.attention(x, 1))
        x = self.norm(x + self.cr(x, k, v))
        x = self.norm(x + self.fnn(x))
        return x

class InitialLayer(nn.Module):
    def __init__(self, embedding: inputEmbedding, positional: positionalEncoding):
        super().__init__()
        self.embedding = embedding
        self.pos = positional
    def forward(self, x):
        x = self.pos(self.embedding(x))
        return x

class OutputLayer(nn.Module):
    def __init__(self, d_k, vocab_size):
        super().__init__()
        self.d_k = d_k
        self.out = nn.Linear(d_k, vocab_size)
    def forward(self, x):
        return self.out(x)

class Encoder(nn.Module):
    def __init__(self, sublist: nn.ModuleList, d_k):
        super().__init__()
        self.sublayer = sublist
        self.norm = layerNorm(d_k)
    def forward(self, x):
        for layer in self.sublayer:
            x = layer(x)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, sublist: nn.ModuleList, d_k):
        super().__init__()
        self.sublayer = sublist
        self.norm = layerNorm(d_k)
    def forward(self, x, k, v):
        for layer in self.sublayer:
            x = layer(x, k, v)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, initial: InitialLayer, out: OutputLayer, d_k):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out = out
        self.initial = initial
        self.K = nn.Linear(d_k, d_k)
        self.V = nn.Linear(d_k, d_k)
    def forward(self, x, y):
        x = self.initial(x)
        y = self.initial(y)
        x = self.encoder(x)
        k, v = self.K(x), self.V(x)
        y = self.decoder(y, k, v)
        y = self.out(y)
        return y
