import re
from collections import Counter, defaultdict

def get_vocab(corpus):
    """
    Create a vocabulary from the corpus with character-level tokens.
    """
    vocab = Counter()
    for word in corpus:
        word = ' '.join(list(word)) + ' </w>'
        vocab[word] += 1
    return vocab

def get_stats(vocab):
    """
    Get frequency of pairs of characters.
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """
    Merge the most frequent pair.
    """
    bigram = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    new_vocab = {}
    for word in vocab:
        new_word = pattern.sub(''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def bpe(corpus, num_merges):
    """
    Perform Byte Pair Encoding on the corpus.
    """
    vocab = get_vocab(corpus)
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_vocab(best_pair, vocab)
    return vocab

# Example usage:
corpus = ["hello", "hell", "heaven", "goodbye"]
num_merges = 10
vocab = bpe(corpus, num_merges)
print(vocab)

import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, input_tokens):
        return self.embedding(input_tokens)

# Example usage:
vocab_size = 10000  # Example vocabulary size
d_model = 512       # Embedding dimension
embedding_layer = EmbeddingLayer(vocab_size, d_model)

input_tokens = torch.LongTensor([[1, 2, 3], [4, 5, 6]])  # Example input
embeddings = embedding_layer(input_tokens)
print(embeddings.shape)  # (2, 3, 512)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = self._get_positional_embeddings(max_len, d_model)

    def _get_positional_embeddings(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x):
        return x + self.pos_embedding[:x.size(0), :]

# Example usage:
d_model = 512
max_len = 1000
pos_encoding = PositionalEncoding(d_model, max_len)

# Assume embeddings is the output from the embedding layer
embeddings = torch.randn(50, 32, d_model)  # (seq_len, batch_size, d_model)
embeddings_with_pos = pos_encoding(embeddings)
print(embeddings_with_pos.shape)  # (50, 32, 512)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(TransformerModel, self).__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
    
    def forward(self, input_tokens):
        embeddings = self.embedding_layer(input_tokens)
        embeddings_with_pos = self.positional_encoding(embeddings)
        return embeddings_with_pos

# Example usage:
vocab_size = 10000
d_model = 512
max_len = 1000
model = TransformerModel(vocab_size, d_model, max_len)

input_tokens = torch.randint(0, vocab_size, (50, 32))  # (seq_len, batch_size)
output = model(input_tokens)
print(output.shape)  # (50, 32, 512)
