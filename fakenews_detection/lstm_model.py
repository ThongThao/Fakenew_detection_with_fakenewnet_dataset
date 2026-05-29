import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter

class Vocab:
    """
    Custom Vocabulary builder for mapping words to indices.
    """
    def __init__(self, max_size=10000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.w2i = {self.pad_token: 0, self.unk_token: 1}
        self.i2w = {0: self.pad_token, 1: self.unk_token}
        
    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
            
        # Filter by min frequency
        sorted_words = [word for word, freq in counter.most_common() if freq >= self.min_freq]
        
        # Limit to max size
        sorted_words = sorted_words[:self.max_size - len(self.w2i)]
        
        for idx, word in enumerate(sorted_words):
            i = len(self.w2i)
            self.w2i[word] = i
            self.i2w[i] = word
            
        print(f"Built vocabulary with {len(self.w2i)} unique tokens.")
        
    def encode(self, text):
        return [self.w2i.get(word, self.w2i[self.unk_token]) for word in text.split()]
        
    def __len__(self):
        return len(self.w2i)


class LSTMDataset(Dataset):
    """
    PyTorch Dataset for LSTM, handling tokenization, padding, and truncation.
    """
    def __init__(self, texts, labels, vocab, max_len=50):
        self.labels = labels
        self.max_len = max_len
        self.encoded_texts = []
        
        for text in texts:
            encoded = vocab.encode(text)
            # Truncate
            if len(encoded) > max_len:
                encoded = encoded[:max_len]
            # Pad
            else:
                encoded = encoded + [vocab.w2i[vocab.pad_token]] * (max_len - len(encoded))
            self.encoded_texts.append(encoded)
            
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class LSTMClassifier(nn.Module):
    """
    LSTM text classifier with embedding, bidirectional LSTM, Max-Pooling, and Fully Connected layer.
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, output_dim=2, 
                 n_layers=1, bidirectional=True, dropout_rate=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            bidirectional=bidirectional, 
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0.0
        )
        
        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)
        
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)
        # embedded: (batch_size, seq_len, embedding_dim)
        
        # LSTM output
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        
        # Apply global max pooling over sequence length
        pooled, _ = torch.max(lstm_out, dim=1)
        # pooled: (batch_size, hidden_dim * num_directions)
        
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        # logits: (batch_size, output_dim)
        
        return logits
