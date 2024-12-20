import torch
import torch.nn as nn
import torch.optim as optim
import lzma
import os
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom dataset for character-level data
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        input_seq = [ord(c) % 256 for c in self.text[idx:idx + self.seq_length]]
        target = ord(self.text[idx + self.seq_length]) % 256
        return torch.tensor(input_seq), torch.tensor(target)

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        out = self.fc(out[:, -1, :])
        return out, h

# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = torch.zeros(num_layers, inputs.size(0), hidden_size).to(device)
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

# Compression Function
def compress_text(model, text, seq_length, output_file):
    model.eval()
    compressed_data = b""
    with torch.no_grad():
        hidden = torch.zeros(num_layers, 1, hidden_size).to(device)
        for i in range(len(text) - seq_length):
            input_seq = torch.tensor([ord(c) % 256 for c in text[i:i + seq_length]]).unsqueeze(0).to(device)
            output, hidden = model(input_seq, hidden)
            predicted = torch.argmax(output, dim=1).item()
            compressed_data += bytes([predicted])
    compressed_data = lzma.compress(compressed_data)
    with open(output_file, "wb") as f:
        f.write(compressed_data)
    return len(compressed_data)

# Default LZMA Compression
def default_lzma_compression(input_text, output_file):
    compressed_data = lzma.compress(input_text.encode('utf-8'))
    with open(output_file, "wb") as f:
        f.write(compressed_data)
    return len(compressed_data)

# Main Program
if __name__ == "__main__":
    set_seed(42)  # 固定随机种子

    train_file = "enwik9"
    compress_file = "enwik8"
    rnn_output_file = "compressed_enwik8_rnn.xz"
    default_output_file = "compressed_enwik8_default.xz"

    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()[:10000]  # Limit training data size to 80K characters

    with open(compress_file, "r", encoding="utf-8") as f:
        compress_text_data = f.read()[:10000]  # Limit compression data size to 20K characters

    vocab_size = 256  # ASCII range
    embed_size = 256
    hidden_size = 1024
    num_layers = 4
    seq_length = 35  # Increased sequence length
    batch_size = 16  # Increased batch size
    num_epochs = 3  # Reduced epochs to speed up training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TextDataset(train_text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = RNNModel(vocab_size, embed_size, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs)

    rnn_compressed_size = compress_text(model, compress_text_data, seq_length, rnn_output_file)
    default_compressed_size = default_lzma_compression(compress_text_data, default_output_file)

    print(f"RNN Compression completed. Output file: {rnn_output_file}, Size: {rnn_compressed_size} bytes")
    print(f"Default LZMA Compression completed. Output file: {default_output_file}, Size: {default_compressed_size} bytes")
    print(f"Compression Comparison: RNN-based: {rnn_compressed_size} bytes, Default LZMA: {default_compressed_size} bytes")
