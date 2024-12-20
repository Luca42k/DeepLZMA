import os
import lzma
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import re
import matplotlib.pyplot as plt

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

# NLP 预处理函数
def preprocess_text_optimized(text, threshold=50, min_phrase_length=5):
    """
    对文本进行高频长短语提取和替换，返回处理后的文本和元数据。
    """
    print("Extracting high-frequency long phrases...")
    word_freq = Counter(text.split())  # 按空格分割单词
    # 提取非空且符合条件的高频短语
    high_freq_phrases = [
        word for word, count in word_freq.items()
        if count > threshold and len(word) >= min_phrase_length and word.strip()
    ]

    if not high_freq_phrases:  # 如果没有符合条件的短语
        print("No high-frequency phrases found. Skipping NLP preprocessing.")
        return text, {"phrases": {"high_freq_phrases": []}}

    print("Replacing high-frequency long phrases with compact placeholders...")
    phrase_pattern = re.compile(r'\b(' + '|'.join(re.escape(phrase) for phrase in high_freq_phrases) + r')\b')
    processed_text = phrase_pattern.sub(lambda match: f"#{high_freq_phrases.index(match.group(0))}", text)

    metadata = {"phrases": {"high_freq_phrases": high_freq_phrases}}
    return processed_text, metadata

# NLP 还原函数
def restore_text(preprocessed_text, metadata):
    high_freq_phrases = metadata["phrases"]["high_freq_phrases"]
    for i, phrase in enumerate(high_freq_phrases):
        preprocessed_text = preprocessed_text.replace(f"#{i}", phrase)
    return preprocessed_text

# 数据集类
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

# RNN 模型
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

# RNN 压缩函数
def compress_with_rnn(model, text, seq_length, output_file):
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
    return os.path.getsize(output_file)

# 默认 LZMA 压缩函数
def compress_with_default_lzma(input_text, output_file):
    compressed_data = lzma.compress(input_text.encode('utf-8'))
    with open(output_file, "wb") as f_out:
        f_out.write(compressed_data)
    return os.path.getsize(output_file)

# 绘制对比图
def plot_compression_results(original_size, default_size, nlp_size, rnn_size):
    sizes = [original_size, default_size, nlp_size, rnn_size]
    labels = ["Original", "LZMA", "NLP+LZMA", "NLP+RNN+LZMA"]
    colors = ["gray", "blue", "green", "orange"]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color=colors, alpha=0.7)
    plt.xlabel("Compression Methods")
    plt.ylabel("File Size (Bytes)")
    plt.title("Compression Comparison")
    for i, size in enumerate(sizes):
        plt.text(i, size + 0.05 * max(sizes), f"{size} bytes", ha="center")
    plt.tight_layout()
    plt.show()

# 训练模型函数
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
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

# 主程序
if __name__ == "__main__":
    set_seed(42)

    train_file = "enwik9"
    compress_file = "enwik8"
    rnn_output_file = "compressed_enwik8_rnn.xz"
    default_output_file = "compressed_enwik8_default.xz"
    nlp_file = "compressed_enwik8_nlp.xz"

    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()[:100000]

    with open(compress_file, "r", encoding="utf-8") as f:
        compress_text_data = f.read()[:100000]

    # 原始大小
    original_size = len(compress_text_data.encode('utf-8'))

    # 默认 LZMA 压缩
    default_size = compress_with_default_lzma(compress_text_data, default_output_file)

    # NLP 预处理 + LZMA 压缩
    preprocessed_text, metadata = preprocess_text_optimized(compress_text_data)
    nlp_size = compress_with_default_lzma(preprocessed_text, nlp_file)

    # RNN 压缩
    vocab_size = 256
    embed_size = 256
    hidden_size = 512
    num_layers = 3
    seq_length = 35
    batch_size = 16
    num_epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TextDataset(train_text, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = RNNModel(vocab_size, embed_size, hidden_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs)

    rnn_size = compress_with_rnn(model, preprocessed_text, seq_length, rnn_output_file)

    # 绘制对比图
    plot_compression_results(original_size, default_size, nlp_size, rnn_size)
