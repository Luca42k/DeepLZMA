import os
import time
import lzma
import zipfile
import gzip
import bz2
import zlib
from heapq import heappush, heappop
from collections import Counter
import matplotlib.pyplot as plt


def measure_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


# LZMA Compression
def compress_lzma(file_path):
    print("Compressing with LZMA...")
    with open(file_path, 'rb') as f:
        data = f.read()
    compressed_data, duration = measure_time(lzma.compress, data)
    compressed_size = len(compressed_data)
    print(f"LZMA compression completed in {duration:.4f}s, compressed size: {compressed_size} bytes")
    return compressed_size, duration


# ZIP Compression
def compress_zip(file_path):
    print("Compressing with ZIP...")
    compressed_file = file_path + '.zip'
    start = time.time()
    with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, arcname=os.path.basename(file_path))
    end = time.time()
    compressed_size = os.path.getsize(compressed_file)
    duration = end - start
    print(f"ZIP compression completed in {duration:.4f}s, compressed size: {compressed_size} bytes")
    return compressed_size, duration


# GZIP Compression
def compress_gzip(file_path):
    print("Compressing with GZIP...")
    compressed_file = file_path + '.gz'
    start = time.time()
    with open(file_path, 'rb') as f_in, gzip.open(compressed_file, 'wb') as f_out:
        while chunk := f_in.read(1024 * 1024):
            f_out.write(chunk)
    end = time.time()
    compressed_size = os.path.getsize(compressed_file)
    duration = end - start
    print(f"GZIP compression completed in {duration:.4f}s, compressed size: {compressed_size} bytes")
    return compressed_size, duration


# BZIP2 Compression
def compress_bzip2(file_path):
    print("Compressing with BZIP2...")
    compressed_file = file_path + '.bz2'
    start = time.time()
    with open(file_path, 'rb') as f_in, bz2.open(compressed_file, 'wb') as f_out:
        while chunk := f_in.read(1024 * 1024):
            f_out.write(chunk)
    end = time.time()
    compressed_size = os.path.getsize(compressed_file)
    duration = end - start
    print(f"BZIP2 compression completed in {duration:.4f}s, compressed size: {compressed_size} bytes")
    return compressed_size, duration


# Huffman Compression
def compress_huffman(data):
    print("Compressing with Huffman...")
    start = time.time()
    frequencies = Counter(data)
    heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
    while len(heap) > 1:
        low = heappop(heap)
        high = heappop(heap)
        for pair in low[1:]:
            pair[1] = '0' + pair[1]
        for pair in high[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [low[0] + high[0]] + low[1:] + high[1:])
    huffman_code = {symbol: code for symbol, code in sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))}
    encoded_data = "".join(huffman_code[byte] for byte in data)
    padded_encoded_data = encoded_data + '0' * ((8 - len(encoded_data) % 8) % 8)
    byte_array = bytearray(int(padded_encoded_data[i:i + 8], 2) for i in range(0, len(padded_encoded_data), 8))
    compressed_size = len(byte_array)
    duration = time.time() - start
    print(f"Huffman compression completed in {duration:.4f}s, compressed size: {compressed_size} bytes")
    return compressed_size, duration


# Deflate Compression
def compress_deflate(data):
    print("Compressing with Deflate...")
    start = time.time()
    compressed_data = zlib.compress(data)
    end = time.time()
    compressed_size = len(compressed_data)
    duration = end - start
    print(f"Deflate compression completed in {duration:.4f}s, compressed size: {compressed_size} bytes")
    return compressed_size, duration


# LZ77 Compression
def compress_lz77(data, window_size=100):
    print("Compressing with LZ77...")
    start = time.time()
    compressed_data = []
    i = 0
    data_length = len(data)
    while i < data_length:
        match_length = 0
        match_distance = 0
        start_window = max(0, i - window_size)
        for j in range(start_window, i):
            length = 0
            while (i + length < data_length and data[j + length] == data[i + length]):
                length += 1
                if length > match_length:
                    match_length = length
                    match_distance = i - j
        next_char = data[i + match_length] if i + match_length < data_length else None
        compressed_data.append((match_distance, match_length, next_char))
        i += match_length + 1 if next_char is not None else 1
    end = time.time()
    compressed_size = len(compressed_data) * 3  # Estimate each tuple takes ~3 bytes
    duration = end - start
    print(f"LZ77 compression completed in {duration:.4f}s, compressed size: {compressed_size} bytes")
    return compressed_size, duration


# Plot Results with Different Colors
def plot_results(original_size, results):
    algorithms = list(results.keys())
    sizes = [original_size] + [results[alg][0] for alg in algorithms]
    labels = ['Original'] + algorithms

    # Adding RAR manually
    rar_size = 28662796  # RAR 压缩结果
    sizes.append(rar_size)
    labels.append('RAR')

    # 定义颜色，每种算法一个颜色，包括 RAR
    colors = ['gray', 'blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color=colors[:len(labels)], edgecolor='black')
    plt.xlabel("Compression Algorithms")
    plt.ylabel("File Size (Bytes)")
    plt.title("Compression Algorithm Size Comparison")
    plt.xticks(rotation=45)
    for i, size in enumerate(sizes):
        plt.text(i, size + 500, f"{size}", ha='center')
    plt.tight_layout()
    plt.show()

# Combined Experiment
def run_combined_experiment(file_path):
    results = {}
    original_size = os.path.getsize(file_path)

    with open(file_path, 'rb') as f:
        data = f.read()

    print(f"Original file size: {original_size} bytes\n")

    # LZMA
    results['LZMA'] = compress_lzma(file_path)

    # ZIP
    results['ZIP'] = compress_zip(file_path)

    # GZIP
    results['GZIP'] = compress_gzip(file_path)

    # BZIP2
    results['BZIP2'] = compress_bzip2(file_path)

    # Huffman
    huffman_size, huffman_duration = compress_huffman(data)
    results['Huffman'] = (huffman_size, huffman_duration)

    # Deflate
    deflate_size, deflate_duration = compress_deflate(data)
    results['Deflate'] = (deflate_size, deflate_duration)

    # LZ77
    lz77_size, lz77_duration = compress_lz77(data)
    results['LZ77'] = (lz77_size, lz77_duration)

    plot_results(original_size, results)


# Ensure test file exists
test_file_path = "enwik8"
if not os.path.exists(test_file_path):
    with open(test_file_path, 'w') as f:
        f.write("This is a simple test file for compression.\n" * 1000)

# Run the experiment
run_combined_experiment(test_file_path)
