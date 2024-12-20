import os
import lzma
import re
from collections import Counter
import json
import matplotlib.pyplot as plt

# Reversible preprocessing function
def preprocess_text_optimized(text, threshold=50, min_phrase_length=5):
    """
    Extracts and replaces high-frequency long phrases in the text, returning the processed text and metadata.
    """
    print("Extracting high-frequency long phrases...")
    word_freq = Counter(text.split())
    # Extract phrases with high frequency and length greater than the minimum length
    high_freq_phrases = [word for word, count in word_freq.items() if count > threshold and len(word) >= min_phrase_length]
    phrase_metadata = {"high_freq_phrases": high_freq_phrases}

    print("Replacing high-frequency long phrases with compact placeholders...")
    # Create compact placeholders (e.g., #0, #1, ...)
    phrase_pattern = re.compile(r'\b(' + '|'.join(re.escape(phrase) for phrase in high_freq_phrases) + r')\b')
    processed_text = phrase_pattern.sub(lambda match: f"#{high_freq_phrases.index(match.group(0))}", text)

    metadata = {"phrases": phrase_metadata}
    return processed_text, metadata

# Reversible restoration function
def restore_text(preprocessed_text, metadata):
    """
    Restores the original text based on metadata.
    """
    print("Restoring high-frequency long phrases...")
    high_freq_phrases = metadata["phrases"]["high_freq_phrases"]
    for i, phrase in enumerate(high_freq_phrases):
        preprocessed_text = preprocessed_text.replace(f"#{i}", phrase)
    return preprocessed_text

# Default LZMA compression function
def compress_with_default_lzma(input_text, output_file):
    """
    Compresses the text using default LZMA.
    """
    print("Compressing with default LZMA...")
    compressed_data = lzma.compress(input_text.encode('utf-8'))
    with open(output_file, "wb") as f_out:
        f_out.write(compressed_data)
    compressed_size = os.path.getsize(output_file)
    print(f"Default LZMA compressed size: {compressed_size} bytes")
    return compressed_size

# NLP + LZMA compression function
def compress_with_nlp_and_lzma(input_text, metadata, output_file):
    """
    Compresses the text using NLP preprocessing followed by LZMA.
    """
    print("Compressing with NLP + LZMA...")
    compressed_data = lzma.compress(input_text.encode('utf-8'))
    with open(output_file, "wb") as f_out:
        f_out.write(compressed_data)
    compressed_size = os.path.getsize(output_file)
    print(f"NLP + LZMA compressed size: {compressed_size} bytes")
    return compressed_size

# Plot compression results
def plot_compression_results(original_size, default_size, nlp_size):
    """
    Plots a comparison of original data size, LZMA compressed size, and NLP+LZMA compressed size.
    """
    sizes = [original_size, default_size, nlp_size]
    labels = ["Original Size", "LZMA", "NLP + LZMA"]
    colors = ["gray", "blue", "green"]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color=colors, alpha=0.7)
    plt.xlabel("Compression Methods")
    plt.ylabel("File Size (Bytes)")
    plt.title("Compression Comparison")
    for i, size in enumerate(sizes):
        plt.text(i, size + 0.05 * max(sizes), f"{size} bytes", ha="center")
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    input_file = "enwik8"  # Original file
    default_file = "enwik8_default.xz"  # Default LZMA compressed file
    nlp_file = "enwik8_nlp.xz"  # NLP + LZMA compressed file

    # Read the entire enwik8 file
    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    original_size = len(raw_text.encode('utf-8'))  # Original data size (in bytes)
    print(f"Original file size: {original_size} bytes")

    # 1. Default LZMA compression
    default_size = compress_with_default_lzma(raw_text, default_file)

    # 2. NLP preprocessing + LZMA compression
    # Set high-frequency phrase threshold to 50 and minimum phrase length to 5
    preprocessed_text, metadata = preprocess_text_optimized(raw_text, threshold=50, min_phrase_length=5)
    nlp_size = compress_with_nlp_and_lzma(preprocessed_text, metadata, nlp_file)

    # 3. Plot comparison
    plot_compression_results(original_size, default_size, nlp_size)

    # 4. Comparison results
    print("\nComparison Results:")
    print(f"Original file size: {original_size} bytes")
    print(f"Default LZMA compressed size: {default_size} bytes")
    print(f"NLP + LZMA compressed size: {nlp_size} bytes")
    improvement = 100 * (default_size - nlp_size) / default_size
    print(f"Compression improvement: {improvement:.2f}%")
