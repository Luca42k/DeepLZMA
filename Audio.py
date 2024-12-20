import os
import time
import lzma
import zipfile
import gzip
import bz2
import zlib
import matplotlib.pyplot as plt


# Utility: Measure time for a function
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


# Deflate Compression
def compress_deflate(file_path):
    print("Compressing with Deflate...")
    with open(file_path, 'rb') as f:
        data = f.read()
    start = time.time()
    compressed_data = zlib.compress(data)
    end = time.time()
    compressed_size = len(compressed_data)
    duration = end - start
    print(f"Deflate compression completed in {duration:.4f}s, compressed size: {compressed_size} bytes")
    return compressed_size, duration


# Plot Results with RAR and Scatter Plot
def plot_results_with_rar_and_scatter(original_size, results):
    # Prepare data for bar chart and scatter plot
    algorithms = list(results.keys())  # Dynamically fetch all algorithms from results
    sizes = [original_size] + [results[alg][0] for alg in algorithms] + [39086488]  # Add RAR size manually
    durations = [0] + [results[alg][1] for alg in algorithms] + [None]  # Add duration with None for RAR
    labels = ['Original'] + algorithms + ['RAR']
    
    # Bar chart for file sizes
    plt.figure(figsize=(12, 6))
    plt.bar(labels, sizes, color=['gray', 'blue', 'green', 'red', 'purple', 'orange', 'gold'], edgecolor='black')
    plt.xlabel("Compression Algorithms")
    plt.ylabel("File Size (Bytes)")
    plt.title("Compression Algorithm Size Comparison (Including RAR)")
    plt.xticks(rotation=45)
    for i, size in enumerate(sizes):
        plt.text(i, size + 500, f"{size}", ha='center')
    plt.tight_layout()
    plt.show()

    # Scatter plot for file size vs compression time
    valid_algorithms = [alg for alg in algorithms if results[alg][1] is not None]  # Exclude algorithms without time data
    valid_sizes = [results[alg][0] for alg in valid_algorithms]
    valid_durations = [results[alg][1] for alg in valid_algorithms]

    plt.figure(figsize=(10, 6))
    scatter_colors = ['blue', 'green', 'red', 'purple', 'orange'][:len(valid_algorithms)]  # Match colors to algorithms
    for i, alg in enumerate(valid_algorithms):
        plt.scatter(valid_sizes[i], valid_durations[i], label=alg, color=scatter_colors[i], s=100)
    plt.xlabel("Compressed File Size (Bytes)")
    plt.ylabel("Compression Time (Seconds)")
    plt.title("Compression Time vs File Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Run the experiment and plot results
def run_audio_compression_experiment_with_scatter(file_path):
    results = {}
    original_size = os.path.getsize(file_path)

    print(f"Original WAV file size: {original_size} bytes\n")

    # LZMA
    results['LZMA'] = compress_lzma(file_path)

    # ZIP
    results['ZIP'] = compress_zip(file_path)

    # GZIP
    results['GZIP'] = compress_gzip(file_path)

    # BZIP2
    results['BZIP2'] = compress_bzip2(file_path)

    # Deflate
    deflate_size, deflate_duration = compress_deflate(file_path)
    results['Deflate'] = (deflate_size, deflate_duration)

    # Plot results including scatter plot for time vs size
    plot_results_with_rar_and_scatter(original_size, results)


# Main Execution
if __name__ == "__main__":
    wav_file = "爱在西元前.wav"

    # Check if WAV file exists
    if not os.path.exists(wav_file):
        print(f"Error: {wav_file} does not exist!")
    else:
        # Run compression experiment with updated plotting
        run_audio_compression_experiment_with_scatter(wav_file)
