import os
import time
import lzma
import zipfile
import gzip
import zlib
from PIL import Image
import matplotlib.pyplot as plt

def measure_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

# Compression algorithms
def compress_lzma(file_path):
    """LZMA compression."""
    with open(file_path, 'rb') as f:
        data = f.read()
    compressed_data, duration = measure_time(lzma.compress, data)
    return len(compressed_data), duration

def compress_zip(file_path):
    """ZIP compression."""
    compressed_file = file_path + '.zip'
    start = time.time()
    with zipfile.ZipFile(compressed_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, arcname=os.path.basename(file_path))
    end = time.time()
    compressed_size = os.path.getsize(compressed_file)
    return compressed_size, end - start

def compress_gzip(file_path):
    """GZIP compression."""
    compressed_file = file_path + '.gz'
    start = time.time()
    with open(file_path, 'rb') as f_in, gzip.open(compressed_file, 'wb') as f_out:
        while chunk := f_in.read(1024 * 1024):
            f_out.write(chunk)
    end = time.time()
    compressed_size = os.path.getsize(compressed_file)
    return compressed_size, end - start

def compress_deflate(file_path):
    """Deflate compression."""
    with open(file_path, 'rb') as f:
        data = f.read()
    compressed_data, duration = measure_time(zlib.compress, data)
    return len(compressed_data), duration

def compress_lzw(data):
    """LZW compression."""
    dictionary = {bytes([i]): i for i in range(256)}
    current = b""
    compressed_data = []
    code = 256

    for byte in data:
        current_with_byte = current + bytes([byte])
        if current_with_byte in dictionary:
            current = current_with_byte
        else:
            compressed_data.append(dictionary[current])
            if code < 4096:
                dictionary[current_with_byte] = code
                code += 1
            current = bytes([byte])

    if current:
        compressed_data.append(dictionary[current])

    output = bytearray()
    for value in compressed_data:
        output.append((value >> 8) & 0xFF)
        output.append(value & 0xFF)
    return output

def compress_lzw_file(file_path):
    """LZW-based file compression."""
    with open(file_path, 'rb') as f:
        data = f.read()
    start = time.time()
    compressed_data = compress_lzw(data)
    end = time.time()
    return len(compressed_data), end - start

def compress_png(input_path, output_path):
    """PNG compression."""
    img = Image.open(input_path)
    start = time.time()
    img.save(output_path, format='PNG', optimize=True)
    end = time.time()
    return os.path.getsize(output_path), end - start

# Plot comparison chart with RAR
def plot_compression_results_with_rar(original_size, compressed_sizes):
    """Plot comparison of original and compressed file sizes, including RAR."""
    algorithms = list(compressed_sizes.keys())
    sizes = [original_size] + list(compressed_sizes.values()) + [67235245]  # Add RAR size
    labels = ['Original'] + algorithms + ['RAR']

    # Assign different colors for each bar
    colors = ['gray', 'blue', 'green', 'red', 'purple', 'orange', 'cyan', 'gold']

    plt.figure(figsize=(12, 6))
    plt.bar(labels, sizes, color=colors[:len(labels)], edgecolor='black', alpha=0.7)
    plt.xlabel("Compression Methods")
    plt.ylabel("Total Size (Bytes)")
    plt.title("Original vs Compressed File Sizes (Including RAR)")
    for i, size in enumerate(sizes):
        plt.text(i, size + 500, f"{size}", ha='center')
    plt.tight_layout()
    plt.show()

# Main function
def run_experiment(input_dir):
    """Perform compression experiments on the dataset using multiple algorithms."""
    algorithms = {
        'LZMA': compress_lzma,
        'ZIP': compress_zip,
        'GZIP': compress_gzip,
        'Deflate': compress_deflate,
        'LZW': compress_lzw_file,
        'PNG': compress_png,
    }
    
    # Calculate total size of original files
    original_total_size = sum(os.path.getsize(os.path.join(input_dir, file)) for file in os.listdir(input_dir) if file.endswith('.ppm'))
    print(f"Total original file size: {original_total_size} bytes")
    
    compressed_sizes = {}  # Store total compressed sizes for each algorithm

    for alg_name, alg_func in algorithms.items():
        print(f"\nCompressing with {alg_name}...")
        
        total_compressed_size = 0
        total_compression_time = 0
        
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.ppm'):
                file_path = os.path.join(input_dir, file_name)
                if alg_name == 'PNG':
                    output_path = os.path.join(input_dir, file_name.replace('.ppm', '.png'))
                    compressed_size, compression_time = alg_func(file_path, output_path)
                else:
                    compressed_size, compression_time = alg_func(file_path)
                
                total_compressed_size += compressed_size
                total_compression_time += compression_time
        
        # Save results
        compressed_sizes[alg_name] = total_compressed_size

        # Print results
        print(f"{alg_name} compression completed!")
        print(f"Total compression time: {total_compression_time:.2f} seconds")
        print(f"Total compressed file size: {total_compressed_size} bytes")

    # Plot results including RAR
    plot_compression_results_with_rar(original_total_size, compressed_sizes)

# Run the experiment
input_folder = "ppmXrays"
if not os.path.exists(input_folder):
    print(f"Input folder '{input_folder}' does not exist.")
else:
    run_experiment(input_folder)
