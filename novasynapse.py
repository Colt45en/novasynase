import zlib
import numpy as np

class NovaSynapse:
    def __init__(self, original_data):
        self.original_data = original_data
        self.compressed_data = None

    def compress_data(self):
        compressed_data = zlib.compress(self.original_data.tobytes())
        self.compressed_data = compressed_data

    def decompress_data(self):
        decompressed_data = np.frombuffer(zlib.decompress(self.compressed_data), dtype=np.float64)
        return decompressed_data
def _intel__(self, data_size):
    """
    Initializes the NovaSynapse instance.
    :param data_size: The number of data points to generate.
    """
    if data_size <= 0:
        raise ValueError("data_size must be a positive integer.")
    self.original_data = np.random.rand(data_size)
    self.compressed_data = None
    self.compression_ratios = []
    self.memory_file = "ai_memory.json"
    self.load_memory()

def load_memory(self):
    """
    Loads previously saved compression ratio data from memory.
    """
    if os.path.exists(self.memory_file):
        try:
            with open(self.memory_file, "r") as file:
                data = json.load(file)
                self.compression_ratios = data.get("compression_ratios", [])
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading memory: {e}")
    else:
        print("No previous memory found. Starting fresh.")

def compress(self, method='zlib', components=5):
    """
    Compresses data using the specified method.
    :param method: Compression method ('zlib' or 'pca').
    :param components: Number of components to retain for PCA compression.
    :return: Compression ratio.
    """
    if method == 'zlib':
        compressed_data = zlib.compress(self.original_data.tobytes())
        compression_ratio = len(compressed_data) / self.original_data.nbytes
    elif method == 'pca':
        pca = PCA(n_components=components)
        compressed_data = pca.fit_transform(self.original_data.reshape(-1, 1))
        compression_ratio = components / len(self.original_data)
    else:
        raise ValueError("Unsupported compression method.")
    
    self.compressed_data = compressed_data
    self.compression_ratios.append(compression_ratio)
    return compression_ratio

def decompress(self, method='zlib', components=5):
    """
    Decompresses data based on the previously used compression method.
    :param method: Decompression method ('zlib' or 'pca').
    :param components: Number of components used in PCA compression.
    :return: Decompressed data.
    """
    if self.compressed_data is None:
        raise ValueError("No data has been compressed yet.")

    if method == 'zlib':
        decompressed_data = np.frombuffer(zlib.decompress(self.compressed_data), dtype=np.float64)
    elif method == 'pca':
        pca = PCA(n_components=components)
        decompressed_data = pca.inverse_transform(self.compressed_data).flatten()
    else:
        raise ValueError("Unsupported decompression method.")
    
    return decompressed_data

def save_memory(self):
    """
    Saves the current compression ratios to memory.
    """
    data = {"compression_ratios": self.compression_ratios}
    try:
        with open(self.memory_file, "w") as file:
            json.dump(data, file)
        print("Memory saved.")
    except IOError as e:
        print(f"Error saving memory: {e}")

def visualize(self):
    """
    Visualizes the compression performance over time.
    """
    plt.plot(self.compression_ratios, label='Compression Ratio', marker='o')
    plt.xlabel("Iterations")
    plt.ylabel("Compression Ratio")
    plt.title("Compression Performance")
    plt.legend()
    plt.show()
