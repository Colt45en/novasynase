import zlib
import numpy as np
import matplotlib.pyplot as plt
import os # Add this import statement
import json
from flask import Flask, jsonify, request
from sklearn.decomposition import PCA

class NovaSynapse:
    def __init__(self, data_size=1000, memory_file="memory.json", compression_method="zlib"):
        """Initialize NovaSynapse with random data and selected compression method."""
        self.original_data = np.random.rand(data_size)
        self.memory_file = memory_file
        self.compressed_data = None
        self.compression_method = compression_method
        self.compression_ratios = []
        self.load_memory()

    def load_memory(self):
        """Load stored compression ratios from memory file if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as file:
                    data = json.load(file)
                    self.compression_ratios = data.get("compression_ratios", [])
            except json.JSONDecodeError:
                print("Memory file is corrupted. Starting fresh.")
        else:
            print("No previous memory found. Starting fresh.")

    def compress_data(self, factor=2):
        """Compress data using PCA or simple downsampling based on method."""
        if self.compression_method == "pca":
            pca = PCA(n_components=factor)
            self.compressed_data = pca.fit_transform(self.original_data.reshape(-1, 1))
        elif self.compression_method == "zlib":
            self.compressed_data = zlib.compress(self.original_data.tobytes())
        else:
            raise ValueError("Invalid compression method. Choose 'pca' or 'zlib'.")
        
        if self.compression_method == "zlib":
            compression_ratio = len(self.compressed_data) / len(self.original_data)
        else:
            compression_ratio = factor / len(self.original_data)
        
        self.compression_ratios.append(compression_ratio)
        print(f"Compression Ratio: {compression_ratio:.4f}")
        return compression_ratio

    def decompress_data(self):
        """Decompress data based on the selected compression method."""
        if self.compressed_data is None:
            return None
        if self.compression_method == "pca":
            print("PCA compression is not reversible without original PCA components.")
            return None
        elif self.compression_method == "zlib":
            return np.frombuffer(zlib.decompress(self.compressed_data), dtype=np.float64)
        else:
            raise ValueError("Invalid compression method.")

    def save_memory(self):
        """Save compression performance metrics to memory file."""
        try:
            data = {"compression_ratios": self.compression_ratios}
            with open(self.memory_file, "w") as file:
                json.dump(data, file)
            print("Memory saved successfully.")
        except Exception as e:
            print(f"Error saving memory: {e}")

    def plot_compression_performance(self):
        """Plot compression ratio performance over multiple iterations."""
        if not self.compression_ratios:
            print("No compression data available to plot.")
            return
        plt.plot(self.compression_ratios, label='Compression Ratio', marker='o', linestyle='--')
        plt.xlabel("Iterations")
        plt.ylabel("Compression Ratio")
        plt.title("Compression Performance Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

# Flask API for NovaSynapse
the_app = Flask(__name__)

@the_app.route('/')
def home():
    return "NovaSynapse Flask API is running!"

@the_app.route('/compress', methods=['POST'])
def compress_data():
    data_size = request.json.get('data_size', 1000)
    compression_method = request.json.get('compression_method', 'zlib')
    factor = request.json.get('factor', 2)
    
    nova = NovaSynapse(data_size, compression_method=compression_method)
    compression_ratio = nova.compress_data(factor=factor)
    nova.save_memory()
    
    return jsonify({"message": "Compression performed", "compression_ratio": compression_ratio})

@the_app.route('/decompress', methods=['POST'])
def decompress_data():
    data_size = request.json.get('data_size', 1000)
    compression_method = request.json.get('compression_method', 'zlib')
    
    nova = NovaSynapse(data_size, compression_method=compression_method)
    decompressed_data = nova.decompress_data()
    
    if decompressed_data is not None:
        return jsonify({"message": "Decompression successful", "decompressed_data": decompressed_data.tolist()})
    return jsonify({"error": "Decompression failed or no data to decompress."})

@the_app.route('/visualize', methods=['GET'])
def visualize():
    data_size = int(request.args.get('data_size', 1000))
    compression_method = request.args.get('compression_method', 'zlib')
    
    nova = NovaSynapse(data_size, compression_method=compression_method)
    nova.plot_compression_performance()
    
    return jsonify({"message": "Visualization displayed."})

if __name__ == "__main__":
    the_app.run(debug=True)
