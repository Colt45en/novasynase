import numpy as np
import matplotlib.pyplot as plt
import json
import os
from flask import Flask

class NovaSynapse:
    def __init__(self, data_size=1000, memory_file="ai_memory.json"):
        """Initialize NovaSynapse with random data and memory handling."""
        self.original_data = np.random.rand(data_size)
        self.compressed_data = None
        self.compression_ratios = []
        self.memory_file = memory_file
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

    def compress(self, factor=2):
        """Compress data using a simple downsampling method."""
        if factor < 1 or factor > len(self.original_data):
            raise ValueError("Compression factor must be between 1 and the length of data.")
        
        self.compressed_data = self.original_data[::factor]
        compression_ratio = len(self.compressed_data) / len(self.original_data)
        self.compression_ratios.append(compression_ratio)
        print(f"Compression Ratio: {compression_ratio:.4f}")
        return compression_ratio

    def save_memory(self):
        """Save compression performance metrics to memory file."""
        try:
            data = {"compression_ratios": self.compression_ratios}
            with open(self.memory_file, "w") as file:
                json.dump(data, file)
            print("Memory saved successfully.")
        except Exception as e:
            print(f"Error saving memory: {e}")

    def visualize(self):
        """Plot compression ratio performance over multiple iterations."""
        if not self.compression_ratios:
            print("No compression data available to plot.")
            return
        plt.figure(figsize=(8, 5))
        plt.plot(self.compression_ratios, label='Compression Ratio', marker='o', linestyle='--')
        plt.xlabel("Iterations")
        plt.ylabel("Compression Ratio")
        plt.title("Compression Performance Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

app = Flask(__name__)

@app.route('/')
def home():
    return "NovaSynapse Flask API is running!"

@app.route('/compress')
def compress_data():
    nova = NovaSynapse(1000)
    compression_ratio = nova.compress()
    nova.save_memory()
    return {"message": "Compression performed", "compression_ratio": compression_ratio}

if __name__ == "__main__":
    app.run(debug=True)
