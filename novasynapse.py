import numpy as np
import matplotlib.pyplot as plt
import json

class NovaSynapse:
    def __init__(self, data_size):
        self.original_data = np.random.rand(data_size)
        self.compressed_data = self.original_data.copy()
        self.compression_ratios = []
        self.memory_file = "ai_memory.json"
        self.load_memory()

    def load_memory(self):
        try:
            with open(self.memory_file, "r") as file:
                data = json.load(file)
                self.compression_ratios = data.get("compression_ratios", [])
        except FileNotFoundError:
            print("No previous memory found. Starting fresh.")

    def compress(self):
        self.compressed_data = self.original_data[::2]
        compression_ratio = len(self.compressed_data) / len(self.original_data)
        self.compression_ratios.append(compression_ratio)
        return compression_ratio

    def save_memory(self):
        data = {"compression_ratios": self.compression_ratios}
        with open(self.memory_file, "w") as file:
            json.dump(data, file)
        print("Memory saved.")

    def visualize(self):
        plt.plot(self.compression_ratios, label='Compression Ratio', marker='o')
        plt.xlabel("Iterations")
        plt.ylabel("Compression Ratio")
        plt.title("Compression Performance")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    nova = NovaSynapse(1000)
    nova.compress()
    nova.save_memory()
    nova.visualize()
