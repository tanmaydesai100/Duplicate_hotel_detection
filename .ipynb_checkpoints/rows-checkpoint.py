import csv

# Use a more permissive encoding:
file_path = "ukDatasetKaggle/uk_hotels.csv"

with open(file_path, mode="r", encoding="latin-1", newline="") as f:
    reader = csv.reader(f)
    row_count = sum(1 for _ in reader)

print(f"Number of rows: {row_count}")

# import os, psutil

# print("CPU cores:", os.cpu_count())

# ram = psutil.virtual_memory()
# print(f"Total RAM: {ram.total / (1024**3):.2f} GB")
# print(f"Available RAM: {ram.available / (1024**3):.2f} GB")

# try:
#     import torch
#     print("GPU Available:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         print("GPU Name:", torch.cuda.get_device_name(0))
# except ImportError:
#     print("PyTorch not installed – skipping GPU check")

