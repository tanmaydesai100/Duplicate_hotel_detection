import os
import psutil
import pandas as pd

def print_cpu_info():
    n_cpus = os.cpu_count() or 1
    print(f"→ Logical CPU cores available: {n_cpus}")

def print_ram_info():
    vm = psutil.virtual_memory()
    total_gb     = vm.total    / (1024**3)
    available_gb = vm.available / (1024**3)
    used_gb      = (vm.total - vm.available) / (1024**3)
    print(f"→ RAM total:     {total_gb:6.2f} GB")
    print(f"→ RAM available: {available_gb:6.2f} GB")
    print(f"→ RAM used:      {used_gb:6.2f} GB")

def print_gpu_info():
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        print(f"→ PyTorch reports CUDA available? {has_cuda}")
        if has_cuda:
            print(f"   • GPU name (device 0): {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("→ PyTorch not installed; skipping GPU check.")

def print_dataframe_sizes(pathA, pathB):
    """
    Load each CSV in “memory‐estimate” mode (without full parsing), then compute 
    memory_usage once loaded. This shows you roughly how many MB each DF uses.
    """
    print(f"\nLoading and measuring memory for:\n  • {pathA}\n  • {pathB}")
    # Load with pandas (it may take a few seconds)
    dfA = pd.read_csv(pathA)
    dfB = pd.read_csv(pathB)

    sizeA_mb = dfA.memory_usage(deep=True).sum() / (1024**2)
    sizeB_mb = dfB.memory_usage(deep=True).sum() / (1024**2)

    print(f"→ DataFrame A ({os.path.basename(pathA)}) size: {sizeA_mb:6.1f} MB")
    print(f"→ DataFrame B ({os.path.basename(pathB)}) size: {sizeB_mb:6.1f} MB")

if __name__ == "__main__":
    # 1) CPU cores
    print_cpu_info()
    # 2) RAM info
    print_ram_info()
    # 3) GPU info (if you have torch installed)
    print_gpu_info()

    # 4) If you already know the two CSV paths, fill them here:
    fileA = "kaggle/hotel_with_id.csv"
    fileB = "ukDataset/dataset2_final.csv"

    # Check that the files exist before trying to read:
    missing = [p for p in (fileA, fileB) if not os.path.isfile(p)]
    if missing:
        print(f"\nERROR: The following files do not exist:\n  {missing}\n"
              "Please adjust the paths before running.")
    else:
        print_dataframe_sizes(fileA, fileB)
