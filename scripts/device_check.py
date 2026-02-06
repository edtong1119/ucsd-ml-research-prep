import torch
import time

def run_smoke_test():
    print("-" * 30)
    print("ML ENGINEER SMOKE TEST (UNIVERSAL)")
    print("-" * 30)

    # 1. Logic: Device Detection
    # We check for NVIDIA (cuda) first, then Apple (mps), then default to cpu.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Selected Device: {device} ({device_name})")

    # 2. Logic: Benchmark Preparation
    size = 5000
    print(f"\n[1/2] Benchmarking CPU...")
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    
    start = time.time()
    _ = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time:.4f}s")

    # 3. Logic: Accelerator Benchmark
    if device.type != "cpu":
        print(f"[2/2] Benchmarking {device.type.upper()}...")
        
        # Move data to the accelerator
        x_acc = x_cpu.to(device)
        y_acc = y_cpu.to(device)

        # WARM-UP: The first run on a GPU/MPS is often slow due to setup
        _ = torch.matmul(x_acc, y_acc)
        
        # SYNCHRONIZATION: Wait for the hardware to actually finish
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        start = time.time()
        _ = torch.matmul(x_acc, y_acc)
        
        # Synchronize again before stopping the clock
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
            
        acc_time = time.time() - start
        print(f"{device.type.upper()} Time: {acc_time:.4f}s")
        print(f"Speedup: {cpu_time/acc_time:.1f}x faster than CPU!")
    else:
        print("\n[2/2] Skipping Accelerator test (No GPU/MPS found).")

if __name__ == "__main__":
    run_smoke_test()