import torch
import time
import csv
from datetime import datetime

def check_gpus():
    """
    Checks the availability and functionality of all GPUs on the system.

    Returns:
        List of indices of available and functioning GPUs.
    """
    available_gpus = []
    unavailable_gpus = []

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("No GPU available or CUDA not properly initialized!")
        return available_gpus, unavailable_gpus

    # Get the number of GPUs
    try:
        num_gpus = torch.cuda.device_count()
        print(f"{num_gpus} GPU(s) detected:")
    except RuntimeError as e:
        print(f"CUDA Runtime Error: {e}")
        return available_gpus, unavailable_gpus

    for i in range(num_gpus):
        device_name = torch.cuda.get_device_name(i)
        try:
            # Perform a simple operation to ensure the GPU is functional
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.randn(1, device=f"cuda:{i}")  # Simple operation
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"GPU {i}: {device_name} is functioning correctly.")
                print(f"  - Total Memory: {gpu_memory:.2f} GB")
                available_gpus.append(i)
        except Exception as e:
            print(f"GPU {i}: {device_name} encountered an error: {e}")
            unavailable_gpus.append(i)

    return available_gpus, unavailable_gpus

def gpu_stats(device_id):
    """
    Retrieves detailed statistics for a specific GPU.

    Args:
        device_id (int): The index of the GPU.

    Returns:
        A dictionary containing memory usage, power usage, and temperature.
    """
    memory_allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    memory_cached = torch.cuda.memory_reserved(device_id) / 1024**3

    # Initialize placeholders
    power_usage = None
    temperature = None

    try:
        # Retrieve power usage and temperature using nvidia-smi
        power_usage = torch.cuda._C._cuda_get_power_usage(device_id) / 1000
        temperature = torch.cuda._C._cuda_get_temperature(device_id)
    except AttributeError:
        pass  # These functions may not be available in PyTorch, handle gracefully

    return {
        'Memory Allocated (GB)': memory_allocated,
        'Memory Cached (GB)': memory_cached,
        'Power Usage (W)': power_usage,
        'Temperature (C)': temperature
    }

def run_benchmark(gpus, matrix_size=4096):
    """
    Runs a detailed matrix multiplication benchmark on all available GPUs and saves the results to a CSV file.

    Args:
        gpus (list): List of GPU indices to benchmark.
        matrix_size (int): The size of the square matrices used in the benchmark.
    """
    results = []

    for i in gpus:
        print(f"Running benchmark on GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device(f"cuda:{i}")

        # Create two large random matrices on the current GPU
        mat1 = torch.randn((matrix_size, matrix_size), device=device)
        mat2 = torch.randn((matrix_size, matrix_size), device=device)

        # Warm-up to ensure the GPU is ready for the benchmark
        torch.matmul(mat1, mat2)
        torch.cuda.synchronize(device)

        # Capture statistics before running the benchmark
        stats_before = gpu_stats(i)

        # Measure the time taken for matrix multiplication
        start_time = time.time()
        torch.matmul(mat1, mat2)
        torch.cuda.synchronize(device)
        end_time = time.time()

        # Capture statistics after the benchmark
        stats_after = gpu_stats(i)

        elapsed_time = end_time - start_time
        print(f"Elapsed time on GPU {i}: {elapsed_time:.4f} seconds")

        # Store the results in a dictionary for later CSV export
        result = {
            'GPU Index': i,
            'GPU Name': torch.cuda.get_device_name(i),
            'Elapsed Time (s)': elapsed_time,
            'Memory Allocated Before (GB)': stats_before['Memory Allocated (GB)'],
            'Memory Cached Before (GB)': stats_before['Memory Cached (GB)'],
            'Power Usage Before (W)': stats_before['Power Usage (W)'],
            'Temperature Before (C)': stats_before['Temperature (C)'],
            'Memory Allocated After (GB)': stats_after['Memory Allocated (GB)'],
            'Memory Cached After (GB)': stats_after['Memory Cached (GB)'],
            'Power Usage After (W)': stats_after['Power Usage (W)'],
            'Temperature After (C)': stats_after['Temperature (C)'],
        }
        results.append(result)

    # Save the collected benchmark results to a CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'gpu_benchmark_results_{timestamp}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Benchmark results saved to {filename}")

if __name__ == "__main__":
    # Check for available GPUs and run the benchmark if any are found
    available_gpus, unavailable_gpus = check_gpus()

    if available_gpus:
        print("\nSummary of GPU availability:")
        print(f"Available GPUs: {available_gpus}")
        print(f"Unavailable GPUs: {unavailable_gpus}")
        run_benchmark(available_gpus)
    else:
        print("No GPUs available for benchmarking.")
