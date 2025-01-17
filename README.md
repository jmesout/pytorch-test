# PyTorch GPU Benchmark Pod

A Kubernetes Pod specification that launches a PyTorch-based GPU benchmark. It performs:
1. **System and Kubernetes environment checks**  
2. **NVIDIA GPU checks** (via `nvidia-smi` and PyTorch)  
3. **Mixed-precision matrix multiplication** (if supported)  
4. **FP32 matrix multiplication benchmark**  
5. **Simple CNN training benchmark** on random data  

## How to Use

### Deploy via GitHub URL

You can apply this Pod spec directly from the GitHub repository:

```bash
kubectl apply -f https://github.com/jmesout/pytorch-test/raw/main/pytorch-test.yaml
```

This will create a Pod named gpu-benchmark-pod in your current namespace. Then, check its logs:

```bash
kubectl logs -f gpu-benchmark-pod
```

To download your benchmarks you can simply use:
```bash
kubectl cp gpu-benchmark-pod:/gpu_benchmark_results_matmul_<TIMESTAMP>.csv .
kubectl cp gpu-benchmark-pod:/gpu_benchmark_results_cnn_<TIMESTAMP>.csv .
```

### Overriding the Benchmark size
By default, the YAML sets TEST_SIZE="small". You can edit the Pod’s environment variable before or after applying the file to change the scale of the test.
```yaml
- name: TEST_SIZE
  value: "medium"
```

If you wish to override the test size on apply, you may do so on a linux machine using curl and sed. 

```bash
curl -s https://github.com/jmesout/pytorch-test/raw/main/pytorch-test.yaml \
  | sed 's/value: "small"/value: "large"/g' \
  | kubectl apply -f -
```

### Example output
Here is an example output of the scripts output.

```
Created /tmp/gpu_benchmark.py. Now running it...

=== Kubernetes & Environment Variables ===
Node Name:       <NODE NAME>
Pod Name:        gpu-benchmark-pod
Pod Namespace:   default
Pod IP:          <POD IP HERE>
Host IP:         <HOST IP HERE>
Container Hostname (HOSTNAME): gpu-benchmark-pod
TEST_SIZE:       small
==========================================

=== System & Environment Information ===
Platform:           <LINUX PLATFORM INFO>
Python Version:     <PYTHON VERSION>
PyTorch Version:    <TORCH VERSION>
CUDA Available:     True
CUDA Version:       <CUDA VERSION>
cuDNN Enabled:      True
cuDNN Version:      <CUDNN VERSION>
========================================

=== Checking NVIDIA System Management Interface (nvidia-smi) ===
Driver Version: <DRIVER VERSION>
GPU 0:
  - Name:       <GPU DEVICE HERE>
  - Total Mem:  <TOTAL MEMORY MB HERE>
===============================================================

Checking GPU availability and basic functionality...
Number of GPUs detected: 1

GPU 0: <GPU DEVICE HERE>
  - Device Name:         <GPU DEVICE HERE>
  - Total Memory (GB):   <TOTAL MEMORY IN GB HERE>
  - Compute Capability:  <MAJOR>.<MINOR>

=== Summary of GPU Checks ===
Available GPUs:   [0]
Unavailable GPUs: []

=== Mixed Precision Test (GPU 0) ===
Matrix size: 4096x4096, Iterations: 10
Mixed Precision matmul completed on GPU 0.
Average time per iteration: 0.012345 s
Std Dev: 0.003210 s
GPU 0: Temp before: None, after: None

=== Matrix Multiplication Benchmark (FP32) ===
Matrix size: 4096x4096
Number of iterations per GPU: 10

Running matmul benchmark on GPU 0: <GPU DEVICE HERE>
  Iteration 1/10 time: 0.002851 seconds
  Iteration 2/10 time: 0.002697 seconds
  ...
  Average Time (s): 0.002703
  Std Dev (s):      0.000049
GPU 0: Temp before: None, after: None

Benchmark results saved to gpu_benchmark_results_matmul_<TIMESTAMP>.csv

=== CNN Training Benchmark ===
Image Size: 32x32, Channels: 3, Classes: 10
Batch Size: 64, Batches per Epoch: 50, Epochs: 2

Running CNN benchmark on GPU 0: <GPU DEVICE HERE>
  Epoch 1/2 time: 0.219786 seconds
  Epoch 2/2 time: 0.056953 seconds
  Average Epoch Time (s): 0.138370
  Std Dev (s):            0.081416
GPU 0: Temp before: None, after: None

Benchmark results saved to gpu_benchmark_results_cnn_<TIMESTAMP>.csv
```



