import torch
import time
import logging

# 确保至少有一个GPU设备
assert torch.cuda.device_count() >= 1, "Need at least one GPU available."

# 创建设备对象
device = torch.device('cuda:0')
device_cpu = torch.device('cpu')

# Benchmark parameters
batch_sizes = [1,4,8,16]  # Batch sizes from 1 to 16
sequence_lengths = [64, 128, 256, 512, 1024, 2048]  # Different sequence lengths
embedding_dims = [4096, 7168]  # Different embedding dimensions
num_tensors = [4, 8, 16,32]  # Different numbers of tensors to transfer
num_repeats = 20  # Number of repetitions for each experiment

# Set up logging
logging.basicConfig(filename='benchmarkIO_in8_c2g_1g.txt', level=logging.INFO, 
                    format='%(message)s')  # Change the filename as needed

def copy_to_gpu(tensors_on_cpu, tensors_on_gpu):
    for i in range(len(tensors_on_cpu)):
        tensors_on_gpu[i].copy_(tensors_on_cpu[i], non_blocking=True)

for bs in batch_sizes:
    for seq_len in sequence_lengths:
        for emb_dim in embedding_dims:
            for num_tens in num_tensors:
                total_time_taken = 0
                total_bandwidth = 0

                for repeat in range(num_repeats):
                    # Initialize tensors on CPU
                    tensors_on_cpu = [torch.randint(-127, 128, (bs, seq_len, emb_dim), dtype=torch.int8, device=device_cpu) for _ in range(num_tens)]

                    # Pre-allocate memory on the GPU
                    tensors_on_gpu = [torch.empty_like(tensor, device=device) for tensor in tensors_on_cpu]

                    # Compute total data size (GB)
                    total_data_size_GB = num_tens * bs * seq_len * emb_dim * torch.tensor(1).item() / (1024 ** 3)  # 1 for bytes of int8

                    # Time the copy from CPU to the GPU
                    start = time.time()

                    copy_to_gpu(tensors_on_cpu, tensors_on_gpu)

                    torch.cuda.synchronize()  # Ensure all copy operations are complete

                    end = time.time()
                    time_taken = end - start  # Time taken for the operation
                    bandwidth = total_data_size_GB / time_taken  # Compute bandwidth

                    total_time_taken += time_taken
                    total_bandwidth += bandwidth

                avg_time_taken = total_time_taken / num_repeats
                avg_bandwidth = total_bandwidth / num_repeats

                # Log the result
                log_message = f"Batch size: {bs}, Sequence length: {seq_len}, Embedding dimension: {emb_dim}, Number of tensors: {num_tens}, Average Time taken: {avg_time_taken:.4f}s, Average Bandwidth: {avg_bandwidth:.2f} GB/s"
                print(log_message)  # Print to console
                logging.info(log_message)  # Write to log file
