import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Define the model serving function
def serve_model(batch_size, max_concurrency):
    # Simulate model serving
    time.sleep(0.1 * batch_size)  # Assume 0.1 seconds per sample
    return batch_size

# Define the benchmark function
def run_benchmark(batch_sizes, max_concurrency, num_requests):
    latencies = []
    throughputs = []

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        for batch_size in batch_sizes:
            start_time = time.time()
            futures = [executor.submit(serve_model, batch_size, max_concurrency) for _ in range(num_requests)]
            for future in futures:
                future.result()
            end_time = time.time()

            total_time = end_time - start_time
            total_samples = num_requests * batch_size
            latency = total_time / num_requests
            throughput = total_samples / total_time

            latencies.append(latency)
            throughputs.append(throughput)

    return latencies, throughputs




def main():
    # Example usage
    batch_sizes = [1, 8, 16, 32]
    max_concurrency = 8
    num_requests = 100

    latencies, throughputs = run_benchmark(batch_sizes, max_concurrency, num_requests)

    print("Batch Size\tLatency (s)\tThroughput (samples/s)")
    for i in range(len(batch_sizes)):
        print(f"{batch_sizes[i]}\t{latencies[i]:.4f}\t{throughputs[i]:.2f}")


if __name__ == "__main__":
    main()