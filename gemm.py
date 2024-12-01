import os
import time

# os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

N = 4098

if __name__ == "__main__":
    while True:
        A = np.random.randn(N, N).astype(np.float32)
        B = np.random.randn(N, N).astype(np.float32)

        s = time.monotonic()
        C = A @ B
        e = time.monotonic()

        flops = N * N * 2 * N  # rows * cols * sum items
        t = e - s
        print(f"{flops/t * 1e-9:.2f} GFLOP/s")
        time.sleep(0.5)


# Results:
# 1.07 TFLOP/s using Ryzen 9 5900x (all threads)
# 135.7 GFLOP/s using Ryzen 9 5900x (1 threads)
