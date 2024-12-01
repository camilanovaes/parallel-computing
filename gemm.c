#include <bits/time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <threads.h>
#include <time.h>

#define N 1024

float A[N][N];
float B[N][N];
float C[N][N];

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1e9 + (uint64_t)start.tv_nsec;
}

void matmul_naive() {
  // super simple matrix multiplication (not optimized at all)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float tmp = 0;
      for (int k = 0; k < N; k++) {
        tmp += A[i][k] * B[k][j];
      }
      C[i][j] = tmp;
    }
  }
}

int main() {
  // init A and B with random values
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = rand() % 10;
      B[i][j] = rand() % 10;
      C[i][j] = 0;
    }
  }

  // run naive method
  uint64_t s = nanos();
  matmul_naive();
  uint64_t e = nanos();

  double gflops = N * N * 2.0 * N * 1e-9;
  double t = (e - s) * 1e-9; // seconds
  printf("%f GFLOP/s", gflops / t);
}

// Result:
// Naive: 0.71 GFLOP/s
