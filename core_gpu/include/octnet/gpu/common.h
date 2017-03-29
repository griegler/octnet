// Copyright (c) 2017, The OctNet authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL OCTNET AUTHORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#ifndef COMMON_H
#define COMMON_H

#include <cublas_v2.h>

#include <cstdio>

#define CUDA_DEBUG_DEVICE_SYNC 0

/// Gets the number of free and available bytes for the current GPU device.
/// @param free_byte
/// @param total_byte
inline void getCudaMemUsage(size_t* free_byte, size_t* total_byte) {
  cudaError_t stat = cudaMemGetInfo(free_byte, total_byte);
  if(stat != cudaSuccess) {
    printf("[ERROR] failed to query cuda memory (%f,%f) information, %s\n", *free_byte/1024.0/1024.0, *total_byte/1024.0/1024.0, cudaGetErrorString(stat));
    exit(-1);
  }
}

/// Prints the current memory usage of the current GPU device to stdout.
inline void printCudaMemUsage() {
  size_t free_byte, total_byte;
  getCudaMemUsage(&free_byte, &total_byte);
  printf("[INFO] CUDA MEM Free=%f[MB], Used=%f[MB]\n", free_byte/1024.0/1024.0, total_byte/1024.0/1024.0);
}

// cuda check for cudaMalloc and so on
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    if(CUDA_DEBUG_DEVICE_SYNC) { cudaDeviceSynchronize(); } \
    cudaError_t error = condition; \
    if(error != cudaSuccess) { \
      printf("%s in %s at %d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
      exit(-1); \
    } \
  } while (0)


/// Get error string for error code.
/// @param error
inline const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unknown cublas status";
}

#define CUBLAS_CHECK(condition) \
  do { \
    if(CUDA_DEBUG_DEVICE_SYNC) { cudaDeviceSynchronize(); } \
    cublasStatus_t status = condition; \
    if(status != CUBLAS_STATUS_SUCCESS) { \
      printf("%s in %s at %d\n", cublasGetErrorString(status), __FILE__, __LINE__); \
      exit(-1); \
    } \
  } while (0)


// check if there is a error after kernel execution
#define CUDA_POST_KERNEL_CHECK \
  CUDA_CHECK(cudaPeekAtLastError()); \
  CUDA_CHECK(cudaGetLastError()); 

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS_T(const int N, const int N_THREADS) {
  return (N + N_THREADS - 1) / N_THREADS;
}
inline int GET_BLOCKS(const int N) {
  return GET_BLOCKS_T(N, CUDA_NUM_THREADS);
  // return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}




/// Allocate memory on the current GPU device.
/// @param N lenght of the array.
/// @return pointer to device array.
template<typename T>
T* device_malloc(long N) {
  T* dptr;
  CUDA_CHECK(cudaMalloc(&dptr, N * sizeof(T)));
  if(DEBUG) { printf("[DEBUG] device_malloc %p, %ld\n", dptr, N); }
  return dptr;
}

/// Frees device memory.
/// @param dptr
template<typename T>
void device_free(T* dptr) {
  if(DEBUG) { printf("[DEBUG] device_free %p\n", dptr); }
  CUDA_CHECK(cudaFree(dptr));
}

/// Copies host to device memory.
/// @param hptr
/// @param dptr
/// @param N
template<typename T>
void host_to_device(const T* hptr, T* dptr, long N) {
  if(DEBUG) { printf("[DEBUG] host_to_device %p => %p, %ld\n", hptr, dptr, N); }
  CUDA_CHECK(cudaMemcpy(dptr, hptr, N * sizeof(T), cudaMemcpyHostToDevice));
}

/// Copies host memory the newly allocated device memory.
/// @param hptr
/// @param N
/// @return dptr
template<typename T>
T* host_to_device_malloc(const T* hptr, long N) {
  T* dptr = device_malloc<T>(N);
  host_to_device(hptr, dptr, N);
  return dptr;
}

/// Copies device to host memory.
/// @param dptr
/// @param hptr
/// @param N
template<typename T>
void device_to_host(const T* dptr, T* hptr, long N) {
  if(DEBUG) { printf("[DEBUG] device_to_host %p => %p, %ld\n", dptr, hptr, N); }
  CUDA_CHECK(cudaMemcpy(hptr, dptr, N * sizeof(T), cudaMemcpyDeviceToHost));
}
/// Copies device memory the newly allocated host memory.
/// @param dptr
/// @param N
/// @return hptr
template<typename T>
T* device_to_host_malloc(const T* dptr, long N) {
  T* hptr = new T[N];
  device_to_host(dptr, hptr, N);
  return hptr;
}

/// Copies device to device memory.
/// @param src
/// @param dst
/// @param N
template<typename T>
void device_to_device(const T* src, T* dst, long N) {
  if(DEBUG) { printf("[DEBUG] device_to_device %p => %p, %ld\n", src, dst, N); }
  CUDA_CHECK(cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyDeviceToDevice));
}


#endif
