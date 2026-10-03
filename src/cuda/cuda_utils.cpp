/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */

#ifdef USE_CUDA

#include <LightGBM/cuda/cuda_rocm_interop.h>
#include <LightGBM/cuda/cuda_utils.hu>

#include <mutex>
#include <unordered_map>

namespace LightGBM {

namespace {

std::mutex& NCCLCommunicatorStreamMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<ncclComm_t, cudaStream_t>& NCCLCommunicatorStreams() {
  static std::unordered_map<ncclComm_t, cudaStream_t> streams;
  return streams;
}

// Runs func with the communicator's device current, restoring the caller's device afterwards.
template <typename FUNC>
void RunOnCommunicatorDevice(ncclComm_t comm, FUNC func) {
  int comm_device = 0;
  NCCLCHECK(ncclCommCuDevice(comm, &comm_device));
  int previous_device = 0;
  CUDASUCCESS_OR_FATAL(cudaGetDevice(&previous_device));
  if (previous_device != comm_device) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(comm_device));
  }
  func();
  if (previous_device != comm_device) {
    CUDASUCCESS_OR_FATAL(cudaSetDevice(previous_device));
  }
}

}  // namespace

void SynchronizeCUDADevice(const char* file, const int line) {
  gpuAssert(cudaDeviceSynchronize(), file, line);
}

void SynchronizeCUDAStream(cudaStream_t cuda_stream, const char* file, const int line) {
  gpuAssert(cudaStreamSynchronize(cuda_stream), file, line);
}

void PrintLastCUDAError() {
  const char* error_name = cudaGetErrorName(cudaGetLastError());
  Log::Fatal(error_name);
}

void SetCUDADevice(int gpu_device_id, const char* file, int line) {
  int cur_gpu_device_id = 0;
  CUDASUCCESS_OR_FATAL_OUTER(cudaGetDevice(&cur_gpu_device_id));
  if (cur_gpu_device_id != gpu_device_id) {
    CUDASUCCESS_OR_FATAL_OUTER(cudaSetDevice(gpu_device_id));
  }
}

int GetCUDADevice(const char* file, int line) {
  int cur_gpu_device_id = 0;
  CUDASUCCESS_OR_FATAL_OUTER(cudaGetDevice(&cur_gpu_device_id));
  return cur_gpu_device_id;
}

cudaStream_t CUDAStreamCreate() {
  cudaStream_t cuda_stream;
  CUDASUCCESS_OR_FATAL(cudaStreamCreate(&cuda_stream));
  return cuda_stream;
}

void CUDAStreamDestroy(cudaStream_t cuda_stream) {
  CUDASUCCESS_OR_FATAL(cudaStreamDestroy(cuda_stream));
}

void NCCLGroupStart() {
  NCCLCHECK(ncclGroupStart());
}

void NCCLGroupEnd() {
  NCCLCHECK(ncclGroupEnd());
}

cudaStream_t NCCLCommunicatorStream(ncclComm_t comm) {
  std::lock_guard<std::mutex> lock(NCCLCommunicatorStreamMutex());
  auto& streams = NCCLCommunicatorStreams();
  const auto it = streams.find(comm);
  if (it != streams.end()) {
    return it->second;
  }
  cudaStream_t stream = nullptr;
  RunOnCommunicatorDevice(comm, [&stream]() { stream = CUDAStreamCreate(); });
  streams.emplace(comm, stream);
  return stream;
}

void ReleaseNCCLCommunicatorStream(ncclComm_t comm) {
  cudaStream_t stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(NCCLCommunicatorStreamMutex());
    auto& streams = NCCLCommunicatorStreams();
    const auto it = streams.find(comm);
    if (it == streams.end()) {
      return;
    }
    stream = it->second;
    streams.erase(it);
  }
  RunOnCommunicatorDevice(comm, [stream]() {
    CUDASUCCESS_OR_FATAL(cudaStreamSynchronize(stream));
    CUDAStreamDestroy(stream);
  });
}

}  // namespace LightGBM

#endif  // USE_CUDA
