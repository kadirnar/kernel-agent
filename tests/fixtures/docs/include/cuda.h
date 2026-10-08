#ifndef FAKE_CUDA_H
#define FAKE_CUDA_H

#define CUDA_VERSION 13000

/**
 * \brief Launches a CUDA function with launch-time configuration
 *
 * Invokes the kernel \p f with the launch attributes of \p config, such as
 * ::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.
 */
CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                                  void **kernelParams, void **extra);

#endif
