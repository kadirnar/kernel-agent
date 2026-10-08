#ifndef FAKE_CUDA_RUNTIME_API_H
#define FAKE_CUDA_RUNTIME_API_H

/**
 * \defgroup CUDART_EXECUTION Execution Control
 *
 * This section describes the execution control functions of the CUDA runtime.
 */

/**
 * \brief Launches a CUDA function with launch-time configuration
 *
 * Note that the functionally equivalent variadic template ::cudaLaunchKernelEx
 * is available for C++11 and newer.
 */
extern __host__ cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                                         const void *func, void **args);

/**
 * Launch attributes enum; used as id field of ::cudaLaunchAttribute
 */
typedef enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore                          = 0, /**< Ignored entry */
    cudaLaunchAttributeProgrammaticStreamSerialization = 6  /**< Valid for launches. Setting
                                                              it to non-0 signals that the
                                                              kernel resolves its stream
                                                              dependency programmatically. */
} cudaLaunchAttributeID;

typedef enum cublasLtMatmulMatrixScale_t {
  /** Scaling factors are single precision scalars applied to the whole tensor */
  CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F = 0,
  /** Scaling factors are tensors that contain a dedicated scaling factor stored as an 8-bit
   * CUDA_R_8F_UE4M3 value for each 16-element block in the innermost dimension */
  CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 = 1,
} cublasLtMatmulMatrixScale_t;

#endif
