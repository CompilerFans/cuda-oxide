/*
 * cu-bridge compatibility header for cuda-bindings
 *
 * This header maps cu-bridge types to CUDA Driver API types,
 * allowing cuda-bindings to compile against cu-bridge headers.
 *
 * cu-bridge uses different type names (mcDrv*_t) than CUDA (CU*).
 * This header provides typedefs to make them compatible.
 */

#ifndef CUDA_BRIDGE_COMPAT_H
#define CUDA_BRIDGE_COMPAT_H

/* Include cu-bridge headers */
#include <cuda.h>

/* Map cu-bridge types to CUDA types */
typedef mcDrvError_t CUresult;
typedef mcDrvDevice_t CUdevice;
typedef mcDrvDeviceptr_t CUdeviceptr;
typedef mcDrvContext_t CUcontext;
typedef mcDrvModule_t CUmodule;
typedef mcDrvFunction_t CUfunction;
typedef mcDrvStream_t CUstream;
typedef mcDrvEvent_t CUevent;
typedef mcDrvUuid_t CUuuid;

/* Map cu-bridge error codes to CUDA error codes */
#define CUDA_SUCCESS MC_SUCCESS
#define CUDA_ERROR_INVALID_VALUE MC_ERROR_INVALID_VALUE
#define CUDA_ERROR_OUT_OF_MEMORY MC_ERROR_OUT_OF_MEMORY
#define CUDA_ERROR_NOT_INITIALIZED MC_ERROR_NOT_INITIALIZED
#define CUDA_ERROR_DEINITIALIZED MC_ERROR_DEINITIALIZED
#define CUDA_ERROR_NO_DEVICE MC_ERROR_NO_DEVICE
#define CUDA_ERROR_INVALID_DEVICE MC_ERROR_INVALID_DEVICE
#define CUDA_ERROR_INVALID_IMAGE MC_ERROR_INVALID_IMAGE
#define CUDA_ERROR_INVALID_CONTEXT MC_ERROR_INVALID_CONTEXT
#define CUDA_ERROR_INVALID_HANDLE MC_ERROR_INVALID_HANDLE
#define CUDA_ERROR_NOT_FOUND MC_ERROR_NOT_FOUND
#define CUDA_ERROR_NOT_READY MC_ERROR_NOT_READY
#define CUDA_ERROR_ILLEGAL_ADDRESS MC_ERROR_ILLEGAL_ADDRESS
#define CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES MC_ERROR_LAUNCH_OUT_OF_RESOURCES
#define CUDA_ERROR_LAUNCH_TIMEOUT MC_ERROR_LAUNCH_TIMEOUT
#define CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED MC_ERROR_PEER_ACCESS_ALREADY_ENABLED
#define CUDA_ERROR_PEER_ACCESS_NOT_ENABLED MC_ERROR_PEER_ACCESS_NOT_ENABLED
#define CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE MC_ERROR_PRIMARY_CONTEXT_ACTIVE
#define CUDA_ERROR_CONTEXT_IS_DESTROYED MC_ERROR_CONTEXT_IS_DESTROYED
#define CUDA_ERROR_NOT_SUPPORTED MC_ERROR_NOT_SUPPORTED
#define CUDA_ERROR_UNKNOWN MC_ERROR_UNKNOWN

/* Map cu-bridge device attributes to CUDA device attributes */
#define CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK MC_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X MC_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y MC_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y
#define CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z MC_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X MC_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y MC_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y
#define CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z MC_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z
#define CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK MC_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT MC_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
#define CU_DEVICE_ATTRIBUTE_WARP_SIZE MC_DEVICE_ATTRIBUTE_WARP_SIZE

/* Map cu-bridge functions to CUDA functions */
#define cuInit wcuInit
#define cuDeviceGet wcuDeviceGet
#define cuDeviceGetCount wcuDeviceGetCount
#define cuDeviceGetName wcuDeviceGetName
#define cuDeviceGetAttribute wcuDeviceGetAttribute
#define cuCtxCreate wcuCtxCreate
#define cuCtxDestroy wcuCtxDestroy
#define cuCtxSetCurrent wcuCtxSetCurrent
#define cuCtxGetCurrent wcuCtxGetCurrent
#define cuCtxSynchronize wcuCtxSynchronize
#define cuModuleLoad wcuModuleLoad
#define cuModuleLoadData wcuModuleLoadData
#define cuModuleUnload wcuModuleUnload
#define cuModuleGetFunction wcuModuleGetFunction
#define cuModuleGetGlobal wcuModuleGetGlobal
#define cuMemAlloc wcuMemAlloc
#define cuMemFree wcuMemFree
#define cuMemAllocHost wcuMemAllocHost
#define cuMemFreeHost wcuMemFreeHost
#define cuMemcpyHtoD wcuMemcpyHtoD
#define cuMemcpyDtoH wcuMemcpyDtoH
#define cuMemcpyHtoDAsync wcuMemcpyHtoDAsync
#define cuMemcpyDtoHAsync wcuMemcpyDtoHAsync
#define cuMemsetD8 wcuMemsetD8
#define cuMemsetD32 wcuMemsetD32
#define cuStreamCreate wcuStreamCreate
#define cuStreamDestroy wcuStreamDestroy
#define cuStreamSynchronize wcuStreamSynchronize
#define cuStreamQuery wcuStreamQuery
#define cuEventCreate wcuEventCreate
#define cuEventDestroy wcuEventDestroy
#define cuEventRecord wcuEventRecord
#define cuEventSynchronize wcuEventSynchronize
#define cuEventQuery wcuEventQuery
#define cuEventElapsedTime wcuEventElapsedTime
#define cuLaunchKernel wcuLaunchKernel
#define cuFuncSetAttribute wcuFuncSetAttribute
#define cuFuncGetAttribute wcuFuncGetAttribute
#define cuOccupancyMaxActiveBlocksPerMultiprocessor wcuOccupancyMaxActiveBlocksPerMultiprocessor
#define cuGetErrorName wcuGetErrorName
#define cuGetErrorString wcuGetErrorString

/* Map CUDA stream flags */
#define CU_STREAM_DEFAULT 0x0
#define CU_STREAM_NON_BLOCKING 0x1

/* Map CUDA event flags */
#define CU_EVENT_DEFAULT 0x0
#define CU_EVENT_BLOCKING_SYNC 0x1
#define CU_EVENT_DISABLE_TIMING 0x2
#define CU_EVENT_INTERPROCESS 0x4

/* Map CUDA function attributes */
#define CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK MC_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES MC_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
#define CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES MC_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY
#define CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES 8
#define CU_FUNC_ATTRIBUTE_NUM_REGS MC_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK
#define CU_FUNC_ATTRIBUTE_PTX_VERSION 50
#define CU_FUNC_ATTRIBUTE_BINARY_VERSION 50

/* Map CUDA memory types */
#define CU_MEMORYTYPE_HOST 1
#define CU_MEMORYTYPE_DEVICE 2
#define CU_MEMORYTYPE_ARRAY 3
#define CU_MEMORYTYPE_UNIFIED 4

#endif /* CUDA_BRIDGE_COMPAT_H */
