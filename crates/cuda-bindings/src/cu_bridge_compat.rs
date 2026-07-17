//! CUDA Driver API spellings backed by MXMACA cu-bridge bindings.
//!
//! Bindgen sees the post-preprocessor cu-bridge declarations, whose public
//! names use `mcDrv*` types and `wcu*` functions. This module restores the CUDA
//! names expected by `cuda-core` while preserving the generated ABI exactly.

// Driver handle and scalar types.
pub type CUresult = crate::mcDrvError_t;
pub type CUevent = crate::mcDrvEvent_t;
pub type CUdevice = crate::mcDrvDevice_t;
pub type CUcontext = crate::mcDrvContext_t;
pub type CUmodule = crate::mcDrvModule_t;
pub type CUfunction = crate::mcDrvFunction_t;
pub type CUstream = crate::mcDrvStream_t;
pub type CUdeviceptr = crate::mcDrvDeviceptr_t;

// Enum and aggregate types used by cuda-core.
pub type CUdevice_attribute = crate::mcDrvDeviceAttribute_t;
pub type CUevent_flags = crate::mcDrvEventFlags;
pub type CUfunction_attribute = crate::mcDrvFunction_attribute;
pub type CUlaunchAttribute_st = crate::mcDrvlaunchAttribute_st;
pub type CUlaunchConfig_st = crate::mcDrvLaunchConfigExtension_st;
pub type CUmemLocation_st = crate::mcMemLocation;
pub type CUmemGenericAllocationHandle = crate::mcDrvMemGenericAllocationHandle_t;
pub type CUmemAllocationProp_st = crate::mcMemAllocationProp_st;
pub type CUmemAccessDesc_st = crate::mcMemAccessDesc;

// Error spellings emitted by NVIDIA's cuda.h.
pub use crate::mcDrvError_enum_MC_ERROR_INVALID_VALUE as cudaError_enum_CUDA_ERROR_INVALID_VALUE;
pub use crate::mcDrvError_enum_MC_ERROR_NOT_READY as cudaError_enum_CUDA_ERROR_NOT_READY;
pub use crate::mcDrvError_enum_MC_ERROR_NOT_SUPPORTED as cudaError_enum_CUDA_ERROR_NOT_SUPPORTED;
pub use crate::mcDrvError_enum_MC_ERROR_PEER_ACCESS_ALREADY_ENABLED as cudaError_enum_CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
pub use crate::mcDrvError_enum_MC_ERROR_PEER_ACCESS_NOT_ENABLED as cudaError_enum_CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
pub use crate::mcDrvError_enum_MC_SUCCESS as cudaError_enum_CUDA_SUCCESS;

// Preserve the conventional CUDA error aliases already exposed by this crate.
pub const CUDA_SUCCESS: CUresult = crate::mcDrvError_enum_MC_SUCCESS;
pub const CUDA_ERROR_INVALID_VALUE: CUresult = crate::mcDrvError_enum_MC_ERROR_INVALID_VALUE;
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = crate::mcDrvError_enum_MC_ERROR_OUT_OF_MEMORY;
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = crate::mcDrvError_enum_MC_ERROR_NOT_INITIALIZED;
pub const CUDA_ERROR_DEINITIALIZED: CUresult = crate::mcDrvError_enum_MC_ERROR_DEINITIALIZED;
pub const CUDA_ERROR_NO_DEVICE: CUresult = crate::mcDrvError_enum_MC_ERROR_NO_DEVICE;
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = crate::mcDrvError_enum_MC_ERROR_INVALID_DEVICE;
pub const CUDA_ERROR_INVALID_IMAGE: CUresult = crate::mcDrvError_enum_MC_ERROR_INVALID_IMAGE;
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = crate::mcDrvError_enum_MC_ERROR_INVALID_CONTEXT;
pub const CUDA_ERROR_INVALID_HANDLE: CUresult = crate::mcDrvError_enum_MC_ERROR_INVALID_HANDLE;
pub const CUDA_ERROR_NOT_FOUND: CUresult = crate::mcDrvError_enum_MC_ERROR_NOT_FOUND;
pub const CUDA_ERROR_NOT_READY: CUresult = crate::mcDrvError_enum_MC_ERROR_NOT_READY;
pub const CUDA_ERROR_ILLEGAL_ADDRESS: CUresult = crate::mcDrvError_enum_MC_ERROR_ILLEGAL_ADDRESS;
pub const CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: CUresult =
    crate::mcDrvError_enum_MC_ERROR_LAUNCH_OUT_OF_RESOURCES;
pub const CUDA_ERROR_LAUNCH_TIMEOUT: CUresult = crate::mcDrvError_enum_MC_ERROR_LAUNCH_TIMEOUT;
pub const CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: CUresult =
    crate::mcDrvError_enum_MC_ERROR_PEER_ACCESS_ALREADY_ENABLED;
pub const CUDA_ERROR_PEER_ACCESS_NOT_ENABLED: CUresult =
    crate::mcDrvError_enum_MC_ERROR_PEER_ACCESS_NOT_ENABLED;
pub const CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE: CUresult =
    crate::mcDrvError_enum_MC_ERROR_PRIMARY_CONTEXT_ACTIVE;
pub const CUDA_ERROR_CONTEXT_IS_DESTROYED: CUresult =
    crate::mcDrvError_enum_MC_ERROR_CONTEXT_IS_DESTROYED;
pub const CUDA_ERROR_NOT_SUPPORTED: CUresult = crate::mcDrvError_enum_MC_ERROR_NOT_SUPPORTED;
pub const CUDA_ERROR_UNKNOWN: CUresult = crate::mcDrvError_enum_MC_ERROR_UNKNOWN;

// Stream and event flags.
pub use crate::mcDrvEventFlagsEnum_MC_EVENT_DEFAULT as CUevent_flags_enum_CU_EVENT_DEFAULT;
pub use crate::mcDrvEventFlagsEnum_MC_EVENT_DISABLE_TIMING as CUevent_flags_enum_CU_EVENT_DISABLE_TIMING;
pub use crate::mcDrvEventWaitFlagsEnum_MC_EVENT_WAIT_DEFAULT as CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT;
pub use crate::mcDrvStreamFlags_enum_MC_STREAM_NON_BLOCKING as CUstream_flags_enum_CU_STREAM_NON_BLOCKING;

// Device attributes. A few cu-bridge names differ from CUDA beyond the prefix.
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_CLUSTERLAUNCH as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTION as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
pub use crate::mcDrvDeviceAttribute_enum_MC_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;

// Function attributes. cu-bridge exposes CUDA's original attributes by name.
pub use crate::mcFunctionAttribute_enum_MC_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES as CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
pub use crate::mcFunctionAttribute_enum_MC_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK as CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
pub use crate::mcFunctionAttribute_enum_MC_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES as CUfunction_attribute_enum_CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES;

// Extended-launch attributes and VMM constants.
pub use crate::mcLaunchAttributeID_mcLaunchAttributeClusterDimension as CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
pub use crate::mcLaunchAttributeID_mcLaunchAttributeCooperative as CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
pub use crate::mcMemAccessFlags_mcMemAccessFlagsProtReadWrite as CUmemAccess_flags_enum_CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
pub use crate::mcMemAllocationGranularityFlags_enum_MC_MEM_ALLOC_GRANULARITY_MINIMUM as CUmemAllocationGranularity_flags_enum_CU_MEM_ALLOC_GRANULARITY_MINIMUM;
pub use crate::mcMemAllocationType_mcMemAllocationTypePinned as CUmemAllocationType_enum_CU_MEM_ALLOCATION_TYPE_PINNED;
pub use crate::mcMemLocationType_mcMemLocationTypeDevice as CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE;

// Driver entry points. Versioned CUDA names map to cu-bridge's unversioned ABI.
pub use crate::wcuCtxDisablePeerAccess as cuCtxDisablePeerAccess;
pub use crate::wcuCtxEnablePeerAccess as cuCtxEnablePeerAccess;
pub use crate::wcuCtxGetCurrent as cuCtxGetCurrent;
pub use crate::wcuCtxSetCurrent as cuCtxSetCurrent;
pub use crate::wcuCtxSynchronize as cuCtxSynchronize;
pub use crate::wcuDeviceCanAccessPeer as cuDeviceCanAccessPeer;
pub use crate::wcuDeviceGet as cuDeviceGet;
pub use crate::wcuDeviceGetAttribute as cuDeviceGetAttribute;
pub use crate::wcuDeviceGetCount as cuDeviceGetCount;
pub use crate::wcuDeviceGetName as cuDeviceGetName;
pub use crate::wcuDevicePrimaryCtxRelease as cuDevicePrimaryCtxRelease_v2;
pub use crate::wcuDevicePrimaryCtxRetain as cuDevicePrimaryCtxRetain;
pub use crate::wcuEventCreate as cuEventCreate;
pub use crate::wcuEventDestroy as cuEventDestroy_v2;
pub use crate::wcuEventQuery as cuEventQuery;
pub use crate::wcuEventRecord as cuEventRecord;
pub use crate::wcuEventSynchronize as cuEventSynchronize;
pub use crate::wcuFuncGetAttribute as cuFuncGetAttribute;
pub use crate::wcuFuncSetAttribute as cuFuncSetAttribute;
pub use crate::wcuGetErrorName as cuGetErrorName;
pub use crate::wcuGetErrorString as cuGetErrorString;
pub use crate::wcuInit as cuInit;
pub use crate::wcuLaunchHostFunc as cuLaunchHostFunc;
pub use crate::wcuLaunchKernel as cuLaunchKernel;
pub use crate::wcuLaunchKernelEx as cuLaunchKernelEx;
pub use crate::wcuMemAddressFree as cuMemAddressFree;
pub use crate::wcuMemAddressReserve as cuMemAddressReserve;
pub use crate::wcuMemAlloc as cuMemAlloc_v2;
pub use crate::wcuMemAllocAsync as cuMemAllocAsync;
pub use crate::wcuMemAllocHost as cuMemAllocHost_v2;
pub use crate::wcuMemCreate as cuMemCreate;
pub use crate::wcuMemFree as cuMemFree_v2;
pub use crate::wcuMemFreeAsync as cuMemFreeAsync;
pub use crate::wcuMemFreeHost as cuMemFreeHost;
pub use crate::wcuMemGetAllocationGranularity as cuMemGetAllocationGranularity;
pub use crate::wcuMemGetInfo as cuMemGetInfo_v2;
pub use crate::wcuMemMap as cuMemMap;
pub use crate::wcuMemRelease as cuMemRelease;
pub use crate::wcuMemSetAccess as cuMemSetAccess;
pub use crate::wcuMemUnmap as cuMemUnmap;
pub use crate::wcuMemcpyDtoDAsync as cuMemcpyDtoDAsync_v2;
pub use crate::wcuMemcpyDtoHAsync as cuMemcpyDtoHAsync_v2;
pub use crate::wcuMemcpyHtoD as cuMemcpyHtoD_v2;
pub use crate::wcuMemcpyHtoDAsync as cuMemcpyHtoDAsync_v2;
pub use crate::wcuMemsetD8Async as cuMemsetD8Async;
pub use crate::wcuModuleGetFunction as cuModuleGetFunction;
pub use crate::wcuModuleGetGlobal as cuModuleGetGlobal_v2;
pub use crate::wcuModuleLoad as cuModuleLoad;
pub use crate::wcuModuleLoadData as cuModuleLoadData;
pub use crate::wcuModuleUnload as cuModuleUnload;
pub use crate::wcuOccupancyMaxActiveBlocksPerMultiprocessor as cuOccupancyMaxActiveBlocksPerMultiprocessor;
pub use crate::wcuStreamCreate as cuStreamCreate;
pub use crate::wcuStreamDestroy as cuStreamDestroy_v2;
pub use crate::wcuStreamSynchronize as cuStreamSynchronize;
pub use crate::wcuStreamWaitEvent as cuStreamWaitEvent;

/// Calls cu-bridge's cluster-size query with CUDA's typed launch config.
///
/// # Safety
///
/// `cluster_size`, `func`, and `config` must satisfy the CUDA Driver API
/// contract for `cuOccupancyMaxPotentialClusterSize`.
pub unsafe fn cuOccupancyMaxPotentialClusterSize(
    cluster_size: *mut std::ffi::c_int,
    func: CUfunction,
    config: *const CUlaunchConfig_st,
) -> CUresult {
    unsafe { crate::wcuOccupancyMaxPotentialClusterSize(cluster_size, func, config.cast()) }
}

/// Calls cu-bridge's active-cluster query with CUDA's typed launch config.
///
/// # Safety
///
/// `num_clusters`, `func`, and `config` must satisfy the CUDA Driver API
/// contract for `cuOccupancyMaxActiveClusters`.
pub unsafe fn cuOccupancyMaxActiveClusters(
    num_clusters: *mut std::ffi::c_int,
    func: CUfunction,
    config: *const CUlaunchConfig_st,
) -> CUresult {
    unsafe { crate::wcuOccupancyMaxActiveClusters(num_clusters, func, config.cast()) }
}
