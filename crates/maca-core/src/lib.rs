/*
 * Safe RAII wrappers for MXMACA Runtime API
 */

use std::ptr;

/// MXMACA error type
#[derive(Debug)]
pub enum MacaError {
    RuntimeError(i32),
    NullPointer,
}

impl std::fmt::Display for MacaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MacaError::RuntimeError(code) => write!(f, "MXMACA runtime error: {}", code),
            MacaError::NullPointer => write!(f, "Null pointer"),
        }
    }
}

impl std::error::Error for MacaError {}

/// Check MXMACA runtime result
fn check(result: maca_bindings::mcError_t) -> Result<(), MacaError> {
    if result == maca_bindings::_mcError_t_mcSuccess as maca_bindings::mcError_t {
        Ok(())
    } else {
        Err(MacaError::RuntimeError(result as i32))
    }
}

/// MXMACA context
pub struct MacaContext {
    device: i32,
}

impl MacaContext {
    /// Create a new context for the given device
    pub fn new(device: i32) -> Result<Self, MacaError> {
        unsafe {
            check(maca_bindings::mcSetDevice(device))?;
        }
        Ok(Self { device })
    }

    /// Get the device ordinal
    pub fn device(&self) -> i32 {
        self.device
    }
}

/// Device buffer
pub struct DeviceBuffer<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> DeviceBuffer<T> {
    /// Allocate a device buffer
    pub fn new(len: usize) -> Result<Self, MacaError> {
        let size = len * std::mem::size_of::<T>();
        let mut ptr = ptr::null_mut();
        unsafe {
            check(maca_bindings::mcMalloc(
                &mut ptr as *mut *mut _ as *mut *mut std::ffi::c_void,
                size,
            ))?;
        }
        Ok(Self {
            ptr: ptr as *mut T,
            len,
        })
    }

    /// Create a device buffer from host data
    pub fn from_host(data: &[T]) -> Result<Self, MacaError> {
        let buf = Self::new(data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    /// Copy data from host to device
    pub fn copy_from_host(&self, data: &[T]) -> Result<(), MacaError> {
        let size = data.len() * std::mem::size_of::<T>();
        unsafe {
            check(maca_bindings::mcMemcpy(
                self.ptr as *mut std::ffi::c_void,
                data.as_ptr() as *const std::ffi::c_void,
                size,
                maca_bindings::_mcMemcpyKind_mcMemcpyHostToDevice,
            ))?;
        }
        Ok(())
    }

    /// Copy data from device to host
    pub fn to_host_vec(&self) -> Result<Vec<T>, MacaError> {
        let mut result = Vec::with_capacity(self.len);
        unsafe {
            result.set_len(self.len);
            check(maca_bindings::mcMemcpy(
                result.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr as *const std::ffi::c_void,
                self.len * std::mem::size_of::<T>(),
                maca_bindings::_mcMemcpyKind_mcMemcpyDeviceToHost,
            ))?;
        }
        Ok(result)
    }

    /// Get the device pointer
    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = maca_bindings::mcFree(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

/// Synchronize the device
pub fn device_synchronize() -> Result<(), MacaError> {
    unsafe {
        check(maca_bindings::mcDeviceSynchronize())?;
    }
    Ok(())
}

/// Launch configuration
pub struct LaunchConfig {
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem: u32,
}

impl LaunchConfig {
    /// Create a 1D launch configuration
    pub fn for_num_elems(n: u32) -> Self {
        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;
        Self {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem: 0,
        }
    }
}
