/*
 * Host-side kernel launch infrastructure for MXMACA backend
 *
 * This crate provides the same traits as cuda-host but uses maca-core
 * for MXMACA GPU support.
 */

use std::ffi::c_void;
use std::ptr;

// Re-export traits from cuda-host for compatibility
pub use cuda_macros::kernel;

/// Trait implemented by kernel functions (MXMACA version).
///
/// This is the same as cuda_host::CudaKernel but for MXMACA backend.
pub trait MacaKernel {
    /// The kernel entry point name
    const KERNEL_NAME: &'static str;
}

/// Trait for generic kernel functions (MXMACA version).
pub trait GenericMacaKernel {
    /// Get the kernel entry point name for this specific instantiation.
    fn kernel_name() -> &'static str;
}

/// MXMACA module handle
pub struct MacaModule {
    module: maca_bindings::mcModule_t,
}

impl MacaModule {
    /// Load a module from file
    pub fn load(path: &str) -> Result<Self, anyhow::Error> {
        let path_cstr = std::ffi::CString::new(path)?;
        let mut module = ptr::null_mut();
        unsafe {
            let result = maca_bindings::mcModuleLoad(&mut module, path_cstr.as_ptr());
            if result != maca_bindings::_mcError_t_mcSuccess as maca_bindings::mcError_t {
                anyhow::bail!("Failed to load module: {}", result);
            }
        }
        Ok(Self { module })
    }

    /// Get a function handle from the module
    pub fn get_function(&self, name: &str) -> Result<MacaFunction, anyhow::Error> {
        let name_cstr = std::ffi::CString::new(name)?;
        let mut function = ptr::null_mut();
        unsafe {
            let result = maca_bindings::mcModuleGetFunction(
                &mut function,
                self.module,
                name_cstr.as_ptr(),
            );
            if result != maca_bindings::_mcError_t_mcSuccess as maca_bindings::mcError_t {
                anyhow::bail!("Failed to get function '{}': {}", name, result);
            }
        }
        Ok(MacaFunction { function })
    }
}

impl Drop for MacaModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = maca_bindings::mcModuleUnload(self.module);
            }
        }
    }
}

/// MXMACA function handle
pub struct MacaFunction {
    function: maca_bindings::mcFunction_t,
}

impl MacaFunction {
    /// Launch the kernel using mcModuleLaunchKernel
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - Grid and block dimensions are valid
    /// - Arguments match the kernel signature
    /// - Device memory pointers are valid
    pub unsafe fn launch(
        &self,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem: u32,
        args: &mut Vec<*mut c_void>,
    ) -> Result<(), anyhow::Error> {
        let result = maca_bindings::mcModuleLaunchKernel(
            self.function,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            shared_mem,
            ptr::null_mut(), // stream (default)
            args.as_mut_ptr(),
            ptr::null_mut(), // extra
        );
        if result != maca_bindings::_mcError_t_mcSuccess as maca_bindings::mcError_t {
            anyhow::bail!("Kernel launch failed: {}", result);
        }
        Ok(())
    }
}

/// Launch a kernel using mcLaunchKernel (runtime API)
///
/// This uses the function pointer directly, similar to CUDA's <<<>>> syntax.
///
/// # Safety
///
/// The caller must ensure:
/// - function_address is a valid kernel function pointer
/// - Grid and block dimensions are valid
/// - Arguments match the kernel signature
/// - Device memory pointers are valid
pub unsafe fn launch_kernel(
    function_address: *const std::ffi::c_void,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem: u32,
    args: &mut Vec<*mut c_void>,
) -> Result<(), anyhow::Error> {
    let grid = maca_bindings::dim3 {
        x: grid_dim.0,
        y: grid_dim.1,
        z: grid_dim.2,
    };
    let block = maca_bindings::dim3 {
        x: block_dim.0,
        y: block_dim.1,
        z: block_dim.2,
    };
    let result = maca_bindings::mcLaunchKernel(
        function_address,
        grid,
        block,
        args.as_mut_ptr(),
        shared_mem as usize,
        ptr::null_mut(), // stream (default)
    );
    if result != maca_bindings::_mcError_t_mcSuccess as maca_bindings::mcError_t {
        anyhow::bail!("Kernel launch failed: {}", result);
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

/// Push a scalar argument
pub fn push_scalar<T: Copy>(args: &mut Vec<*mut c_void>, value: &mut T) {
    if std::mem::size_of::<T>() == 0 {
        return;
    }
    args.push(value as *mut T as *mut c_void);
}

/// Push a device buffer argument (pointer + length)
pub fn push_device_buffer<T>(
    args: &mut Vec<*mut c_void>,
    buffer: &maca_core::DeviceBuffer<T>,
) {
    let ptr = buffer.as_ptr();
    let len = buffer.len() as u64;
    args.push(&ptr as *const _ as *mut c_void);
    args.push(&len as *const _ as *mut c_void);
}
