/*
 * maca-vecadd: Vector addition using cuda-oxide MXMACA backend
 *
 * This example demonstrates direct kernel launch using a C launcher
 * compiled by mxcc. The launcher handles device context initialization
 * and kernel launch via <<<>>> syntax.
 */

use maca_core::{DeviceBuffer, device_synchronize};
use std::process::Command;

// The launcher function is compiled from vecadd_launcher.maca
extern "C" {
    fn launch_vecadd(
        d_out: *mut f32,
        d_a: *const f32,
        d_b: *const f32,
        n: i32,
    ) -> i32;
}

fn main() {
    println!("=== cuda-oxide maca-vecadd: Direct Kernel Launch ===\n");

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    println!("Input vectors (first 5 elements):");
    println!("  a = {:?}", &a_host[0..5]);
    println!("  b = {:?}", &b_host[0..5]);

    // Compile the launcher with mxcc
    let launcher_src = concat!(env!("CARGO_MANIFEST_DIR"), "/kernel/vecadd_launcher.maca");
    let launcher_lib = "/tmp/libvecadd_launcher.so";

    println!("\nCompiling launcher with mxcc...");
    let status = Command::new("mxcc")
        .args([
            "-shared",
            "-fPIC",
            launcher_src,
            "-o",
            launcher_lib,
            "--maca-path=/opt/maca",
        ])
        .status()
        .expect("Failed to run mxcc");

    if !status.success() {
        eprintln!("mxcc compilation failed");
        std::process::exit(1);
    }
    println!("Launcher compiled to {}", launcher_lib);

    // Load the launcher library
    println!("Loading launcher library...");
    let lib = unsafe { libloading::Library::new(launcher_lib) }
        .expect("Failed to load launcher library");

    // Get the launch function
    let launch_fn: libloading::Symbol<unsafe extern "C" fn(*mut f32, *const f32, *const f32, i32) -> i32> =
        unsafe { lib.get(b"launch_vecadd") }
            .expect("Failed to find launch_vecadd function");
    println!("Launch function found");

    // Allocate device memory
    println!("\nAllocating device memory...");
    let a_dev = DeviceBuffer::from_host(&a_host).expect("Failed to allocate a");
    let b_dev = DeviceBuffer::from_host(&b_host).expect("Failed to allocate b");
    let c_dev = DeviceBuffer::<f32>::new(N).expect("Failed to allocate c");
    println!("Device memory allocated");

    // Launch kernel
    println!("\nLaunching kernel...");
    let result = unsafe {
        launch_fn(c_dev.as_ptr(), a_dev.as_ptr(), b_dev.as_ptr(), N as i32)
    };

    if result != 0 {
        eprintln!("Kernel launch failed with error: {}", result);
        std::process::exit(1);
    }
    println!("Kernel launched and synchronized");

    // Copy results back
    let c_host = c_dev.to_host_vec().expect("Failed to copy to host");

    println!("\nOutput vector (first 5 elements):");
    println!("  c = {:?}", &c_host[0..5]);

    // Verify
    let mut errors = 0;
    for i in 0..N {
        let expected = a_host[i] + b_host[i];
        if (c_host[i] - expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!(
                    "  Error at [{}]: expected {}, got {}",
                    i, expected, c_host[i]
                );
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("\n✓ SUCCESS: All {} elements correct!", N);
    } else {
        println!("\n✗ FAILED: {} errors", errors);
        std::process::exit(1);
    }
}
