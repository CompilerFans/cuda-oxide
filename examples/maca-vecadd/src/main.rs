/*
 * maca-vecadd: Vector addition using cuda-oxide MXMACA backend
 *
 * This example demonstrates the cuda-oxide MXMACA pipeline:
 * 1. Kernel is compiled with mxcc to a standalone executable
 * 2. The executable is called from Rust to run the kernel on MetaX GPU
 *
 * This is the simplest approach that works end-to-end on MXMACA.
 * Future work will integrate the kernel launch directly into Rust.
 */

use std::process::Command;

fn main() {
    println!("=== cuda-oxide maca-vecadd: MXMACA Kernel Launch ===\n");

    const N: usize = 1024;

    // Compile the kernel with mxcc
    let kernel_src = concat!(env!("CARGO_MANIFEST_DIR"), "/kernel/vecadd.maca");
    let kernel_bin = "/tmp/maca_vecadd_test";

    println!("Compiling kernel with mxcc...");
    let status = Command::new("mxcc")
        .args([
            kernel_src,
            "-o",
            kernel_bin,
            "-O3",
            "--maca-path=/opt/maca",
        ])
        .status()
        .expect("Failed to run mxcc");

    if !status.success() {
        eprintln!("mxcc compilation failed");
        std::process::exit(1);
    }
    println!("Kernel compiled successfully");

    // Run the kernel
    println!("\nRunning kernel...");
    let output = Command::new(kernel_bin)
        .arg(N.to_string())
        .output()
        .expect("Failed to run kernel");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stderr.is_empty() {
        eprintln!("Kernel errors:\n{}", stderr);
    }

    println!("Kernel output:\n{}", stdout);

    // Check result
    if stdout.contains("SUCCESS") {
        println!("\n✓ cuda-oxide MXMACA pipeline working!");
    } else {
        println!("\n✗ Kernel failed");
        std::process::exit(1);
    }
}
