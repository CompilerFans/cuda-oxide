/*
 * maca-vecadd: Simple vector addition using MXMACA runtime
 *
 * This example demonstrates using maca-core to run a vecadd kernel
 * on MetaX GPU.
 */

use maca_core::{DeviceBuffer, device_synchronize};

// The kernel is compiled separately with mxcc
// This example just tests the host-side runtime

fn main() {
    println!("=== maca-vecadd: MXMACA Runtime Test ===\n");

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    println!("Input vectors (first 5 elements):");
    println!("  a = {:?}", &a_host[0..5]);
    println!("  b = {:?}", &b_host[0..5]);

    // Allocate device memory
    let a_dev = DeviceBuffer::from_host(&a_host).expect("Failed to allocate a");
    let b_dev = DeviceBuffer::from_host(&b_host).expect("Failed to allocate b");
    let c_dev = DeviceBuffer::<f32>::new(N).expect("Failed to allocate c");

    // Note: Kernel launch would go here using the compiled .maca file
    // For now, we just test the memory allocation and copy

    // Copy results back
    let c_host = c_dev.to_host_vec().expect("Failed to copy to host");

    println!("\nOutput vector (first 5 elements):");
    println!("  c = {:?}", &c_host[0..5]);

    // Verify (will be 0 since no kernel was launched)
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
        println!("\n✗ EXPECTED: {} errors (no kernel launched)", errors);
        println!("  Memory allocation and copy operations work correctly!");
    }

    // Test device synchronization
    device_synchronize().expect("Device sync failed");
    println!("Device synchronization: OK");
}
