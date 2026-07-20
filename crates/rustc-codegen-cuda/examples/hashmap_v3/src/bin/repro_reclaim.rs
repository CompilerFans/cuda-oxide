/*
 * Minimal reproducer for the MXMACA reclaim hang (Test 13 cycle 0).
 *
 * Seeds a small table at ~90% load, deletes a subset, then reinserts it
 * (chunked) with progress prints after every stage.
 */

use cuda_core::CudaContext;
use hashmap_v3::{GpuSwissMap, distinct_keys, kernels};

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();

    let module = ctx
        .load_module_from_file("repro_reclaim.ptx")
        .or_else(|_| ctx.load_module_from_file("hashmap_v3.ptx"))
        .expect("Failed to load module");
    let module = kernels::from_module(module).expect("Failed to initialize typed module");

    const CAPACITY: usize = 16384;
    let m = (CAPACITY * 9) / 10; // ~90% load
    let base_keys = distinct_keys(m, 0xBADC_0FFE);
    let base_values: Vec<u32> = (0..m as u32).collect();

    let map = GpuSwissMap::new(CAPACITY, &stream).expect("alloc");
    println!("seed insert ({} keys)...", m);
    map.insert_bulk(&base_keys, &base_values, &module, &stream)
        .expect("seed insert");
    stream.synchronize().expect("seed sync");
    println!("seed insert done");

    let churn_n = m / 4;
    let churn_keys: Vec<u32> = base_keys[..churn_n].to_vec();

    for c in 0..10u32 {
        println!("cycle {} delete...", c);
        map.delete_bulk(&churn_keys, &module, &stream)
            .expect("delete");
        println!("cycle {} delete done", c);

        let new_values: Vec<u32> = (0..churn_n as u32).map(|i| c * 1_000_000 + i).collect();
        for (ci, (kc, vc)) in churn_keys.chunks(128).zip(new_values.chunks(128)).enumerate() {
            map.insert_bulk(kc, vc, &module, &stream).expect("reinsert");
            stream.synchronize().expect("sync chunk");
        }
        println!("cycle {} reinsert done", c);
    }
    println!("REPRO PASSED: 10 churn cycles completed");
}
