/*
 * Build script for maca-bindings
 *
 * Generates Rust FFI bindings to MXMACA Runtime API using bindgen.
 */

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=MACA_PATH");

    let maca_path = env::var("MACA_PATH").unwrap_or_else(|_| "/opt/maca".to_string());
    let include_dir = PathBuf::from(&maca_path).join("include").join("mcr");
    let misc_dir = PathBuf::from(&maca_path).join("include");

    if !include_dir.exists() {
        panic!(
            "MXMACA include directory not found at: {}\n\
             Set MACA_PATH environment variable to MXMACA installation root.",
            include_dir.display()
        );
    }

    // Link to MXMACA runtime library
    let lib_dir = PathBuf::from(&maca_path).join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=mcruntime");

    // Generate bindings
    let bindings = bindgen::builder()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", misc_dir.display()))
        .clang_arg("-I/usr/lib/gcc/x86_64-linux-gnu/11/include")
        .clang_arg("-I/usr/include")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Allow only mc* types and functions
        .allowlist_type("mc.*")
        .allowlist_function("mc.*")
        .allowlist_var("mc.*|MC.*")
        // Generate types
        .generate()
        .expect("Failed to generate MXMACA bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");
}
