/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MXMACA device-binary generation.

use crate::error::PipelineError;
use crate::options::BackendOptions;
use std::io::Read;
use std::path::{Path, PathBuf};

pub(crate) const DEFAULT_MACA_TARGET: &str = "xcore1000";
const ELF64_HEADER_SIZE: usize = 64;
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const ET_DYN: u16 = 3;
const EM_METAX: u16 = 0x00fd;

#[derive(Debug)]
pub(crate) struct GeneratedMacaDeviceBinary {
    pub target: String,
    pub diagnostics: Vec<String>,
}

pub(crate) fn generate_maca_device_binary(
    llvm_ir_path: &Path,
    output_path: &Path,
    options: &BackendOptions,
) -> Result<GeneratedMacaDeviceBinary, PipelineError> {
    match std::fs::remove_file(output_path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(PipelineError::MacaGeneration(format!(
                "failed to remove stale device binary {}: {error}",
                output_path.display()
            )));
        }
    }
    let target = resolve_maca_target(options.target_arch.as_deref())?;
    let mxcc = resolve_mxcc(options);
    let mut command = std::process::Command::new(&mxcc);
    command
        .arg("-x")
        .arg("ir")
        .arg("-input-is-device")
        .arg(format!("-offload-arch={target}"))
        .arg("-device-bin")
        .arg(llvm_ir_path)
        .arg("-o")
        .arg(output_path)
        .arg(if options.no_fma {
            "-fmad=false"
        } else {
            "-fmad=true"
        });
    if let Some(maca_path) = options.maca_path.as_deref() {
        command.arg(format!("--maca-path={}", maca_path.display()));
    }

    let result = run_mxcc(&mut command).map_err(|error| {
        PipelineError::MacaGeneration(format!(
            "failed to run {} for {}: {error}",
            mxcc.display(),
            llvm_ir_path.display()
        ))
    })?;

    if !result.status.success() {
        let _ = std::fs::remove_file(output_path);
        let stderr = String::from_utf8_lossy(&result.stderr);
        let stdout = String::from_utf8_lossy(&result.stdout);
        let details = if stderr.trim().is_empty() {
            stdout.trim()
        } else {
            stderr.trim()
        };
        return Err(PipelineError::MacaGeneration(format!(
            "{} failed with status {}{}{}",
            mxcc.display(),
            result.status,
            if details.is_empty() { "" } else { ":\n" },
            details
        )));
    }

    let output_size = std::fs::metadata(output_path)
        .map_err(|error| {
            PipelineError::MacaGeneration(format!(
                "{} reported success but did not produce {}: {error}",
                mxcc.display(),
                output_path.display()
            ))
        })?
        .len();
    if output_size == 0 {
        let _ = std::fs::remove_file(output_path);
        return Err(PipelineError::MacaGeneration(format!(
            "{} produced an empty device binary at {}",
            mxcc.display(),
            output_path.display()
        )));
    }
    if let Err(reason) = validate_maca_device_binary(output_path) {
        let _ = std::fs::remove_file(output_path);
        return Err(PipelineError::MacaGeneration(format!(
            "{} produced an invalid MXMACA device binary at {}: {reason}",
            mxcc.display(),
            output_path.display()
        )));
    }

    let diagnostics = options
        .verbose
        .then(|| {
            format!(
                "mxcc device binary: {} ({} bytes, target: {target})",
                output_path.display(),
                output_size
            )
        })
        .into_iter()
        .collect();
    Ok(GeneratedMacaDeviceBinary {
        target: target.to_string(),
        diagnostics,
    })
}

fn run_mxcc(command: &mut std::process::Command) -> std::io::Result<std::process::Output> {
    const ETXTBSY: i32 = 26;
    for attempt in 0..3 {
        match command.output() {
            Err(error) if error.raw_os_error() == Some(ETXTBSY) && attempt < 2 => {
                std::thread::yield_now();
            }
            result => return result,
        }
    }
    unreachable!("the final command attempt always returns")
}

fn resolve_maca_target(target_arch: Option<&str>) -> Result<&str, PipelineError> {
    match target_arch {
        None | Some(DEFAULT_MACA_TARGET) => Ok(DEFAULT_MACA_TARGET),
        Some(target) => Err(PipelineError::MacaGeneration(format!(
            "unsupported MXMACA target `{target}`; only `{DEFAULT_MACA_TARGET}` is supported"
        ))),
    }
}

fn validate_maca_device_binary(path: &Path) -> Result<(), String> {
    let mut header = [0_u8; ELF64_HEADER_SIZE];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut header))
        .map_err(|error| format!("could not read the complete ELF64 header: {error}"))?;

    if header[..4] != *b"\x7fELF" {
        return Err("missing ELF magic".to_string());
    }
    if header[4] != ELFCLASS64 {
        return Err(format!(
            "expected ELF64 class {ELFCLASS64}, found {}",
            header[4]
        ));
    }
    if header[5] != ELFDATA2LSB {
        return Err(format!(
            "expected little-endian ELF data encoding {ELFDATA2LSB}, found {}",
            header[5]
        ));
    }

    let elf_type = u16::from_le_bytes([header[16], header[17]]);
    if elf_type != ET_DYN {
        return Err(format!(
            "expected ET_DYN (e_type=0x{ET_DYN:04x}), found e_type=0x{elf_type:04x}"
        ));
    }
    let machine = u16::from_le_bytes([header[18], header[19]]);
    if machine != EM_METAX {
        return Err(format!(
            "expected MetaX e_machine=0x{EM_METAX:04x}, found e_machine=0x{machine:04x}"
        ));
    }
    Ok(())
}

fn resolve_mxcc(options: &BackendOptions) -> PathBuf {
    if let Some(path) = options.mxcc_override.as_ref() {
        return path.clone();
    }
    if let Some(maca_path) = options.maca_path.as_ref() {
        return maca_path.join("mxgpu_llvm/bin/mxcc");
    }
    PathBuf::from("mxcc")
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn mxcc_device_binary_uses_ir_mode_sdk_and_fma_policy() {
        let root = unique_test_dir("success");
        let paths = root.join("path with spaces");
        std::fs::create_dir(&paths).unwrap();
        let llvm_ir = paths.join("module.ll");
        let output = paths.join("module.devbin");
        let sdk = root.join("sdk");
        let fixture = root.join("valid.devbin");
        let expected = maca_elf64_header(EM_METAX);
        std::fs::create_dir(&sdk).unwrap();
        std::fs::write(&llvm_ir, "target triple = \"mxc-metax-macahca\"\n").unwrap();
        std::fs::write(&fixture, expected).unwrap();
        let mxcc = write_tool(
            &root,
            "mxcc",
            &format!(
                r#"#!/bin/sh
device_bin=0
device_input=0
language=0
arch=0
fmad=0
sdk=0
input=0
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -device-bin) device_bin=1 ;;
    -input-is-device) device_input=1 ;;
    -x) shift; [ "$1" = ir ] && language=1 ;;
    -offload-arch=xcore1000) arch=1 ;;
    -fmad=false) fmad=1 ;;
    "--maca-path={}") sdk=1 ;;
    "{}") input=1 ;;
    -o) shift; out="$1" ;;
  esac
  shift
done
[ "$device_bin" = 1 ] && [ "$device_input" = 1 ] && [ "$language" = 1 ] && \
  [ "$arch" = 1 ] && [ "$fmad" = 1 ] && \
  [ "$sdk" = 1 ] && [ "$input" = 1 ] && [ -n "$out" ] || exit 17
cp "{}" "$out"
"#,
                sdk.display(),
                llvm_ir.display(),
                fixture.display()
            ),
        );
        let options = BackendOptions {
            target_arch: Some(DEFAULT_MACA_TARGET.to_string()),
            no_fma: true,
            verbose: true,
            mxcc_override: Some(mxcc),
            maca_path: Some(sdk),
            ..BackendOptions::default()
        };

        let generated = generate_maca_device_binary(&llvm_ir, &output, &options).unwrap();
        assert_eq!(generated.target, DEFAULT_MACA_TARGET);
        assert_eq!(std::fs::read(&output).unwrap(), expected);
        assert_eq!(generated.diagnostics.len(), 1);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn mxcc_failure_does_not_publish_a_partial_binary() {
        let root = unique_test_dir("failure");
        let llvm_ir = root.join("module.ll");
        let output = root.join("module.devbin");
        std::fs::write(&llvm_ir, "invalid ir\n").unwrap();
        let mxcc = write_tool(
            &root,
            "mxcc",
            r#"#!/bin/sh
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then shift; out="$1"; fi
  shift
done
printf 'partial' > "$out"
echo 'deliberate mxcc failure' >&2
exit 7
"#,
        );
        let options = BackendOptions {
            mxcc_override: Some(mxcc),
            ..BackendOptions::default()
        };

        let error = generate_maca_device_binary(&llvm_ir, &output, &options).unwrap_err();
        assert!(matches!(error, PipelineError::MacaGeneration(_)));
        assert!(error.to_string().contains("deliberate mxcc failure"));
        assert!(!output.exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn mxcc_success_without_an_output_is_rejected() {
        let root = unique_test_dir("missing-output");
        let llvm_ir = root.join("module.ll");
        let output = root.join("module.devbin");
        std::fs::write(&llvm_ir, "target triple = \"mxc-metax-macahca\"\n").unwrap();
        std::fs::write(&output, b"\x7fELFstale").unwrap();
        let mxcc = write_tool(&root, "mxcc", "#!/bin/sh\nexit 0\n");
        let options = BackendOptions {
            mxcc_override: Some(mxcc),
            ..BackendOptions::default()
        };

        let error = generate_maca_device_binary(&llvm_ir, &output, &options).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("reported success but did not produce")
        );
        assert!(!output.exists());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn unsupported_maca_target_is_rejected_before_mxcc_runs() {
        let root = unique_test_dir("unsupported-target");
        let llvm_ir = root.join("module.ll");
        let output = root.join("module.devbin");
        let marker = root.join("mxcc-ran");
        std::fs::write(&llvm_ir, "target triple = \"mxc-metax-macahca\"\n").unwrap();
        std::fs::write(&output, maca_elf64_header(EM_METAX)).unwrap();
        let mxcc = write_tool(
            &root,
            "mxcc",
            &format!("#!/bin/sh\ntouch \"{}\"\nexit 0\n", marker.display()),
        );
        let options = BackendOptions {
            target_arch: Some("sm_90".to_string()),
            mxcc_override: Some(mxcc),
            ..BackendOptions::default()
        };

        let error = generate_maca_device_binary(&llvm_ir, &output, &options).unwrap_err();
        assert!(error.to_string().contains("only `xcore1000` is supported"));
        assert!(!marker.exists());
        assert!(!output.exists(), "stale output must be invalidated");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn host_elf_is_not_accepted_as_a_maca_device_binary() {
        let root = unique_test_dir("host-elf");
        let llvm_ir = root.join("module.ll");
        let output = root.join("module.devbin");
        let host_elf = root.join("host.so");
        std::fs::write(&llvm_ir, "target triple = \"mxc-metax-macahca\"\n").unwrap();
        std::fs::write(&host_elf, maca_elf64_header(0x003e)).unwrap();
        let mxcc = write_copying_tool(&root, &host_elf);
        let options = BackendOptions {
            mxcc_override: Some(mxcc),
            ..BackendOptions::default()
        };

        let error = generate_maca_device_binary(&llvm_ir, &output, &options).unwrap_err();
        assert!(error.to_string().contains("e_machine=0x003e"));
        assert!(!output.exists(), "invalid output must not be published");

        let _ = std::fs::remove_dir_all(root);
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let counter = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_mxcc_test_{}_{}_{}",
            std::process::id(),
            label,
            counter
        ));
        std::fs::create_dir(&root).unwrap();
        root
    }

    fn write_tool(root: &Path, name: &str, contents: &str) -> PathBuf {
        let path = root.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(contents.as_bytes()).unwrap();
        file.sync_all().unwrap();
        drop(file);
        let mut permissions = std::fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&path, permissions).unwrap();
        path
    }

    fn write_copying_tool(root: &Path, source: &Path) -> PathBuf {
        write_tool(
            root,
            "mxcc",
            &format!(
                r#"#!/bin/sh
out=""
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then shift; out="$1"; fi
  shift
done
cp "{}" "$out"
"#,
                source.display()
            ),
        )
    }

    fn maca_elf64_header(machine: u16) -> [u8; ELF64_HEADER_SIZE] {
        let mut header = [0_u8; ELF64_HEADER_SIZE];
        header[..4].copy_from_slice(b"\x7fELF");
        header[4] = ELFCLASS64;
        header[5] = ELFDATA2LSB;
        header[6] = 1;
        header[16..18].copy_from_slice(&ET_DYN.to_le_bytes());
        header[18..20].copy_from_slice(&machine.to_le_bytes());
        header[20..24].copy_from_slice(&1_u32.to_le_bytes());
        header
    }
}
