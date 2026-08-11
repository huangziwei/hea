//! Which BLAS `--features blas` links, and where to find it.
//!
//! Sets the `accelerate` cfg when the target is macOS and `blas-openblas` has
//! not overridden it; `sparse::blas` keys its `link` and `link_name` attributes
//! off that one predicate so the choice is made in exactly one place.
//!
//! Accelerate is a system framework and needs no search path. OpenBLAS does,
//! and is looked for in this order:
//!
//! 1. `HEA_BLAS_LIB_DIR`, for a build that already has one.
//! 2. `pkg-config --libs-only-L scipy-openblas`, which is what the
//!    `scipy-openblas32` wheel installs; point `PKG_CONFIG_PATH` at its
//!    `lib/pkgconfig`.
//!
//! Both fall through to plain `-lscipy_openblas` on the default search path, so
//! a system install works too and the failure, if there is one, is the linker's
//! and names the library.
//!
//! On the ELF and Mach-O targets an rpath is emitted alongside the search path,
//! because the library is normally a wheel's rather than the system's and would
//! otherwise be found at link time and missing at load time. **Not on Windows**:
//! `link.exe` has no `-Wl,-rpath` and would reject it, and a DLL is found by
//! search path there anyway. `cargo check` never links, so it cannot catch that
//! — every `rustc-link-arg` is a per-linker decision and has to be read as one.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=HEA_BLAS_LIB_DIR");
    println!("cargo:rustc-check-cfg=cfg(accelerate)");
    if std::env::var_os("CARGO_FEATURE_BLAS").is_none() {
        return;
    }
    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if macos && std::env::var_os("CARGO_FEATURE_BLAS_OPENBLAS").is_none() {
        println!("cargo:rustc-cfg=accelerate");
        return;
    }
    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    for dir in openblas_dirs() {
        println!("cargo:rustc-link-search=native={dir}");
        if !windows {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
}

fn openblas_dirs() -> Vec<String> {
    if let Some(dir) = std::env::var_os("HEA_BLAS_LIB_DIR") {
        return vec![dir.to_string_lossy().into_owned()];
    }
    let out = Command::new("pkg-config")
        .args(["--libs-only-L", "scipy-openblas"])
        .output();
    match out {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .split_whitespace()
            .filter_map(|t| t.strip_prefix("-L").map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}
