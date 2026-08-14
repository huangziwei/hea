//! Which BLAS `--features blas` links, whether one is reachable at all, and
//! where to find it.
//!
//! Two cfgs come out of here and they mean different things:
//!
//! - **`accelerate`** — the backend is Apple's, so `sparse::blas` links a
//!   framework and uses unprefixed symbol names. macOS, unless `blas-openblas`
//!   overrides it.
//! - **`vendor_blas`** — a backend was actually *found*. `sparse::blas` and
//!   `sparse::dense`'s twelve routing guards key off this one.
//!
//! The split is the whole point. `blas` is on by default, and a Cargo default
//! feature that can fail to link would make `pip install` from the sdist fail on
//! any Linux box without OpenBLAS. So the feature says "use the platform's best
//! backend"; this file answers "is there one", and when there is not, the build
//! succeeds with the portable NEON kernels and `build_info()["backend"]` reports
//! `None`. Pass `blas-required` to turn the fallback into a hard error, which is
//! what the wheel jobs do — a release wheel that silently shipped the slow path
//! is not something a green CI run would otherwise show.
//!
//! Accelerate is a system framework and needs no search path. OpenBLAS does,
//! and is looked for in this order:
//!
//! 1. `HEA_BLAS_LIB_DIR`, for a build that already has one. This is also how a
//!    plain system install is used: point it at the directory.
//! 2. `pkg-config --libs-only-L scipy-openblas`, which is what the
//!    `scipy-openblas32` wheel installs; point `PKG_CONFIG_PATH` at its
//!    `lib/pkgconfig`.
//!
//! Neither found means no vendor path, rather than a link attempt against the
//! default search path — an implicit `-lscipy_openblas` that may or may not
//! resolve is exactly the unpredictability the availability cfg exists to
//! remove, and it would resolve at link time and vanish at load time as often
//! as not.
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
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rustc-check-cfg=cfg(accelerate)");
    println!("cargo:rustc-check-cfg=cfg(vendor_blas)");
    if std::env::var_os("CARGO_FEATURE_BLAS").is_none() {
        return;
    }
    let required = std::env::var_os("CARGO_FEATURE_BLAS_REQUIRED").is_some();

    let macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    if macos && std::env::var_os("CARGO_FEATURE_BLAS_OPENBLAS").is_none() {
        println!("cargo:rustc-cfg=accelerate");
        println!("cargo:rustc-cfg=vendor_blas");
        return;
    }

    let dirs = openblas_dirs();
    if dirs.is_empty() {
        if required {
            panic!(
                "feature `blas-required` is on and no OpenBLAS was found. Set \
                 HEA_BLAS_LIB_DIR to the directory holding libscipy_openblas, or \
                 PKG_CONFIG_PATH to the `lib/pkgconfig` of an installed \
                 scipy-openblas32 wheel."
            );
        }
        /* No `vendor_blas`: the portable kernels stay, the build succeeds, and
         * `build_info()` says `backend: None` so it is not silent. */
        println!(
            "cargo:warning=hea: no OpenBLAS found (HEA_BLAS_LIB_DIR / \
             pkg-config scipy-openblas); building the portable NEON kernels. \
             build_info()['backend'] will be None."
        );
        return;
    }

    let windows = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    for dir in dirs {
        println!("cargo:rustc-link-search=native={dir}");
        if !windows {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
    }
    println!("cargo:rustc-cfg=vendor_blas");
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
