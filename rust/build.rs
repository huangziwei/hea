//! Where `--features blas` finds a BLAS to link against.
//!
//! Nothing to do without that feature, and nothing to do on macOS, where
//! Accelerate is a system framework the `link` attribute names directly.
//!
//! Elsewhere the library is OpenBLAS and the build has to be told where it is,
//! in this order:
//!
//! 1. `HEA_BLAS_LIB_DIR`, for a build that already has one.
//! 2. `pkg-config --libs-only-L scipy-openblas`, which is what the
//!    `scipy-openblas32` wheel installs. Point `PKG_CONFIG_PATH` at its
//!    `lib/pkgconfig` — `python -c "import scipy_openblas32 as s;
//!    print(s.get_lib_dir())"` gives the sibling directory.
//!
//! Both fall through to plain `-lscipy_openblas` on the default search path, so
//! a system install works too and the failure, if there is one, is the linker's
//! and names the library.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=HEA_BLAS_LIB_DIR");
    if std::env::var_os("CARGO_FEATURE_BLAS").is_none() {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        return;
    }
    if let Some(dir) = std::env::var_os("HEA_BLAS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir.to_string_lossy());
        return;
    }
    let out = Command::new("pkg-config")
        .args(["--libs-only-L", "scipy-openblas"])
        .output();
    if let Ok(out) = out {
        if out.status.success() {
            for tok in String::from_utf8_lossy(&out.stdout).split_whitespace() {
                if let Some(dir) = tok.strip_prefix("-L") {
                    println!("cargo:rustc-link-search=native={dir}");
                }
            }
        }
    }
}
