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
