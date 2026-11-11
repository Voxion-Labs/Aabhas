//! Build-script entry point for the Aabhas Rust safety library.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=AABHAS_CPP_LIB_DIR");

    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let default_dir = manifest_dir.join("..").join("build");
    let library_dir = env::var_os("AABHAS_CPP_LIB_DIR")
        .map(PathBuf::from)
        .unwrap_or(default_dir);

    if !library_dir.join(static_library_name()).is_file() {
        panic!(
            "Aabhas C++ data-plane library not found at {}. Build core-engine with CMake first or set AABHAS_CPP_LIB_DIR.",
            library_dir.display()
        );
    }

    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=static=aabhas_data_plane");

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=pthread");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
}

fn static_library_name() -> &'static str {
    if cfg!(target_env = "msvc") {
        "aabhas_data_plane.lib"
    } else {
        "libaabhas_data_plane.a"
    }
}
