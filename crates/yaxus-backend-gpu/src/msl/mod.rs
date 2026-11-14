//! Kernels written in Metal Shading Language, compiled through wgpu's passthrough on Metal
//! (`Features::PASSTHROUGH_SHADERS`). wgpu binds the immediates at `buffer(0)` and binding
//! `i` of bind group 0 at `buffer(i + 1)`, and sets no threadgroup memory length, so the
//! kernels declare their threadgroup arrays inside the function. A source starts with the
//! line `// msl workgroup_size=N` (see `dev::Compiler::compile`).
//!
//! Each source starts with `#pragma METAL fp math_mode(safe)`: Metal compiles WGSL with
//! fast math, which wgpu cannot turn off; here the compiler keeps the arithmetic as written,
//! so kernels agree to the bit without the `keep` and `fma` precautions of the WGSL ones.

mod access;
mod gemm;
mod gemv2;
mod small;

pub(crate) use gemm::{Gemm, gemm_reads};
pub(crate) use gemv2::{Gemv2, gemv2_reads};
pub(crate) use small::{add_norm as add_norm_source, qk_rope as qk_rope_source, rmsnorm as rmsnorm_source};

/// The marker line of an MSL source and its workgroup size.
pub(crate) fn header(workgroup_size: u32) -> String {
    format!("// msl workgroup_size={workgroup_size}\n")
}

/// The workgroup size of `src` if it is an MSL source.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn workgroup_size(src: &str) -> Option<u32> {
    src.lines().next()?.strip_prefix("// msl workgroup_size=")?.trim().parse().ok()
}

pub(super) const PRELUDE: &str = "#include <metal_stdlib>
#pragma METAL fp math_mode(safe)
using namespace metal;
struct P { uint n; uint k; uint t; uint r0; };
";
