//! WGSL to MSL for the Metal device (naga), with the bindings at the indices the MSL
//! kernels of `msl` use, so that both kinds of kernel are dispatched alike.

use anyhow::{Context, Result, anyhow};

use crate::device::CompileSettings;

/// A kernel's MSL and what the device needs to dispatch it.
#[derive(Debug)]
pub(super) struct Translated {
    pub(super) msl: String,
    /// The entry point's name.
    pub(super) entry: String,
    /// Threads per threadgroup.
    pub(super) threads: usize,
    /// Byte lengths of the threadgroup arguments, by index.
    pub(super) tg_memory: Vec<usize>,
    /// The bindings whose byte lengths naga's `_mslBufferSizes` holds, in order (passed at
    /// the buffer index after the bindings).
    pub(super) sized: Vec<usize>,
}

impl Translated {
    /// An MSL kernel as it is (`msl::header`): entry point `k0`, threadgroup arrays
    /// declared in the function.
    pub(super) fn msl(src: &str, threads: u32) -> Self {
        Translated {
            msl: src.to_string(),
            entry: "k0".into(),
            threads: threads as usize,
            tg_memory: Vec::new(),
            sized: Vec::new(),
        }
    }
}

/// WGSL to MSL with naga: immediates at `buffer(0)`, binding `i` at `buffer(i + 1)`
/// (mutable unless `read_only[i]`), the buffer sizes after the bindings, and the safety code
/// of `settings`.
pub(super) fn translate(src: &str, read_only: &[bool], settings: CompileSettings) -> Result<Translated> {
    use naga::back::msl;
    let module = naga::front::wgsl::parse_str(src).map_err(|e| anyhow!("WGSL: {}", e.emit_to_string(src)))?;
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| anyhow!("validation: {e:?}"))?;
    let ep = module.entry_points.first().context("no entry point")?;
    let mut resources = msl::BindingMap::default();
    for (i, &ro) in read_only.iter().enumerate() {
        resources.insert(
            naga::ResourceBinding { group: 0, binding: i as u32 },
            msl::BindTarget { buffer: Some(i as u8 + 1), mutable: !ro, ..Default::default() },
        );
    }
    let ep_res = msl::EntryPointResources {
        resources,
        immediates_buffer: Some(0),
        sizes_buffer: Some(read_only.len() as u8 + 1),
    };
    let options = msl::Options {
        lang_version: (3, 1),
        per_entry_point_map: msl::EntryPointResourceMap::from([(ep.name.clone(), ep_res)]),
        fake_missing_bindings: false,
        // (as wgpu's `ShaderRuntimeChecks::checked`, which the wgpu device keeps: indices
        // clamped into range, integer division by zero defined)
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Restrict,
            buffer: naga::proc::BoundsCheckPolicy::Restrict,
            image_load: naga::proc::BoundsCheckPolicy::Restrict,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: settings.zero_workgroup_memory,
        force_loop_bounding: settings.loop_bounding,
        emit_int_div_checks: true,
        ..Default::default()
    };
    let pipeline =
        msl::PipelineOptions { entry_point: Some((naga::ShaderStage::Compute, ep.name.clone())), ..Default::default() };
    let (msl_src, tr) = msl::write_string(&module, &info, &options, &pipeline).map_err(|e| anyhow!("MSL: {e:?}"))?;
    let entry = tr.entry_point_names[0].as_ref().map_err(|e| anyhow!("{e}"))?.clone();
    let ep_info = info.get_entry_point(0);
    let tg_memory = module
        .global_variables
        .iter()
        .filter(|(h, v)| v.space == naga::AddressSpace::WorkGroup && !ep_info[*h].is_empty())
        .map(|(_, v)| (module.types[v.ty].inner.size(module.to_ctx()) as usize).next_multiple_of(16))
        .collect();
    // (as naga's `_mslBufferSizes`: every global holding a runtime-sized array, in order)
    let sized = module
        .global_variables
        .iter()
        .filter(|(_, v)| module.types[v.ty].inner.needs_host_buffer_byte_size(&module.types))
        .map(|(_, v)| v.binding.as_ref().map_or(0, |b| b.binding as usize))
        .collect();
    let threads = ep.workgroup_size.iter().product::<u32>() as usize;
    Ok(Translated { msl: msl_src, entry, threads, tg_memory, sized })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SETTINGS: CompileSettings = CompileSettings { loop_bounding: false, zero_workgroup_memory: false };

    #[test]
    fn bindings_take_the_msl_kernels_indices() {
        let src = "
struct P { n: u32, a: u32, b: u32, c: u32 }
var<immediate> p: P;
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
var<workgroup> used: array<f32, 3>;
var<workgroup> unused: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32) {
    used[li % 3u] = x[li];
    workgroupBarrier();
    if (li < p.n) { y[li] = used[0] + f32(arrayLength(&x)); }
}
";
        let t = translate(src, &[false, true], SETTINGS).unwrap();
        assert_eq!(t.threads, 64);
        assert!(t.msl.contains(&format!("kernel void {}(", t.entry)), "{}", t.msl);
        for i in 0..=3 {
            assert!(t.msl.contains(&format!("[[buffer({i})]]")), "buffer({i}) missing:\n{}", t.msl);
        }
        assert!(!t.msl.contains("[[buffer(4)]]"));
        // Only the workgroup memory the entry point uses, in 16-byte multiples.
        assert_eq!(t.tg_memory, vec![16]);
        // Both bindings hold runtime-sized arrays: their lengths follow them, at buffer 3.
        assert_eq!(t.sized, vec![0, 1]);
    }

    #[test]
    fn invalid_wgsl_is_an_error() {
        assert!(translate("fn main( {", &[], SETTINGS).is_err());
    }
}
