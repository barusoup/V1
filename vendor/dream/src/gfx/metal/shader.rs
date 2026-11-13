//! WGSL -> MSL compilation for the native Metal backend.
//!
//! Translates WGSL shaders to Metal Shading Language (MSL) via `naga::back::msl`
//! with explicit `(group, binding)` -> Metal slot mappings and caches the
//! compiled `MTLLibrary` objects on `GpuContext`.

use naga::back::msl;
use std::collections::BTreeMap;

/// Metal buffer slot reserved for `var<push_constant>` blocks.
/// Slots `0..27` are available for uniform/storage buffers; slot `29` is
/// reserved for the vertex buffer (`[[stage_in]]`).
pub const PUSH_CONSTANT_BUFFER_SLOT: u8 = 28;
pub const VERTEX_BUFFER_SLOT: u8 = 29;

/// Declarative mapping from WGSL `@group(g) @binding(b)` to Metal resource slots.
#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct MslBindings {
    pub buffers: Vec<(u32, u32, u8, bool)>,
    pub textures: Vec<(u32, u32, u8)>,
    pub samplers: Vec<(u32, u32, u8)>,
    pub push_constant_slot: Option<u8>,
}

impl MslBindings {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn buffer(mut self, group: u32, binding: u32, slot: u8, mutable: bool) -> Self {
        self.buffers.push((group, binding, slot, mutable));
        self
    }

    pub fn texture(mut self, group: u32, binding: u32, slot: u8) -> Self {
        self.textures.push((group, binding, slot));
        self
    }

    pub fn sampler(mut self, group: u32, binding: u32, slot: u8) -> Self {
        self.samplers.push((group, binding, slot));
        self
    }

    pub fn push_constants(mut self, slot: u8) -> Self {
        self.push_constant_slot = Some(slot);
        self
    }

    pub fn hash_key(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mix = |h: &mut u64, v: u64| {
            *h = (*h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
        };
        for &(g, b, s, m) in &self.buffers {
            mix(&mut h, g as u64);
            mix(&mut h, b as u64);
            mix(&mut h, s as u64);
            mix(&mut h, m as u64);
        }
        mix(&mut h, 0xaaaa_bbbb);
        for &(g, b, s) in &self.textures {
            mix(&mut h, g as u64);
            mix(&mut h, b as u64);
            mix(&mut h, s as u64);
        }
        mix(&mut h, 0xcccc_dddd);
        for &(g, b, s) in &self.samplers {
            mix(&mut h, g as u64);
            mix(&mut h, b as u64);
            mix(&mut h, s as u64);
        }
        mix(&mut h, self.push_constant_slot.map_or(255, |s| s as u64));
        h
    }

    fn to_naga_resources(&self) -> msl::BindingMap {
        let mut map = msl::BindingMap::new();
        for &(group, binding, slot, mutable) in &self.buffers {
            map.insert(
                naga::ResourceBinding { group, binding },
                msl::BindTarget {
                    buffer: Some(slot),
                    mutable,
                    ..Default::default()
                },
            );
        }
        for &(group, binding, slot) in &self.textures {
            map.insert(
                naga::ResourceBinding { group, binding },
                msl::BindTarget {
                    texture: Some(slot),
                    ..Default::default()
                },
            );
        }
        for &(group, binding, slot) in &self.samplers {
            map.insert(
                naga::ResourceBinding { group, binding },
                msl::BindTarget {
                    sampler: Some(msl::BindSamplerTarget::Resource(slot)),
                    ..Default::default()
                },
            );
        }
        map
    }
}

/// Compile a WGSL source string into Metal Shading Language (MSL).
pub struct CompiledMsl {
    pub source: String,
    pub entry_points: Vec<(String, String)>,
}

pub fn compile_wgsl_to_msl(source: &str, label: &str, bindings: &MslBindings) -> CompiledMsl {
    compile_wgsl_to_msl_opts(source, label, bindings, true)
}

pub fn compile_wgsl_to_msl_unflipped(
    source: &str,
    label: &str,
    bindings: &MslBindings,
) -> CompiledMsl {
    compile_wgsl_to_msl_opts(source, label, bindings, false)
}

fn compile_wgsl_to_msl_opts(
    source: &str,
    label: &str,
    bindings: &MslBindings,
    flip_y: bool,
) -> CompiledMsl {
    let mut module = naga::front::wgsl::parse_str(source).unwrap_or_else(|err| {
        panic!(
            "{label}: WGSL parse failed:\n{}",
            err.emit_to_string(source)
        );
    });

    if flip_y {
        flip_vertex_y(&mut module);
    }

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .unwrap_or_else(|err| panic!("{label}: WGSL validation failed: {err:?}"));

    let binding_map = bindings.to_naga_resources();
    let mut per_entry_point_map = BTreeMap::new();
    for ep in &module.entry_points {
        per_entry_point_map.insert(
            ep.name.clone(),
            msl::EntryPointResources {
                resources: binding_map.clone(),
                push_constant_buffer: bindings.push_constant_slot,
                sizes_buffer: Some(27),
            },
        );
    }

    let options = msl::Options {
        lang_version: (2, 4),
        per_entry_point_map,
        inline_samplers: Vec::new(),
        spirv_cross_compatibility: false,
        fake_missing_bindings: false,
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Unchecked,
            buffer: naga::proc::BoundsCheckPolicy::Unchecked,
            image_load: naga::proc::BoundsCheckPolicy::Unchecked,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        },
        zero_initialize_workgroup_memory: true,
        force_loop_bounding: false,
    };

    let pipeline_options = msl::PipelineOptions {
        entry_point: None,
        allow_and_force_point_size: false,
        ..Default::default()
    };

    let (mut msl_source, translation_info) =
        msl::write_string(&module, &info, &options, &pipeline_options)
            .unwrap_or_else(|err| panic!("{label}: MSL generation failed: {err:?}"));
    // Impostor cards rasterize only the front faces of the bounding box
    // (CullMode::Back) and every ray-marched hit lies inside the box, so
    // `hit_depth >= rasterized_depth` always. Declaring `[[depth(greater)]]`
    // enables Early-Z / TBDR depth rejection on Apple Silicon.
    if label == "impostor" {
        msl_source = msl_source.replace("[[depth(any)]]", "[[depth(greater)]]");
    }
    let entry_points = module
        .entry_points
        .iter()
        .zip(translation_info.entry_point_names.into_iter())
        .filter_map(|(ep, res)| res.ok().map(|msl_name| (ep.name.clone(), msl_name)))
        .collect();
    CompiledMsl {
        source: msl_source,
        entry_points,
    }
}

/// Negate `@builtin(position).y` in every vertex entry point so Metal's
/// positive-height `MTLViewport` matches Dream's Vulkan negative-viewport
/// convention (`out.pos.y = -1` at top, `+1` at bottom).
fn flip_vertex_y(module: &mut naga::Module) {
    let types = &module.types;
    for ep in &mut module.entry_points {
        if ep.stage != naga::ShaderStage::Vertex {
            continue;
        }
        let Some(result) = ep.function.result.as_ref() else {
            continue;
        };
        let result_ty = result.ty;
        let pos_member = if matches!(
            result.binding,
            Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { .. }))
        ) {
            None
        } else if let naga::TypeInner::Struct { members, .. } = &types[result_ty].inner {
            let Some(idx) = members.iter().position(|m| {
                matches!(
                    m.binding,
                    Some(naga::Binding::BuiltIn(naga::BuiltIn::Position { .. }))
                )
            }) else {
                continue;
            };
            Some(idx as u32)
        } else {
            continue;
        };

        let tmp = ep.function.local_variables.append(
            naga::LocalVariable {
                name: Some("_dream_pos_flip".into()),
                ty: result_ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let tmp_ptr = ep
            .function
            .expressions
            .append(naga::Expression::LocalVariable(tmp), naga::Span::UNDEFINED);

        let mut body = std::mem::take(&mut ep.function.body);
        rewrite_returns(&mut body, &mut ep.function.expressions, tmp_ptr, pos_member);
        ep.function.body = body;
    }
}

fn rewrite_returns(
    block: &mut naga::Block,
    exprs: &mut naga::Arena<naga::Expression>,
    tmp_ptr: naga::Handle<naga::Expression>,
    pos_member: Option<u32>,
) {
    let old = std::mem::take(block);
    for (stmt, span) in old.span_into_iter() {
        match stmt {
            naga::Statement::Return { value: Some(val) } => {
                block.push(
                    naga::Statement::Store {
                        pointer: tmp_ptr,
                        value: val,
                    },
                    span,
                );
                let pos_ptr = match pos_member {
                    Some(idx) => exprs.append(
                        naga::Expression::AccessIndex {
                            base: tmp_ptr,
                            index: idx,
                        },
                        span,
                    ),
                    None => tmp_ptr,
                };
                let y_ptr = exprs.append(
                    naga::Expression::AccessIndex {
                        base: pos_ptr,
                        index: 1,
                    },
                    span,
                );
                let y_val = exprs.append(naga::Expression::Load { pointer: y_ptr }, span);
                let neg_y = exprs.append(
                    naga::Expression::Unary {
                        op: naga::UnaryOperator::Negate,
                        expr: y_val,
                    },
                    span,
                );
                block.push(naga::Statement::Emit(exprs.range_from(y_val.index())), span);
                block.push(
                    naga::Statement::Store {
                        pointer: y_ptr,
                        value: neg_y,
                    },
                    span,
                );
                let ret_val = exprs.append(naga::Expression::Load { pointer: tmp_ptr }, span);
                block.push(
                    naga::Statement::Emit(exprs.range_from(ret_val.index())),
                    span,
                );
                block.push(
                    naga::Statement::Return {
                        value: Some(ret_val),
                    },
                    span,
                );
            }
            naga::Statement::Block(mut inner) => {
                rewrite_returns(&mut inner, exprs, tmp_ptr, pos_member);
                block.push(naga::Statement::Block(inner), span);
            }
            naga::Statement::If {
                condition,
                mut accept,
                mut reject,
            } => {
                rewrite_returns(&mut accept, exprs, tmp_ptr, pos_member);
                rewrite_returns(&mut reject, exprs, tmp_ptr, pos_member);
                block.push(
                    naga::Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    span,
                );
            }
            naga::Statement::Loop {
                mut body,
                mut continuing,
                break_if,
            } => {
                rewrite_returns(&mut body, exprs, tmp_ptr, pos_member);
                rewrite_returns(&mut continuing, exprs, tmp_ptr, pos_member);
                block.push(
                    naga::Statement::Loop {
                        body,
                        continuing,
                        break_if,
                    },
                    span,
                );
            }
            naga::Statement::Switch {
                selector,
                mut cases,
            } => {
                for case in &mut cases {
                    rewrite_returns(&mut case.body, exprs, tmp_ptr, pos_member);
                }
                block.push(naga::Statement::Switch { selector, cases }, span);
            }
            other => block.push(other, span),
        }
    }
}
