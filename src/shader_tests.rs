//! Bevy's shader cache with naga standing in for the GPU device, for the
//! tests that compose and validate the meadow shaders.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy::asset::{AssetId, uuid::Uuid};
use bevy::shader::{
    Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ShaderDefVal, ValidateShader,
};

/// The bevy checkout the tests read the real shader library from:
/// `BEVY_CHECKOUT` if set, else the sibling `../bevy-main` or `../bevy`
/// checkout. `None` (test skips) when none holds a bevy source tree.
#[cfg_attr(
    not(feature = "mesh-shaders"),
    expect(
        dead_code,
        reason = "the mesh path's tests compose against bevy's shaders"
    )
)]
pub(crate) fn bevy_checkout() -> Option<PathBuf> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = match std::env::var_os("BEVY_CHECKOUT") {
        Some(root) => vec![PathBuf::from(root)],
        None => vec![manifest.join("../bevy-main"), manifest.join("../bevy")],
    };
    candidates.into_iter().find(|root| {
        root.join("crates/bevy_pbr/src/render/pbr_functions.wesl")
            .exists()
    })
}

/// Bevy's own shader cache, with naga standing in for the GPU device:
/// the composed WGSL is parsed and validated (mesh-shader
/// capabilities included) and kept as the "module" for assertions.
pub(crate) struct Library {
    cache: ShaderCache<String, ()>,
}

fn validate_wgsl(
    _device: &(),
    source: ShaderCacheSource,
    _validate: &ValidateShader,
) -> Result<String, ShaderCacheError> {
    let ShaderCacheSource::Wgsl(wgsl) = source else {
        panic!("wesl compiles to WGSL");
    };
    let module = naga::front::wgsl::parse_str(&wgsl)
        .map_err(|e| ShaderCacheError::ProcessShaderError(e.emit_to_string(&wgsl)))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| ShaderCacheError::ProcessShaderError(format!("{e:?}")))?;
    Ok(wgsl)
}

impl Library {
    /// The meadow shaders alone — enough for the geometry module.
    pub(crate) fn meadow() -> Self {
        let mut lib = Self {
            cache: ShaderCache::new((), validate_wgsl),
        };
        for (source, path) in [
            (
                include_str!("meadow_shared.wesl"),
                "embedded://bevy_meadow/meadow_shared.wesl",
            ),
            (
                include_str!("meadow_mesh_bindings.wesl"),
                "embedded://bevy_meadow/meadow_mesh_bindings.wesl",
            ),
        ] {
            lib.add(Shader::from_wesl(source, path));
        }
        lib
    }

    /// The meadow shaders (PBR-common module included) plus bevy's
    /// shader crates from the checkout, with the loader-settings defs
    /// bevy registers (`mesh_view_types.wesl` in
    /// `MeshRenderPlugin::build`) carried on the Shader asset for the
    /// def closure. `None` (tests skip) without a bevy checkout.
    #[cfg_attr(
        not(feature = "mesh-shaders"),
        expect(
            dead_code,
            reason = "the mesh path's tests compose against bevy's shaders"
        )
    )]
    pub(crate) fn with_bevy() -> Option<Self> {
        let root = bevy_checkout()?;
        let mut lib = Self::meadow();
        lib.add(Shader::from_wesl(
            include_str!("meadow_mesh_pbr_common.wesl"),
            "embedded://bevy_meadow/meadow_mesh_pbr_common.wesl",
        ));
        for crate_name in ["bevy_pbr", "bevy_render", "bevy_core_pipeline"] {
            let src = root.join("crates").join(crate_name).join("src");
            lib.add_wesl_dir(crate_name, &src, &src);
        }
        Some(lib)
    }

    pub(crate) fn add(&mut self, shader: Shader) -> AssetId<Shader> {
        let id = AssetId::Uuid {
            uuid: Uuid::new_v4(),
        };
        self.cache.set_shader(id, shader);
        id
    }

    /// Register every `.wesl` under `dir` the way `load_shader_library!`
    /// does at runtime: `embedded://<crate>/<path-under-src>` paths, so
    /// module names and import scanning match the real asset registry.
    #[cfg_attr(
        not(feature = "mesh-shaders"),
        expect(
            dead_code,
            reason = "the mesh path's tests compose against bevy's shaders"
        )
    )]
    pub(crate) fn add_wesl_dir(&mut self, crate_name: &str, src_root: &Path, dir: &Path) {
        for entry in std::fs::read_dir(dir).expect("readable bevy source dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                self.add_wesl_dir(crate_name, src_root, &path);
            } else if path.extension().is_some_and(|e| e == "wesl") {
                let rel = path
                    .strip_prefix(src_root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                let source = std::fs::read_to_string(&path).expect("readable wesl file");
                let mut shader =
                    Shader::from_wesl(source, format!("embedded://{crate_name}/{rel}"));
                if rel == "render/mesh_view_types.wesl" {
                    shader.shader_defs = vec![
                        ShaderDefVal::UInt("MAX_DIRECTIONAL_LIGHTS".into(), 10),
                        ShaderDefVal::UInt("MAX_CASCADES_PER_LIGHT".into(), 4),
                        ShaderDefVal::UInt("MAX_RECT_LIGHTS".into(), 8),
                    ];
                }
                self.add(shader);
            }
        }
    }

    /// Compile `shader` with `defs` (plus the device-global defs bevy's
    /// `PipelineCache` splices onto every shader), panicking on any
    /// error, and return the validated WGSL.
    pub(crate) fn compile(
        &mut self,
        shader: AssetId<Shader>,
        defs: &[ShaderDefVal],
    ) -> Arc<String> {
        let mut defs = defs.to_vec();
        defs.extend([
            ShaderDefVal::UInt("AVAILABLE_STORAGE_BUFFER_BINDINGS".into(), 12),
            ShaderDefVal::Bool("AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_3".into(), true),
            ShaderDefVal::Bool("AVAILABLE_STORAGE_BUFFER_BINDINGS__GE_6".into(), true),
        ]);
        match self.cache.get(0, shader, &defs) {
            Ok(wgsl) => wgsl,
            Err(ShaderCacheError::ShaderImportNotYetAvailable) => {
                panic!("an import is missing from a full checkout (see the warning above)")
            }
            Err(err) => panic!("{err}"),
        }
    }
}
