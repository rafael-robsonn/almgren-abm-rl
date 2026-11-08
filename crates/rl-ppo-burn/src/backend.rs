use burn::tensor::Device;

/// Execution device by name; the matching Cargo feature must be enabled.
pub fn device(name: &str) -> Device {
    match name {
        #[cfg(feature = "ndarray")]
        "ndarray" => {
            #[allow(deprecated)]
            let d = Device::ndarray();
            d
        }
        #[cfg(feature = "flex")]
        "flex" => Device::flex(),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(Default::default()),
        other => panic!("backend `{other}` is not compiled in (features: ndarray, flex, wgpu)"),
    }
}

pub fn default_backend() -> &'static str {
    if cfg!(feature = "flex") {
        "flex"
    } else if cfg!(feature = "ndarray") {
        "ndarray"
    } else {
        "wgpu"
    }
}
