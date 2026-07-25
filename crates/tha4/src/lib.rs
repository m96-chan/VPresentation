//! THA4 character-animation engine built on the candle fork.
//!
//! Phase 0: device selection + foundation. Later phases add the `grid_sample`
//! op, the five THA4 networks, and the pose pipeline (see issue #4).

use candle_core::Device;

pub mod image_io;
pub mod poser;
pub mod warp;

/// Pick the best available compute device: Metal (Apple GPU) if compiled with
/// the `metal` feature and a Metal device is present, otherwise CPU.
///
/// CUDA is intentionally not attempted here — the supported CUDA path is
/// selected explicitly elsewhere; on Apple Silicon we target Metal.
pub fn best_device() -> candle_core::Result<Device> {
    #[cfg(feature = "metal")]
    {
        // candle-metal recycles pooled buffers that may still be referenced by
        // in-flight GPU kernels, which corrupts large graphs (THA4 upscaler).
        // Disable reuse for correctness before any Metal buffer is allocated.
        if std::env::var_os("CANDLE_METAL_NO_BUFFER_REUSE").is_none() {
            std::env::set_var("CANDLE_METAL_NO_BUFFER_REUSE", "1");
        }
        match Device::new_metal(0) {
            Ok(dev) => return Ok(dev),
            Err(e) => {
                eprintln!("[tha4] Metal unavailable ({e}); falling back to CPU");
            }
        }
    }
    Ok(Device::Cpu)
}

/// Human-readable name of a device, for logging/diagnostics.
pub fn device_label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "cpu",
        Device::Cuda(_) => "cuda",
        Device::Metal(_) => "metal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn device_runs_a_real_op() -> candle_core::Result<()> {
        let device = best_device()?;
        // A 2x2 matmul that must actually execute on the selected backend.
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
        let b = Tensor::new(&[[5.0f32, 6.0], [7.0, 8.0]], &device)?;
        let c = (a.matmul(&b))?.to_vec2::<f32>()?;
        assert_eq!(c, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
        Ok(())
    }
}
