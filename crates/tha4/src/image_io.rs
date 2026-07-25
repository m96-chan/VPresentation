//! PNG <-> THA4 tensor conversion, matching THA4's exact preprocessing.
//!
//! Load:  srgb->linear on RGB, premultiply alpha, then `v*2 - 1` on all 4
//!        channels; layout CHW -> tensor (1, 4, H, W).
//! Save:  inverse — `(v+1)/2`, straight (un-premultiply) alpha, linear->srgb.

use candle_core::{Device, Tensor};

fn srgb_to_linear(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.003_130_805 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// Load an RGBA PNG into a THA4 input tensor `(1, 4, size, size)` in [-1, 1].
/// The image is resized to `size` x `size` if needed (nearest — assumes it is
/// already correctly framed; real framing happens in preprocessing).
pub fn load_thaa_image(path: &str, size: u32, device: &Device) -> anyhow::Result<Tensor> {
    let img = image::open(path)?.to_rgba8();
    let img = if img.width() != size || img.height() != size {
        image::imageops::resize(&img, size, size, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut chw = vec![0f32; 4 * h * w];
    for (x, y, px) in img.enumerate_pixels() {
        let (x, y) = (x as usize, y as usize);
        let r = px[0] as f32 / 255.0;
        let g = px[1] as f32 / 255.0;
        let b = px[2] as f32 / 255.0;
        let a = (px[3] as f32 / 255.0).clamp(0.0, 1.0);
        // srgb->linear, premultiply, then *2-1
        let r = srgb_to_linear(r) * a * 2.0 - 1.0;
        let g = srgb_to_linear(g) * a * 2.0 - 1.0;
        let b = srgb_to_linear(b) * a * 2.0 - 1.0;
        let a = a * 2.0 - 1.0;
        let base = y * w + x;
        chw[base] = r;
        chw[h * w + base] = g;
        chw[2 * h * w + base] = b;
        chw[3 * h * w + base] = a;
    }
    Ok(Tensor::from_vec(chw, (1, 4, h, w), device)?)
}

/// Save a THA4 output tensor `(1, 4, H, W)` in [-1, 1] to an RGBA PNG.
pub fn save_thaa_image(t: &Tensor, path: &str) -> anyhow::Result<()> {
    let t = t.to_device(&Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
    let (_, c, h, w) = t.dims4()?;
    anyhow::ensure!(c == 4, "expected 4 channels, got {c}");
    let data = t.flatten_all()?.to_vec1::<f32>()?;
    let plane = h * w;
    let mut img = image::RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let to01 = |v: f32| (v + 1.0) * 0.5;
            let a = to01(data[3 * plane + idx]).clamp(0.0, 1.0);
            let unmul = |v: f32| if a < 1e-5 { 0.0 } else { (to01(v) / a).clamp(0.0, 1.0) };
            let r = linear_to_srgb(unmul(data[idx]));
            let g = linear_to_srgb(unmul(data[plane + idx]));
            let b = linear_to_srgb(unmul(data[2 * plane + idx]));
            img.put_pixel(
                x as u32,
                y as u32,
                image::Rgba([
                    (r * 255.0).round() as u8,
                    (g * 255.0).round() as u8,
                    (b * 255.0).round() as u8,
                    (a * 255.0).round() as u8,
                ]),
            );
        }
    }
    img.save(path)?;
    Ok(())
}
