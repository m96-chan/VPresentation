//! THA4 poser: chains the 5 ONNX networks into `image + pose -> posed image`,
//! reproducing FiveStepPoserComputationProtocol (mode_07).
//!
//! Pose vector = 45 params: eyebrow[0..12], face[12..39], rotation[39..45].

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_onnx::onnx::ModelProto;

use crate::warp::{grid_sample_bilinear_border, identity_grid};

pub const NUM_POSE_PARAMS: usize = 45;

// Output indices within each network's returned list.
const EBD_EYEBROW_LAYER: usize = 0;
const EBD_BACKGROUND_LAYER: usize = 3;
const COMB_EYEBROW_NO_COMBINE_ALPHA: usize = 2;
const MORPHER_MERGED: usize = 0;
const MORPHER_GRID_CHANGE: usize = 3;

pub struct Poser {
    eyebrow_decomposer: ModelProto,
    eyebrow_morphing_combiner: ModelProto,
    face_morpher: ModelProto,
    body_morpher: ModelProto,
    upscaler: ModelProto,
    device: Device,
}

impl Poser {
    /// Load the five exported ONNX models from `dir` (e.g. `data/tha4/onnx`).
    pub fn load(dir: &str, device: Device) -> anyhow::Result<Self> {
        let read = |n: &str| candle_onnx::read_file(format!("{dir}/{n}.onnx"));
        Ok(Self {
            eyebrow_decomposer: read("eyebrow_decomposer")?,
            eyebrow_morphing_combiner: read("eyebrow_morphing_combiner")?,
            face_morpher: read("face_morpher")?,
            body_morpher: read("body_morpher")?,
            upscaler: read("upscaler")?,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run a model, feeding inputs to graph inputs by position (in0, in1, ...),
    /// and return outputs in graph-output order.
    fn run(model: &ModelProto, inputs: &[Tensor]) -> anyhow::Result<Vec<Tensor>> {
        let graph = model.graph.as_ref().expect("graph");
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for (gi, t) in graph.input.iter().zip(inputs.iter()) {
            map.insert(gi.name.clone(), t.clone());
        }
        let timed = std::env::var_os("THA4_TIME").is_some();
        let t0 = timed.then(std::time::Instant::now);
        let out = candle_onnx::simple_eval(model, map)?;
        if let Some(t0) = t0 {
            if let Some(first) = graph.output.first().and_then(|o| out.get(&o.name)) {
                let _ = first.sum_all().and_then(|s| s.to_scalar::<f32>()); // sync
            }
            eprintln!("[pose]   {} took {:?}", graph.name, t0.elapsed());
        }
        let mut result = Vec::with_capacity(graph.output.len());
        for o in graph.output.iter() {
            result.push(
                out.get(&o.name)
                    .ok_or_else(|| anyhow::anyhow!("missing output {}", o.name))?
                    .clone(),
            );
        }
        Ok(result)
    }

    /// Pose a THA4 image `(1, 4, 512, 512)` in [-1, 1] with a 45-dim pose vector.
    pub fn pose(&self, image: &Tensor, pose: &[f32]) -> anyhow::Result<Tensor> {
        anyhow::ensure!(pose.len() == NUM_POSE_PARAMS, "pose must be {NUM_POSE_PARAMS} long");
        let dev = &self.device;
        let sub = |lo: usize, hi: usize| -> anyhow::Result<Tensor> {
            Ok(Tensor::from_vec(pose[lo..hi].to_vec(), (1, hi - lo), dev)?)
        };
        let eyebrow_pose = sub(0, 12)?;
        let face_pose = sub(12, 39)?;
        let rotation_pose = sub(39, 45)?;

        // 1. eyebrow decomposer on the 128x128 eyebrow crop.
        let ebd_in = crop(image, 64, 192, 128)?;
        let ebd = Self::run(&self.eyebrow_decomposer, &[ebd_in])?;

        // 2. eyebrow morphing combiner.
        let comb = Self::run(
            &self.eyebrow_morphing_combiner,
            &[ebd[EBD_BACKGROUND_LAYER].clone(), ebd[EBD_EYEBROW_LAYER].clone(), eyebrow_pose],
        )?;
        let eyebrow_morphed = comb[COMB_EYEBROW_NO_COMBINE_ALPHA].clone();

        // 3. face morpher: 192x192 face crop with the morphed eyebrow pasted in.
        let face_crop = crop(image, 32, 160, 192)?;
        let face_in = place(&face_crop, &eyebrow_morphed, 32, 32)?;
        let fm = Self::run(&self.face_morpher, &[face_in, face_pose])?;
        let face_morphed = fm[0].clone();

        // 4. full-res image with the morphed face region put back.
        let face_full = place(image, &face_morphed, 32, 160)?;

        // 5. half-res for the body morpher.
        let face_half = resize(&face_full, 256, 256)?;

        // 6. body morpher (rotation).
        let bm = Self::run(&self.body_morpher, &[face_half, rotation_pose.clone()])?;
        let coarse_posed = resize(&bm[MORPHER_MERGED], 512, 512)?;
        let coarse_grid = resize(&bm[MORPHER_GRID_CHANGE], 512, 512)?;

        // 7. upscaler -> final posed image.
        let up = Self::run(
            &self.upscaler,
            &[face_full.clone(), coarse_posed, coarse_grid, rotation_pose],
        )?;
        Ok(up[MORPHER_MERGED].clone())
    }
}

/// Crop a square `size`x`size` region at (row, col) from `(1, C, H, W)`.
fn crop(x: &Tensor, row: usize, col: usize, size: usize) -> anyhow::Result<Tensor> {
    Ok(x.narrow(2, row, size)?.narrow(3, col, size)?.contiguous()?)
}

/// Paste `patch` (1, C, ph, pw) into `base` (1, C, H, W) at (row, col).
fn place(base: &Tensor, patch: &Tensor, row: usize, col: usize) -> anyhow::Result<Tensor> {
    let (_, _, _ph, pw) = patch.dims4()?;
    let base = base.contiguous()?;
    let band = base.narrow(3, col, pw)?.contiguous()?; // (1, C, H, pw)
    let band = band.slice_scatter(&patch.contiguous()?, 2, row)?;
    Ok(base.slice_scatter(&band, 3, col)?)
}

/// Bilinear resize (align_corners=false) via grid_sample, matching THA4's
/// `interpolate(mode='bilinear', align_corners=False)`.
fn resize(x: &Tensor, h: usize, w: usize) -> anyhow::Result<Tensor> {
    let (n, _, _, _) = x.dims4()?;
    let grid = identity_grid(n, h, w, x.device())?;
    Ok(grid_sample_bilinear_border(x, &grid, false)?)
}
