//! Image warping: `affine_grid` (identity) + `grid_sample`.
//!
//! THA4 warps images with exactly one configuration, used in the face morpher,
//! body morpher, and SIREN modules:
//!
//! ```text
//! grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
//! ```
//!
//! where `grid = affine_grid(identity, ...) + grid_change`. candle has no
//! `grid_sample`, so we implement it here from primitive tensor ops. Semantics
//! match PyTorch's `torch.nn.functional.grid_sample`: for `padding_mode='border'`
//! the source *coordinate* is clamped to `[0, size-1]` before interpolation.

use candle_core::{DType, Device, Result, Tensor};

/// Identity sampling grid, shape `(n, h, w, 2)`, last dim ordered `(x, y)`.
///
/// Matches `torch.nn.functional.affine_grid(identity_2x3, [n,c,h,w],
/// align_corners=False)`: pixel `j` maps to normalized coord `(2j+1)/w - 1`.
pub fn identity_grid(n: usize, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    let xs = Tensor::arange(0f32, w as f32, device)?.affine(2.0 / w as f64, 1.0 / w as f64 - 1.0)?;
    let ys = Tensor::arange(0f32, h as f32, device)?.affine(2.0 / h as f64, 1.0 / h as f64 - 1.0)?;
    let x_grid = xs.reshape((1, 1, w, 1))?.broadcast_as((n, h, w, 1))?;
    let y_grid = ys.reshape((1, h, 1, 1))?.broadcast_as((n, h, w, 1))?;
    Tensor::cat(&[&x_grid, &y_grid], 3)?.contiguous()
}

/// Bilinear `grid_sample` with `padding_mode='border'`.
///
/// * `image`: `(n, c, h, w)`
/// * `grid`:  `(n, ho, wo, 2)`, values in `[-1, 1]`, last dim `(x, y)`
/// * returns: `(n, c, ho, wo)`
pub fn grid_sample_bilinear_border(image: &Tensor, grid: &Tensor, align_corners: bool) -> Result<Tensor> {
    let (n, c, h, w) = image.dims4()?;
    let (gn, ho, wo, two) = grid.dims4()?;
    if gn != n || two != 2 {
        candle_core::bail!("grid_sample: image {:?} incompatible with grid {:?}", image.shape(), grid.shape());
    }

    // Split grid into x/y coordinate planes: (n, ho, wo).
    let gx = grid.narrow(3, 0, 1)?.squeeze(3)?;
    let gy = grid.narrow(3, 1, 1)?.squeeze(3)?;

    // Un-normalize [-1,1] -> pixel coordinates.
    let (ix_raw, iy_raw) = if align_corners {
        // ix = (gx+1)/2 * (w-1)
        (gx.affine(0.5 * (w as f64 - 1.0), 0.5 * (w as f64 - 1.0))?,
         gy.affine(0.5 * (h as f64 - 1.0), 0.5 * (h as f64 - 1.0))?)
    } else {
        // ix = ((gx+1)*w - 1)/2 = gx*(w/2) + (w-1)/2
        (gx.affine(0.5 * w as f64, 0.5 * (w as f64 - 1.0))?,
         gy.affine(0.5 * h as f64, 0.5 * (h as f64 - 1.0))?)
    };

    // Border padding: clamp the coordinate itself (PyTorch semantics), then interpolate.
    let ix = ix_raw.clamp(0f32, (w - 1) as f32)?;
    let iy = iy_raw.clamp(0f32, (h - 1) as f32)?;

    let x0 = ix.floor()?;
    let y0 = iy.floor()?;
    let wx1 = ix.sub(&x0)?; // fractional part -> weight toward x1
    let wy1 = iy.sub(&y0)?;
    let wx0 = wx1.affine(-1.0, 1.0)?; // 1 - wx1
    let wy0 = wy1.affine(-1.0, 1.0)?;
    let x1 = x0.affine(1.0, 1.0)?;
    let y1 = y0.affine(1.0, 1.0)?;

    // Clamp neighbor indices to the border for gathering.
    let x0c = x0.clamp(0f32, (w - 1) as f32)?;
    let x1c = x1.clamp(0f32, (w - 1) as f32)?;
    let y0c = y0.clamp(0f32, (h - 1) as f32)?;
    let y1c = y1.clamp(0f32, (h - 1) as f32)?;

    // Linear index into the flattened H*W plane, as u32.
    let lin = |yy: &Tensor, xx: &Tensor| -> Result<Tensor> {
        yy.affine(w as f64, 0.0)?.add(xx)?.to_dtype(DType::U32)
    };
    let i00 = lin(&y0c, &x0c)?;
    let i01 = lin(&y0c, &x1c)?;
    let i10 = lin(&y1c, &x0c)?;
    let i11 = lin(&y1c, &x1c)?;

    // Gather the four corners for every channel.
    let img_flat = image.contiguous()?.reshape((n, c, h * w))?;
    let gather_corner = |idx: &Tensor| -> Result<Tensor> {
        let idx = idx
            .reshape((n, 1, ho * wo))?
            .broadcast_as((n, c, ho * wo))?
            .contiguous()?;
        img_flat.gather(&idx, 2)
    };
    let v00 = gather_corner(&i00)?;
    let v01 = gather_corner(&i01)?;
    let v10 = gather_corner(&i10)?;
    let v11 = gather_corner(&i11)?;

    // Bilinear weights per corner, shaped (n, 1, ho*wo) to broadcast over channels.
    let wt = |a: &Tensor, b: &Tensor| -> Result<Tensor> { a.mul(b)?.reshape((n, 1, ho * wo)) };
    let w00 = wt(&wy0, &wx0)?;
    let w01 = wt(&wy0, &wx1)?;
    let w10 = wt(&wy1, &wx0)?;
    let w11 = wt(&wy1, &wx1)?;

    let out = v00
        .broadcast_mul(&w00)?
        .add(&v01.broadcast_mul(&w01)?)?
        .add(&v10.broadcast_mul(&w10)?)?
        .add(&v11.broadcast_mul(&w11)?)?;
    out.reshape((n, c, ho, wo))
}

/// THA4's `apply_grid_change`: warp `image` by `identity_grid + grid_change`.
///
/// * `image`:       `(n, c, h, w)`
/// * `grid_change`: `(n, h, w, 2)`
pub fn apply_grid_change(image: &Tensor, grid_change: &Tensor) -> Result<Tensor> {
    let (n, _c, h, w) = image.dims4()?;
    let base = identity_grid(n, h, w, image.device())?;
    let grid = base.add(grid_change)?;
    grid_sample_bilinear_border(image, &grid, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_image(n: usize, c: usize, h: usize, w: usize, device: &Device) -> Result<Tensor> {
        // value = channel*1000 + row*w + col, so every pixel is distinct.
        let hw = (h * w) as f32;
        let base = Tensor::arange(0f32, hw, device)?.reshape((1, 1, h, w))?;
        let mut planes = Vec::new();
        for ch in 0..c {
            planes.push(base.affine(1.0, (ch as f64) * 1000.0)?);
        }
        let img = Tensor::cat(&planes.iter().collect::<Vec<_>>(), 1)?; // (1,c,h,w)
        img.broadcast_as((n, c, h, w))?.contiguous()
    }

    #[test]
    fn identity_grid_resamples_to_identity() -> Result<()> {
        let device = crate::best_device()?;
        let img = ramp_image(2, 3, 4, 5, &device)?;
        // grid_change = 0 -> sampling at identity grid -> output == input.
        let grid_change = Tensor::zeros((2, 4, 5, 2), DType::F32, &device)?;
        let out = apply_grid_change(&img, &grid_change)?;
        let a = img.flatten_all()?.to_vec1::<f32>()?;
        let b = out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(a, b, "identity warp must reproduce the input exactly");
        Ok(())
    }

    #[test]
    fn border_padding_clamps_out_of_range() -> Result<()> {
        let device = crate::best_device()?;
        // 1x1x2x2 image: [[10, 20],[30, 40]]
        let img = Tensor::new(&[[[[10f32, 20.0], [30.0, 40.0]]]], &device)?;
        // Sample far outside on the negative side (x=y=-2) -> clamps to top-left = 10.
        // And far outside positive (x=y=+2) -> clamps to bottom-right = 40.
        let grid = Tensor::new(
            &[[[[-2f32, -2.0], [2.0, 2.0]], [[-2.0, 2.0], [2.0, -2.0]]]],
            &device,
        )?; // (1,2,2,2)
        let out = grid_sample_bilinear_border(&img, &grid, false)?;
        // (1,1,2,2) flattened row-major: [(-2,-2),(2,2),(-2,2),(2,-2)]
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(v[0], 10.0); // (-2,-2) -> top-left
        assert_eq!(v[1], 40.0); // ( 2, 2) -> bottom-right
        assert_eq!(v[2], 30.0); // (-2, 2) -> bottom-left
        assert_eq!(v[3], 20.0); // ( 2,-2) -> top-right
        Ok(())
    }

    #[test]
    fn bilinear_midpoint_averages_four_neighbors() -> Result<()> {
        let device = crate::best_device()?;
        // 1x1x2x2 image: [[0,10],[20,40]]. Sample the exact center of the 4 pixels.
        let img = Tensor::new(&[[[[0f32, 10.0], [20.0, 40.0]]]], &device)?;
        // Center in normalized coords (align_corners=false) for a 2px axis is 0.0.
        let grid = Tensor::new(&[[[[0f32, 0.0]]]], &device)?; // (1,1,1,2)
        let out = grid_sample_bilinear_border(&img, &grid, false)?;
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        // Mean of 0,10,20,40 = 17.5
        assert!((v[0] - 17.5).abs() < 1e-4, "got {}", v[0]);
        Ok(())
    }
}
