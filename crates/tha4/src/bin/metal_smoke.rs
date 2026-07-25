//! Phase-0 smoke test: prove the candle fork runs a real tensor op on the
//! Apple M3 GPU via Metal. Run: `cargo run -p tha4 --bin metal-smoke`.

use candle_core::Tensor;

fn main() -> anyhow::Result<()> {
    let device = tha4::best_device()?;
    println!("[tha4] selected device: {}", tha4::device_label(&device));

    // Non-trivial op so the backend actually does work: (1024x1024) matmul.
    let n = 1024usize;
    let a = Tensor::rand(0f32, 1f32, (n, n), &device)?;
    let b = Tensor::rand(0f32, 1f32, (n, n), &device)?;
    let c = a.matmul(&b)?;
    let sum = c.sum_all()?.to_scalar::<f32>()?;

    println!("[tha4] matmul {n}x{n} ok on {}: sum={sum:.1}", tha4::device_label(&device));
    println!("[tha4] Phase 0 foundation: candle + Metal is working.");
    Ok(())
}
