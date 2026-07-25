//! Pose a single THA4 image with a fixed pose and save the result.
//!
//! Usage: cargo run --release -p tha4 --bin pose-image -- <input.png> <output.png> [--cpu]

use tha4::poser::{Poser, NUM_POSE_PARAMS};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| "data/images/lambda_00.png".to_string());
    let output = args.next().unwrap_or_else(|| "out/posed.png".to_string());
    let use_cpu = args.any(|a| a == "--cpu");

    let device = if use_cpu { candle_core::Device::Cpu } else { tha4::best_device()? };
    println!("[pose] device={}", tha4::device_label(&device));

    let image = tha4::image_io::load_thaa_image(&input, 512, &device)?;
    let poser = Poser::load("data/tha4/onnx", device)?;

    // A visible pose: turn head, wink left eye, open mouth a bit, slight breathing.
    let mut pose = [0f32; NUM_POSE_PARAMS];
    pose[12] = 1.0; // eye_wink left (first eye param)
    pose[26] = 0.5; // mouth_aaa
    pose[39] = 0.4; // head_x
    pose[40] = -0.3; // head_y
    pose[41] = 0.2; // neck_z
    pose[44] = 0.6; // breathing

    let mut posed = poser.pose(&image, &pose)?;
    let _ = posed.sum_all()?.to_scalar::<f32>()?; // warm up (shader compile)
    for i in 0..3 {
        let t0 = std::time::Instant::now();
        posed = poser.pose(&image, &pose)?;
        let _ = posed.sum_all()?.to_scalar::<f32>()?;
        println!("[pose] warm run {i}: {:?}", t0.elapsed());
    }

    if let Some(parent) = std::path::Path::new(&output).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    tha4::image_io::save_thaa_image(&posed, &output)?;
    println!("[pose] wrote {output}");
    Ok(())
}
