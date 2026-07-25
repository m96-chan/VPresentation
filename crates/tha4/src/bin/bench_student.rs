//! Benchmark + smoke-test the student (real-time) poser.
//! Usage: cargo run --release -p tha4 --bin bench-student -- [char_dir] [--cpu]

use tha4::poser::{StudentPoser, NUM_POSE_PARAMS};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let char_dir = args.next().unwrap_or_else(|| "data/character_models/lambda_00".to_string());
    let use_cpu = args.any(|a| a == "--cpu");
    let device = if use_cpu { candle_core::Device::Cpu } else { tha4::best_device()? };
    println!("[bench] device={}", tha4::device_label(&device));

    let image = tha4::image_io::load_thaa_image(&format!("{char_dir}/character.png"), 512, &device)?;
    let poser = StudentPoser::load(&char_dir, device)?;

    let mut pose = [0f32; NUM_POSE_PARAMS];
    pose[39] = 0.4; // head_x
    pose[40] = -0.3; // head_y
    pose[12] = 1.0; // wink

    // warm up
    let out = poser.pose(&image, &pose)?;
    let _ = out.sum_all()?.to_scalar::<f32>()?;
    tha4::image_io::save_thaa_image(&out, "out/student_posed.png")?;

    let n = 20;
    let t0 = std::time::Instant::now();
    for i in 0..n {
        pose[40] = -0.3 + 0.01 * i as f32;
        let out = poser.pose(&image, &pose)?;
        let _ = out.sum_all()?.to_scalar::<f32>()?; // force sync
    }
    let per = t0.elapsed().as_secs_f64() / n as f64;
    println!("[bench] {:.1} ms/frame  ({:.1} fps)  over {n} frames", per * 1000.0, 1.0 / per);
    println!("[bench] wrote out/student_posed.png");
    Ok(())
}
