//! Persistent THA4 pose engine for the GUI: load models once, then render a
//! posed frame per request. Line protocol over stdin/stdout:
//!
//!   stdin  : "<out_path>;<p0>,<p1>,...,<p44>\n"   (45 pose floats)
//!   stdout : "OK <out_path> <elapsed_ms>\n"  or  "ERR <message>\n"
//!
//! Usage: cargo run --release -p tha4 --bin serve -- <character.png> [--cpu] [--no-upscale]

use std::io::{BufRead, Write};

use candle_core::Tensor;
use tha4::poser::{Poser, StudentPoser, NUM_POSE_PARAMS};

/// Either poser backend; both expose `pose(image, &[f32]) -> Tensor`.
enum Backend {
    Teacher(Poser),
    Student(StudentPoser),
}

impl Backend {
    fn pose(&self, image: &Tensor, pose: &[f32]) -> anyhow::Result<Tensor> {
        match self {
            Backend::Teacher(p) => p.pose(image, pose),
            Backend::Student(p) => p.pose(image, pose),
        }
    }
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let image_path = args.next().unwrap_or_else(|| "data/images/lambda_00.png".to_string());
    let rest: Vec<String> = args.collect();
    let use_cpu = rest.iter().any(|a| a == "--cpu");
    let student_dir = rest.iter().position(|a| a == "--student").and_then(|i| rest.get(i + 1)).cloned();

    let device = if use_cpu { candle_core::Device::Cpu } else { tha4::best_device()? };
    let image = tha4::image_io::load_thaa_image(&image_path, 512, &device)?;
    let poser = match &student_dir {
        Some(dir) => Backend::Student(StudentPoser::load(dir, device.clone())?),
        None => Backend::Teacher(Poser::load("data/tha4/onnx", device.clone())?),
    };

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    // Signal readiness (models loaded).
    writeln!(stdout, "READY {}", tha4::device_label(&device))?;
    stdout.flush()?;

    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line == "quit" {
            break;
        }
        match render_line(&poser, &image, line) {
            Ok((path, ms)) => writeln!(stdout, "OK {path} {ms}")?,
            Err(e) => writeln!(stdout, "ERR {e}")?,
        }
        stdout.flush()?;
    }
    Ok(())
}

fn render_line(poser: &Backend, image: &candle_core::Tensor, line: &str) -> anyhow::Result<(String, u128)> {
    let (path, nums) = line.split_once(';').ok_or_else(|| anyhow::anyhow!("bad request"))?;
    let pose: Vec<f32> = nums.split(',').map(|s| s.trim().parse::<f32>()).collect::<Result<_, _>>()?;
    anyhow::ensure!(pose.len() == NUM_POSE_PARAMS, "need {NUM_POSE_PARAMS} params, got {}", pose.len());
    let t0 = std::time::Instant::now();
    let posed = poser.pose(image, &pose)?;
    let ms = t0.elapsed().as_millis();
    tha4::image_io::save_thaa_image(&posed, path)?;
    Ok((path.to_string(), ms))
}
