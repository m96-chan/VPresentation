//! Validate a THA4 network exported to ONNX by running it in candle and
//! comparing against the PyTorch reference tensors (the oracle).
//!
//! Usage: cargo run -p tha4 --bin validate-onnx -- eyebrow_decomposer [--metal]

use std::collections::HashMap;

use candle_core::{Device, Tensor};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let name = args.next().unwrap_or_else(|| "eyebrow_decomposer".to_string());
    let use_metal = args.any(|a| a == "--metal");

    let device = if use_metal { tha4::best_device()? } else { Device::Cpu };
    println!("[validate] network={name} device={}", tha4::device_label(&device));

    let onnx_path = format!("data/tha4/onnx/{name}.onnx");
    let ref_path = format!("data/tha4/reference/{name}.safetensors");

    let model = candle_onnx::read_file(&onnx_path)?;
    let refs = candle_core::safetensors::load(&ref_path, &device)?;
    let expected = refs.get("out0").expect("reference must contain 'out0'").clone();

    // Feed every graph input from the reference tensors (in0, in1, ...), by position.
    let graph = model.graph.as_ref().expect("graph");
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    for (i, gi) in graph.input.iter().enumerate() {
        let key = format!("in{i}");
        let t = refs
            .get(&key)
            .unwrap_or_else(|| panic!("reference missing {key} for graph input {}", gi.name))
            .clone();
        inputs.insert(gi.name.clone(), t);
    }

    let outputs = candle_onnx::simple_eval(&model, inputs)?;
    // out0 corresponds to the first graph output.
    let out_name = &graph.output[0].name;
    let got = outputs.get(out_name).expect("missing graph output").clone();

    let got = got.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let exp = expected.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got.len(), exp.len(), "output length mismatch");

    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_sq += (d as f64) * (d as f64);
    }
    let rmse = (sum_sq / got.len() as f64).sqrt();
    println!("[validate] {name}: max_abs_diff={max_abs:.3e} rmse={rmse:.3e} (n={})", got.len());

    if max_abs < 2e-3 {
        println!("[validate] PASS ✅ candle ONNX output matches PyTorch reference.");
        Ok(())
    } else {
        anyhow::bail!("mismatch too large: max_abs_diff={max_abs:.3e}");
    }
}
