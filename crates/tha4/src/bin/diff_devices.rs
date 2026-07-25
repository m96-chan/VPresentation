//! Run an ONNX graph on CPU and Metal, then report — in topological order —
//! the first node whose output diverges. Pinpoints a Metal-specific op bug.
//!
//! Usage: cargo run --release -p tha4 --bin diff-devices -- upscaler_debug upscaler

use std::collections::HashMap;

use candle_core::{Device, Tensor};

fn run(model: &candle_onnx::onnx::ModelProto, refs: &HashMap<String, Tensor>, device: &Device)
    -> anyhow::Result<HashMap<String, Tensor>>
{
    let graph = model.graph.as_ref().unwrap();
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    for (i, gi) in graph.input.iter().enumerate() {
        let t = refs.get(&format!("in{i}")).unwrap().to_device(device)?;
        inputs.insert(gi.name.clone(), t);
    }
    Ok(candle_onnx::simple_eval(model, inputs)?)
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let onnx_name = args.next().unwrap();
    let ref_name = args.next().unwrap();

    let model = candle_onnx::read_file(format!("data/tha4/onnx/{onnx_name}.onnx"))?;
    let refs = candle_core::safetensors::load(format!("data/tha4/reference/{ref_name}.safetensors"), &Device::Cpu)?;

    eprintln!("running CPU...");
    let cpu = run(&model, &refs, &Device::Cpu)?;
    eprintln!("running Metal...");
    let metal = run(&model, &refs, &Device::new_metal(0)?)?;

    let graph = model.graph.as_ref().unwrap();
    let mut first_bad: Option<usize> = None;
    for node in graph.node.iter() {
        for out in node.output.iter() {
            if out.is_empty() { continue; }
            let (Some(c), Some(m)) = (cpu.get(out), metal.get(out)) else { continue };
            let cv = match c.to_dtype(candle_core::DType::F32).and_then(|t| t.flatten_all()?.to_vec1::<f32>()) {
                Ok(v) => v, Err(_) => continue,
            };
            let mv = match m.to_dtype(candle_core::DType::F32).and_then(|t| t.flatten_all()?.to_vec1::<f32>()) {
                Ok(v) => v, Err(_) => continue,
            };
            if cv.len() != mv.len() { continue; }
            let max = cv.iter().zip(&mv).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            if max > 1e-2 {
                let n = first_bad.get_or_insert(0);
                if max > 1e-2 && *n < 60 {
                    println!("{:3} op={:<20} max_abs={:.3e} shape={:?} out={}",
                        *n, node.op_type, max, c.dims(), out);
                    *n += 1;
                }
            }
        }
    }
    Ok(())
}
