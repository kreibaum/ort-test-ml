use ort::{GraphOptimizationLevel, Session};
use pacosako::{self, DenseBoard};

fn main() -> Result<(), ort::Error> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .with_model_from_file("hedwig-0.8.onnx")?;

    // Print some information about the model

    println!("Inputs:");
    for (i, input) in model.inputs.iter().enumerate() {
        println!(
            "    {i} {}: {}",
            input.name,
            display_value_type(&input.input_type)
        );
    }
    println!("Outputs:");
    for (i, output) in model.outputs.iter().enumerate() {
        println!(
            "    {i} {}: {}",
            output.name,
            display_value_type(&output.output_type)
        );
    }

    let board = DenseBoard::new();

    let input_repr: &mut [f32; 8 * 8 * 30] = &mut [0.; 8 * 8 * 30];
    pacosako::ai::repr::tensor_representation(&board, input_repr);

    // convert out to an ndarray
    let out: ndarray::Array<_, _> =
        ndarray::Array::from_shape_vec((1, 30, 8, 8), input_repr.to_vec()).unwrap();

    let outputs = model.run(ort::inputs![out]?)?;

    let output: ort::Tensor<'_, f32> = outputs["OUTPUT"].extract_tensor::<f32>()?;

    let output = output.view();
    let output = output.as_slice().unwrap();

    let value = output[0];
    let policy = &output[1..=132];

    println!("Value: {}", value);
    println!("Policy: {:?}", policy);

    Ok(())
}

fn display_value_type(value: &ValueType) -> String {
    match value {
        ValueType::Tensor { ty, dimensions } => {
            format!(
                "Tensor<{}>({})",
                display_element_type(*ty),
                dimensions
                    .iter()
                    .map(|c| if *c == -1 {
                        "dyn".to_string()
                    } else {
                        c.to_string()
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        ValueType::Map { key, value } => format!(
            "Map<{}, {}>",
            display_element_type(*key),
            display_element_type(*value)
        ),
        ValueType::Sequence(inner) => format!("Sequence<{}>", display_value_type(inner)),
    }
}

use ort::{TensorElementType, ValueType};

fn display_element_type(t: TensorElementType) -> &'static str {
    match t {
        TensorElementType::Bfloat16 => "bf16",
        TensorElementType::Bool => "bool",
        TensorElementType::Float16 => "f16",
        TensorElementType::Float32 => "f32",
        TensorElementType::Float64 => "f64",
        TensorElementType::Int16 => "i16",
        TensorElementType::Int32 => "i32",
        TensorElementType::Int64 => "i64",
        TensorElementType::Int8 => "i8",
        TensorElementType::String => "str",
        TensorElementType::Uint16 => "u16",
        TensorElementType::Uint32 => "u32",
        TensorElementType::Uint64 => "u64",
        TensorElementType::Uint8 => "u8",
    }
}
