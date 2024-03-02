use std::sync::Arc;

use ort::{GraphOptimizationLevel, Session};
use pacosako::{self, DenseBoard, PacoBoard};

fn main() -> Result<(), ort::Error> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        //.with_intra_threads(4)?
        //.with_model_from_memory(model_bytes)? // Must use this for WASM.
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

    let evaluation = evaluate_model(&board, &model)?;

    println!("Model Evaluation: {:?}", evaluation);

    Ok(())
}

/// Runs the given model on the given board and returns the evaluation.
/// Encapsulates all the ONNX/ORT specific data wrangling.
/// This uses the tensor representation of the board.
fn evaluate_model(board: &DenseBoard, model: &Session) -> Result<ModelEvaluation, ort::Error> {
    let input_repr: &mut [f32; 8 * 8 * 30] = &mut [0.; 8 * 8 * 30];
    pacosako::ai::repr::tensor_representation(board, input_repr);

    let input_shape: Vec<i64> = vec![1, 30, 8, 8_i64];
    let input_data: Box<[f32]> = input_repr.to_vec().into_boxed_slice();

    let input = ort::Value::try_from((input_shape, Arc::new(input_data)))?;

    let outputs = model.run(ort::inputs![input]?)?;

    let (o_shape, o_data): (Vec<i64>, &[f32]) = outputs["OUTPUT"].extract_raw_tensor()?;

    assert_eq!(o_shape, vec![1, 133]);

    let actions = board.actions().expect("Legal actions can't be determined");
    let mut policy = Vec::with_capacity(actions.len() as usize);
    for action in actions {
        // action to action index already returns a one-based index.
        // This works great with the first entry being the value.
        let action_index = pacosako::ai::glue::action_to_action_index(action);
        let action_policy = o_data[action_index as usize];
        policy.push((action, action_policy));
    }
    assert_eq!(policy.len(), actions.len() as usize);

    let mut evaluation = ModelEvaluation {
        value: o_data[0],
        policy,
    };
    evaluation.normalize_policy();

    Ok(evaluation)
}

/// Given a DenseBoard, we want to turns this into a model evaluation.
/// This is a list of the relative policy for various legal actions.
/// This also returns the value, but the raw Hedwig value isn't very good.
/// The action values have been normalized to sum up to 1.
#[derive(Debug)]
pub struct ModelEvaluation {
    pub value: f32,
    pub policy: Vec<(pacosako::PacoAction, f32)>,
}

impl ModelEvaluation {
    fn normalize_policy(&mut self) {
        let sum: f32 = self.policy.iter().map(|(_, p)| p).sum();
        if sum == 0. {
            if !self.policy.is_empty() {
                let spread = 1. / self.policy.len() as f32;
                for (_, p) in &mut self.policy {
                    *p = spread;
                }
            }
            return;
        }
        for (_, p) in &mut self.policy {
            *p /= sum;
        }
    }
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
        TensorElementType::Bool => "bool",
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
