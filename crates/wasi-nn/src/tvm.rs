//! Implements the wasi-nn API for TVM Runtime.
use crate::api::{Backend, BackendError, BackendExecutionContext, BackendGraph};
use crate::witx::types::{ExecutionTarget, GraphBuilderArray, Tensor, TensorType};
use tvm::errors::{Error, NDArrayError};
use tvm::module::Module;
use tvm::runtime::graph_rt::GraphRt;
use tvm::runtime::*;

use std::{path::Path, str};

pub(crate) struct TvmBackend(Module);

impl Default for TvmBackend {
    fn default() -> Self {
        // Load tvm's runtime lib from a ENV PATH.
        // TODO Consider ENV Name.
        let module = Module::load(&Path::new(env!("TVM_LIBRARY_PATH"))).unwrap();
        Self(module)
    }
}

impl Backend for TvmBackend {
    fn name(&self) -> &str {
        "tvm"
    }

    fn load(
        &mut self,
        builders: &GraphBuilderArray<'_>,
        target: ExecutionTarget,
    ) -> Result<Box<dyn BackendGraph>, BackendError> {
        if builders.len() != 2 {
            return Err(BackendError::InvalidNumberOfBuilders(2, builders.len()).into());
        }

        // Read the guest array.
        // In tvm, GraphBuilderArray includes graph and weigts of tvm.
        let builders = builders.as_ptr();
        let graph = builders.read()?.as_slice()?;
        let weights = builders.add(1)?.read()?.as_slice()?.to_vec();

        // Convert device.
        let device = map_execution_target_to_string(target);

        Ok(Box::new(TvmGraph(
            self.0.clone(),
            graph.as_ref().to_vec(),
            device,
            weights.clone(),
        )))
    }
}

struct TvmGraph(Module, Vec<u8>, Device, Vec<u8>);

impl BackendGraph for TvmGraph {
    fn init_execution_context(&mut self) -> Result<Box<dyn BackendExecutionContext>, BackendError> {
        // Create a tvm runtime.
        // TODO Module clone is dupulicated..
        let mut graph_rt =
            GraphRt::create_from_parts(str::from_utf8(&self.1)?, self.0.clone(), self.2)?;
        graph_rt.load_params(&*self.3)?;

        Ok(Box::new(TvmExecutionContext(graph_rt, self.2)))
    }
}

struct TvmExecutionContext(GraphRt, Device);

impl BackendExecutionContext for TvmExecutionContext {
    fn set_input(&mut self, index: u32, tensor: &Tensor<'_>) -> Result<(), BackendError> {
        // TODO TVM don't need an index as args to get input_name.
        let input_name = "data";

        // Construct the blob structure.
        let dimensions = tensor
            .dimensions
            .as_slice()?
            .iter()
            .map(|d| *d as i64)
            .collect::<Vec<_>>();
        let precision = map_tensor_type_to_precision(tensor.type_);
        let mut nd = NDArray::empty(&dimensions[..], self.1, precision);

        let f32_data = bytes_vec_to_f32(tensor.data.as_slice()?.to_vec(), nd.len());
        nd.fill_from_iter(f32_data.iter().copied());

        println!(
            "input shape is {:?}, len: {:?}, size: {:?}",
            nd.shape(),
            nd.len(),
            nd.size(),
        );
        self.0.set_input(&input_name, nd)?;
        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        self.0.run()?;
        Ok(())
    }

    fn get_output(&mut self, index: u32, destination: &mut [u8]) -> Result<u32, BackendError> {
        // Prepare to get the output
        //
        // c.f. https://github.com/apache/tvm/blob/main/rust/tvm-rt/src/graph_rt.rs#L97
        let output_nd: NDArray = self.0.get_output(index.into())?;
        // flatten the output as Vec<f32>
        let output: Vec<f32> = output_nd.to_vec::<f32>()?;

        let out = f32_vec_to_bytes(output);
        let output_size = out.len();
        if output_size > destination.len() {
            return Err(BackendError::NotEnoughMemory(output_size));
        }
        destination[..output_size].copy_from_slice(&out);
        Ok(output_size as u32)
    }
}

pub fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let chunks: Vec<[u8; 4]> = data.into_iter().map(|f| f.to_le_bytes()).collect();
    let u8_arr: Vec<u8> = chunks.iter().flatten().copied().collect();
    u8_arr
}

pub fn bytes_vec_to_f32(tensor: Vec<u8>, length: usize) -> Vec<f32> {
    let mut f32_arr: Vec<f32> = vec![0.; length];
    for (i, t) in tensor.chunks(4).enumerate() {
        let mut u8_bytes: [u8; 4] = [0; 4];
        u8_bytes.copy_from_slice(t);
        f32_arr[i] = f32::from_ne_bytes(u8_bytes);
    }
    f32_arr
}

impl From<str::Utf8Error> for BackendError {
    fn from(e: str::Utf8Error) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

impl From<NDArrayError> for BackendError {
    fn from(e: NDArrayError) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

impl From<Error> for BackendError {
    fn from(e: Error) -> Self {
        BackendError::BackendAccess(anyhow::Error::new(e))
    }
}

/// Return the execution target string expected by TVM from the
/// `ExecutionTarget` enum provided by wasi-nn.
///
/// TODO: Need more support device type.
/// cf. https://github.com/apache/tvm/blob/main/rust/tvm-sys/src/device.rs#L66
fn map_execution_target_to_string(target: ExecutionTarget) -> Device {
    match target {
        ExecutionTarget::Cpu => Device::cpu(0),
        ExecutionTarget::Gpu => Device::cuda(0),
        ExecutionTarget::Tpu => unimplemented!("TVM does not support TPU execution targets"),
    }
}

/// Return TVM's precision type for the `TensorType` enum provided by
/// wasi-nn.
/// cf. https://github.com/apache/tvm/blob/main/rust/tvm-sys/src/datatype.rs
fn map_tensor_type_to_precision(tensor_type: TensorType) -> DataType {
    match tensor_type {
        TensorType::F16 => DataType::float(16, 1),
        TensorType::F32 => DataType::float(32, 1),
        TensorType::U8 => DataType::uint(8, 1),
        TensorType::I32 => DataType::int(32, 1),
    }
}
