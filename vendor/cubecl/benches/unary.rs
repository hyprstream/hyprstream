use cubecl::{calculate_cube_count_elemwise, frontend, prelude::*};
use cubecl_runtime::TimestampsResult;
use std::marker::PhantomData;

#[cfg(feature = "cuda")]
use half::f16;

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::future;
use cubecl_linalg::tensor::TensorHandle;

#[cube(launch)]
fn execute<F: Float>(lhs: &Tensor<F>, rhs: &Tensor<F>, out: &mut Tensor<F>) {
    if ABSOLUTE_POS < out.len() {
        for i in 0..256u32 {
            if i % 2 == 0 {
                out[ABSOLUTE_POS] -= F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            } else {
                out[ABSOLUTE_POS] += F::cos(lhs[ABSOLUTE_POS] * rhs[ABSOLUTE_POS]);
            }
        }
    }
}

impl<R: Runtime, E: Float> Benchmark for UnaryBench<R, E> {
    type Args = (TensorHandle<R, E>, TensorHandle<R, E>, TensorHandle<R, E>);

    fn prepare(&self) -> Self::Args {
        let client = R::client(&self.device);
        let lhs = TensorHandle::zeros(&client, self.shape.clone());
        let rhs = TensorHandle::zeros(&client, self.shape.clone());
        let out = TensorHandle::zeros(&client, self.shape.clone());

        (lhs, rhs, out)
    }

    fn execute(&self, (lhs, rhs, out): Self::Args) {
        let num_elems: usize = out.shape.iter().product();

        let cube_dim = CubeDim::new(16, 16, 1);
        let cube_count =
            calculate_cube_count_elemwise(num_elems / self.vectorization as usize, cube_dim);

        execute::launch::<E, R>(
            &self.client,
            cube_count,
            cube_dim,
            lhs.as_arg(self.vectorization),
            rhs.as_arg(self.vectorization),
            out.as_arg(self.vectorization),
        )
    }

    fn num_samples(&self) -> usize {
        100
    }

    fn name(&self) -> String {
        format!(
            "unary-{}-{}-{:?}",
            R::name(),
            E::as_elem(),
            self.vectorization
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn sync_elapsed(&self) -> TimestampsResult {
        future::block_on(self.client.sync_elapsed())
    }
}

#[allow(dead_code)]
struct UnaryBench<R: Runtime, E> {
    shape: Vec<usize>,
    vectorization: u8,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _e: PhantomData<E>,
}

#[allow(dead_code)]
#[derive(Debug)]
enum MatmulKind {
    Tiling2d,
    Cmma,
}

#[allow(dead_code)]
fn run<R: Runtime, E: frontend::Float>(device: R::Device, vectorization: u8) {
    let client = R::client(&device);
    client.enable_timestamps();

    let bench = UnaryBench::<R, E> {
        shape: vec![32, 512, 2048],
        vectorization,
        client,
        device,
        _e: PhantomData,
    };
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::DeviceOnly));
}

fn main() {
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, f16>(Default::default(), 8);
    #[cfg(feature = "cuda")]
    run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), 4);
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), 1);
    #[cfg(feature = "wgpu")]
    run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), 4);
}
