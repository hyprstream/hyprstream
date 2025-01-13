use argminmax::ArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

use argminmax::dtype_strategy::FloatIgnoreNaN;
use argminmax::scalar::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::simd::{SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use argminmax::simd::{SIMDArgMinMax, NEON};

use half::f16;

// _in stands for "ignore nan"

fn argminmax_in_f16_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[f16] = &utils::SampleUniformFullRange::get_random_array(n);
    c.bench_function("scalar_f16_argminmax_in", |b| {
        b.iter(|| SCALAR::<FloatIgnoreNaN>::argminmax(black_box(data)))
    });
    c.bench_function("scalar_f16_argmin_in", |b| {
        b.iter(|| SCALAR::<FloatIgnoreNaN>::argmin(black_box(data)))
    });
    c.bench_function("scalar_f16_argmax_in", |b| {
        b.iter(|| SCALAR::<FloatIgnoreNaN>::argmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_f16_argminmax_in", |b| {
            b.iter(|| unsafe { SSE::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_f16_argmin_in", |b| {
            b.iter(|| unsafe { SSE::<FloatIgnoreNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_f16_argmax_in", |b| {
            b.iter(|| unsafe { SSE::<FloatIgnoreNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f16_argminmax_in", |b| {
            b.iter(|| unsafe { AVX2::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f16_argmin_in", |b| {
            b.iter(|| unsafe { AVX2::<FloatIgnoreNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f16_argmax_in", |b| {
            b.iter(|| unsafe { AVX2::<FloatIgnoreNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_f16_argminmax_in", |b| {
            b.iter(|| unsafe { AVX512::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_f16_argmin_in", |b| {
            b.iter(|| unsafe { AVX512::<FloatIgnoreNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_f16_argmax_in", |b| {
            b.iter(|| unsafe { AVX512::<FloatIgnoreNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_f16_argminmax_in", |b| {
            b.iter(|| unsafe { NEON::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_f16_argmin_in", |b| {
            b.iter(|| unsafe { NEON::<FloatIgnoreNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_f16_argmax_in", |b| {
            b.iter(|| unsafe { NEON::<FloatIgnoreNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f16_argminmax_in", |b| {
            b.iter(|| unsafe { NEON::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f16_argmin_in", |b| {
            b.iter(|| unsafe { NEON::<FloatIgnoreNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f16_argmax_in", |b| {
            b.iter(|| unsafe { NEON::<FloatIgnoreNaN>::argmax(black_box(data)) })
        });
    }
    c.bench_function("impl_f16_argminmax_in", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
    c.bench_function("impl_f16_argmin_in", |b| {
        b.iter(|| black_box(data.argmin()))
    });
    c.bench_function("impl_f16_argmax_in", |b| {
        b.iter(|| black_box(data.argmax()))
    });
}

criterion_group!(benches, argminmax_in_f16_random_array_long,);
criterion_main!(benches);
