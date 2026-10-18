//! Harness-baseline bench (feeds P3.7): what the harness itself costs per
//! item, excluding the subject — run, observation extraction, protocol
//! scoring. The subject here is the in-process [`HashSubject`], so measured
//! time is harness overhead plus hash-draw, i.e. the floor any real arm adds
//! to. Run: `cargo bench -p hyprstream-eval --bench harness_baseline`.

#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use hyprstream_bench::{generate_all, BenchConfig};
use hyprstream_eval::score::ScoreConfig;
use hyprstream_eval::{score_bench_run, EvalItem, HashSubject, Harness};

fn items(seeds: u32) -> Vec<EvalItem> {
    generate_all(&BenchConfig {
        seeds_per_stratum: seeds,
        seed_base: 0x06,
    })
    .iter()
    .map(EvalItem::from)
    .collect()
}

fn bench_run_and_score(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("harness_baseline");
    for seeds in [1, 4] {
        let eval_items = items(seeds);
        let n = eval_items.len();
        group.bench_with_input(
            BenchmarkId::new("run_items", format!("{n} items")),
            &eval_items,
            |b, eval_items| {
                b.iter(|| {
                    runtime.block_on(async {
                        black_box(
                            Harness
                                .run_items(eval_items, &HashSubject::new("hash-bench-1"))
                                .await,
                        )
                    })
                });
            },
        );
        let output = runtime
            .block_on(Harness.run_items(&eval_items, &HashSubject::new("hash-bench-1")))
            .unwrap();
        group.bench_with_input(
            BenchmarkId::new("score_run", format!("{n} items")),
            &output,
            |b, output| {
                b.iter(|| {
                    black_box(score_bench_run(output, &eval_items, &ScoreConfig::default()))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_run_and_score);
criterion_main!(benches);
