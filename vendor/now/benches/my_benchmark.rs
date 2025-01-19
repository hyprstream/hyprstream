use criterion::{criterion_group, criterion_main, Criterion};

use chrono::Utc;
use now::TimeZoneNow;

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("utc now", |b| {
        b.iter(|| {
            Utc::now();
        })
    });
    c.bench_function("utc beginning_of_minute", |b| {
        b.iter(|| {
            Utc.beginning_of_minute();
        })
    });
    c.bench_function("utc beginning_of_hour", |b| {
        b.iter(|| {
            Utc.beginning_of_hour();
        })
    });
    c.bench_function("utc beginning_of_day", |b| {
        b.iter(|| {
            Utc.beginning_of_day();
        })
    });
    c.bench_function("utc beginning_of_week", |b| {
        b.iter(|| {
            Utc.beginning_of_week();
        })
    });
    c.bench_function("utc beginning_of_month", |b| {
        b.iter(|| {
            Utc.beginning_of_month();
        })
    });
    c.bench_function("utc beginning_of_quarter", |b| {
        b.iter(|| {
            Utc.beginning_of_quarter();
        })
    });
    c.bench_function("utc beginning_of_year", |b| {
        b.iter(|| {
            Utc.beginning_of_year();
        })
    });
    c.bench_function("utc end_of_minute", |b| {
        b.iter(|| {
            Utc.end_of_minute();
        })
    });
    c.bench_function("utc end_of_hour", |b| {
        b.iter(|| {
            Utc.end_of_hour();
        })
    });
    c.bench_function("utc end_of_day", |b| {
        b.iter(|| {
            Utc.end_of_day();
        })
    });
    c.bench_function("utc end_of_week", |b| {
        b.iter(|| {
            Utc.end_of_week();
        })
    });
    c.bench_function("utc end_of_month", |b| {
        b.iter(|| {
            Utc.end_of_month();
        })
    });
    c.bench_function("utc end_of_quarter", |b| {
        b.iter(|| {
            Utc.end_of_quarter();
        })
    });
    c.bench_function("utc end_of_year", |b| {
        b.iter(|| {
            Utc.end_of_year();
        })
    });
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
