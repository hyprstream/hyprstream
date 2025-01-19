#![feature(test)]
extern crate test;

use test::{Bencher, black_box};

macro_rules! benches {
    (
        $(
            $(#[$attr:meta])*
            $name:ident($value:expr)
        ),*
    ) => {
        mod bench_itoap_write_to_vec {
            use test::{Bencher, black_box};
            $(
                $(#[$attr])*
                #[bench]
                pub fn $name(b: &mut Bencher) {
                    let mut buf = Vec::with_capacity(40);

                    b.iter(|| {
                        buf.clear();
                        itoap::write_to_vec(&mut buf, black_box($value));
                        black_box(&mut buf);
                    });
                }
            )*
        }

        mod bench_itoap_write_to_ptr {
            use test::{Bencher, black_box};
            $(
                $(#[$attr])*
                #[bench]
                pub fn $name(b: &mut Bencher) {
                    let mut buf = Vec::<u8>::with_capacity(40);

                    b.iter(|| unsafe {
                        itoap::write_to_ptr(buf.as_mut_ptr(), black_box($value));
                        black_box(&mut buf);
                    });
                }
            )*
        }

        mod bench_itoa_write {
            use test::{Bencher, black_box};
            $(
                $(#[$attr])*
                #[bench]
                pub fn $name(b: &mut Bencher) {
                    let mut buf = Vec::with_capacity(40);

                    b.iter(|| {
                        buf.clear();
                        let _ = itoa::write(&mut buf, black_box($value));
                        black_box(&mut buf);
                    });
                }
            )*
        }

        mod bench_std_fmt {
            use test::{Bencher, black_box};
            $(
                $(#[$attr])*
                #[bench]
                pub fn $name(b: &mut Bencher) {
                    use std::io::Write;

                    let mut buf = Vec::with_capacity(40);

                    b.iter(|| {
                        buf.clear();
                        let _ = write!(&mut buf, "{}", black_box($value));
                        black_box(&mut buf);
                    });
                }
            )*
        }
    }
}

benches! {
    bench_u64_0(0u64),
    bench_u64_half(<u32>::max_value() as u64),
    bench_u64_max(<u64>::max_value()),
    bench_i16_0(0i16),
    bench_i16_min(<i16>::min_value()),
    bench_u128_0(0u128),
    bench_u128_max(<u128>::max_value())
}

#[bench]
fn noop(b: &mut Bencher) {
    let mut buf = Vec::<u8>::with_capacity(40);
    b.iter(|| {
        buf.clear();
        black_box(0i16);
        black_box(&mut buf);
    })
}
