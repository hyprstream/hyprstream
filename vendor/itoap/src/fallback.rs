use core::ptr;

use crate::common::{divmod, lookup, write4, write4_pad, write8_pad};

pub unsafe fn write_u32(n: u32, buf: *mut u8) -> usize {
    if n < 10000 {
        write4(n, buf)
    } else if n < 100_000_000 {
        let (n1, n2) = divmod(n, 10000);

        let l = write4(n1, buf);
        write4_pad(n2, buf.add(l));
        l + 4
    } else {
        let (n1, n2) = divmod(n, 100_000_000);

        let l = if n1 >= 10 {
            ptr::copy_nonoverlapping(lookup(n1), buf, 2);
            2
        } else {
            *buf = n1 as u8 + 0x30;
            1
        };

        write8_pad(n2, buf.add(l));
        l + 8
    }
}

pub unsafe fn write_u64(n: u64, buf: *mut u8) -> usize {
    if n < 10000 {
        write4(n as u32, buf)
    } else if n < 100_000_000 {
        let (n1, n2) = divmod(n, 10000);

        let l = write4(n1 as u32, buf);
        write4_pad(n2 as u32, buf.add(l));
        l + 4
    } else if n < 10_000_000_000_000_000 {
        let (n1, n2) = divmod(n, 100_000_000);
        let (n1, n2) = (n1 as u32, n2 as u32);

        let l = if n1 < 10000 {
            write4(n1, buf)
        } else {
            let (n11, n12) = divmod(n1, 10000);
            let l = write4(n11, buf);
            write4_pad(n12, buf.add(l));
            l + 4
        };

        write8_pad(n2, buf.add(l));
        l + 8
    } else {
        let (n1, n2) = divmod(n, 10_000_000_000_000_000);
        let (n21, n22) = divmod(n2, 100_000_000);

        let l = write4(n1 as u32, buf);
        write8_pad(n21 as u32, buf.add(l));
        write8_pad(n22 as u32, buf.add(l + 8));
        l + 16
    }
}
