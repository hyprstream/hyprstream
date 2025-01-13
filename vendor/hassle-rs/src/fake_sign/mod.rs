mod modified_md5;
use modified_md5::Context;

#[repr(C)]
struct FileHeader {
    fourcc: u32,
    hash_value: [u32; 4],
    container_version: u32,
    file_length: u32,
    num_chunks: u32,
}

const DXIL_HEADER_CONTAINER_VERSION_OFFSET: usize = 20;
const DXBC_FOURCC: u32 = u32::from_le_bytes([b'D', b'X', b'B', b'C']);

fn read_fourcc(dxil: &[u8]) -> u32 {
    let header: *const FileHeader = dxil.as_ptr().cast();
    unsafe { (*header).fourcc }
}

fn read_file_length(dxil: &[u8]) -> u32 {
    let header: *const FileHeader = dxil.as_ptr().cast();
    unsafe { (*header).file_length }
}

fn write_hash_value(dxil: &mut [u8], state: [u32; 4]) {
    let header: *mut FileHeader = dxil.as_mut_ptr().cast();

    unsafe {
        (*header).hash_value.copy_from_slice(&state);
    }
}

/// Helper function for signing DXIL binary blobs when
/// `dxil.dll` might not be available (such as on Linux based
/// platforms).
/// This essentially performs the same functionality as [`crate::validate_dxil()`]
/// but in a more cross platform way.
///
/// Ported from <https://github.com/baldurk/renderdoc/blob/v1.x/renderdoc/driver/shaders/dxbc/dxbc_container.cpp#L832>
pub fn fake_sign_dxil_in_place(dxil: &mut [u8]) -> bool {
    if read_fourcc(dxil) != DXBC_FOURCC {
        return false;
    }

    if read_file_length(dxil) != dxil.len() as u32 {
        return false;
    }

    // the hashable data starts immediately after the hash.
    let data = &dxil[DXIL_HEADER_CONTAINER_VERSION_OFFSET..];

    let num_bits: u32 = data.len() as u32 * 8;
    let num_bits_part_2: u32 = (num_bits >> 2) | 1;
    let left_over_len: u32 = data.len() as u32 % 64;

    let (first_part, padding_part) = data.split_at(data.len() - left_over_len as usize);

    let mut ctx = Context::new();
    ctx.consume(first_part);

    let mut block = [0u8; 64];

    if left_over_len >= 56 {
        assert_eq!(padding_part.len(), left_over_len as usize);
        ctx.consume(padding_part);

        block[0..4].copy_from_slice(&0x80u32.to_le_bytes());
        ctx.consume(&block[0..64 - left_over_len as usize]);

        // the final block contains the number of bits in the first dword, and the weird upper bits
        block[0..4].copy_from_slice(&num_bits.to_le_bytes());

        // write to last dword
        block[15 * 4..].copy_from_slice(&num_bits_part_2.to_le_bytes());

        ctx.consume(block);
    } else {
        ctx.consume(num_bits.to_le_bytes());

        if left_over_len != 0 {
            ctx.consume(padding_part)
        }

        let padding_bytes = (64 - left_over_len - 4) as usize;

        block[0] = 0x80;
        block[padding_bytes - 4..padding_bytes].copy_from_slice(&num_bits_part_2.to_le_bytes());
        ctx.consume(&block[0..padding_bytes]);
    }

    // dxil signing is odd - it doesn't run the finalization step of the md5
    // algorithm but instead pokes the hasher state directly into container

    write_hash_value(dxil, ctx.state);

    true
}
