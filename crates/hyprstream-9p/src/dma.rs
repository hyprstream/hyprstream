//! DMA ring buffer transport for 9P messages.
//!
//! Implements `P9Transport` using SharedArrayBuffer + Atomics,
//! matching the protocol from wanix/vfs/dma/ring.go and our
//! tested DmaChannel implementation.
//!
//! Memory layout (shared with Go DmaRing):
//! ```text
//! [0x0000 - 0x0FFF]: Control region (4096 bytes)
//!   Int32 atomics at indices:
//!     [0]: Chan0Head     (write pointer for channel 0)
//!     [1]: Chan0Tail     (read pointer for channel 0)
//!     [2]: Chan1Head     (write pointer for channel 1)
//!     [3]: Chan1Tail     (read pointer for channel 1)
//!     [4]: Status        (closed flags)
//!     [5]: Chan0MsgCount (HWM counter)
//!     [6]: Chan1MsgCount
//!     [7]: Chan0Hwm      (high water mark threshold)
//!     [8]: Chan1Hwm
//!
//! [0x1000 - mid]:    Buffer0 (ring data for channel 0)
//! [mid - end]:       Buffer1 (ring data for channel 1)
//! ```
//!
//! Messages are length-prefixed: [4-byte LE length][payload]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::client::P9Transport;

// Control region indices (Int32Array offsets)
const IDX_CHAN0_HEAD: u32 = 0;
const IDX_CHAN0_TAIL: u32 = 1;
const IDX_CHAN1_HEAD: u32 = 2;
const IDX_CHAN1_TAIL: u32 = 3;
const IDX_STATUS: u32 = 4;
const IDX_CHAN0_MSG_COUNT: u32 = 5;
const IDX_CHAN1_MSG_COUNT: u32 = 6;

const CONTROL_SIZE: u32 = 4096;
const HEADER_SIZE: u32 = 4; // length prefix

/// DMA ring buffer endpoint for one direction.
///
/// Reads from one channel, writes to the other.
/// For the 9P client: we write T-messages to chan0, read R-messages from chan1.
#[cfg(target_arch = "wasm32")]
pub struct DmaTransport {
    control: js_sys::Int32Array,
    write_buf: js_sys::Uint8Array,
    read_buf: js_sys::Uint8Array,
    write_capacity: u32,
    read_capacity: u32,
    // Channel assignment: write to chan0, read from chan1
    write_head_idx: u32,
    write_tail_idx: u32,
    read_head_idx: u32,
    read_tail_idx: u32,
    write_msg_count_idx: u32,
    read_msg_count_idx: u32,
}

#[cfg(target_arch = "wasm32")]
impl DmaTransport {
    /// Create a DMA transport from a SharedArrayBuffer.
    ///
    /// `client_endpoint`: if true, writes to chan0 and reads from chan1.
    /// If false, reversed (for the server side).
    pub fn new(sab: &js_sys::SharedArrayBuffer, client_endpoint: bool) -> Self {
        let total_size = sab.byte_length();
        let data_size = total_size - CONTROL_SIZE;
        let half = data_size / 2;

        let control = js_sys::Int32Array::new_with_byte_offset_and_length(
            &JsValue::from(sab.clone()),
            0,
            CONTROL_SIZE / 4,
        );

        let buf0 = js_sys::Uint8Array::new_with_byte_offset_and_length(
            &JsValue::from(sab.clone()),
            CONTROL_SIZE,
            half,
        );

        let buf1 = js_sys::Uint8Array::new_with_byte_offset_and_length(
            &JsValue::from(sab.clone()),
            CONTROL_SIZE + half,
            half,
        );

        if client_endpoint {
            Self {
                control,
                write_buf: buf0,
                read_buf: buf1,
                write_capacity: half,
                read_capacity: half,
                write_head_idx: IDX_CHAN0_HEAD,
                write_tail_idx: IDX_CHAN0_TAIL,
                read_head_idx: IDX_CHAN1_HEAD,
                read_tail_idx: IDX_CHAN1_TAIL,
                write_msg_count_idx: IDX_CHAN0_MSG_COUNT,
                read_msg_count_idx: IDX_CHAN1_MSG_COUNT,
            }
        } else {
            Self {
                control,
                write_buf: buf1,
                read_buf: buf0,
                write_capacity: half,
                read_capacity: half,
                write_head_idx: IDX_CHAN1_HEAD,
                write_tail_idx: IDX_CHAN1_TAIL,
                read_head_idx: IDX_CHAN0_HEAD,
                read_tail_idx: IDX_CHAN0_TAIL,
                write_msg_count_idx: IDX_CHAN1_MSG_COUNT,
                read_msg_count_idx: IDX_CHAN0_MSG_COUNT,
            }
        }
    }

    fn atomic_load(&self, idx: u32) -> i32 {
        js_sys::Atomics::load(&self.control, idx).unwrap_or(0)
    }

    fn atomic_store(&self, idx: u32, val: i32) {
        let _ = js_sys::Atomics::store(&self.control, idx, val);
    }

    fn atomic_add(&self, idx: u32, val: i32) -> i32 {
        js_sys::Atomics::add(&self.control, idx, val).unwrap_or(0)
    }

    fn atomic_notify(&self, idx: u32) {
        let _ = js_sys::Atomics::notify(&self.control, idx);
    }

    /// Available space for writing.
    fn write_available(&self) -> u32 {
        let head = self.atomic_load(self.write_head_idx) as u32;
        let tail = self.atomic_load(self.write_tail_idx) as u32;
        self.write_capacity.wrapping_sub(head.wrapping_sub(tail))
    }

    /// Available data for reading.
    fn read_available(&self) -> u32 {
        let head = self.atomic_load(self.read_head_idx) as u32;
        let tail = self.atomic_load(self.read_tail_idx) as u32;
        head.wrapping_sub(tail)
    }

    /// Write a length-prefixed message to the ring buffer.
    fn write_message(&self, data: &[u8]) {
        let msg_len = data.len() as u32;
        let total = HEADER_SIZE + msg_len;

        // Wait for space (spin — we're in a worker, blocking is OK)
        while self.write_available() < total + 256 {
            // Yield to other tasks
            // In a real implementation, use Atomics.waitAsync or sleep
        }

        let head = self.atomic_load(self.write_head_idx) as u32;
        let pos = head % self.write_capacity;

        // Check if message fits before wrap
        if pos + total > self.write_capacity {
            // Skip to next cycle
            self.atomic_store(
                self.write_head_idx,
                (head + (self.write_capacity - pos)) as i32,
            );
            return self.write_message(data); // retry from aligned position
        }

        // Write length prefix
        let len_bytes = msg_len.to_le_bytes();
        for (i, &b) in len_bytes.iter().enumerate() {
            self.write_buf.set_index(pos + i as u32, b);
        }

        // Write payload
        let payload = js_sys::Uint8Array::from(data);
        self.write_buf.set(&payload, pos + HEADER_SIZE);

        // Advance head
        self.atomic_store(self.write_head_idx, (head + total) as i32);
        self.atomic_add(self.write_msg_count_idx, 1);
        self.atomic_notify(self.read_tail_idx); // wake reader
    }

    /// Read a length-prefixed message from the ring buffer.
    fn read_message(&self) -> Option<Vec<u8>> {
        if self.read_available() < HEADER_SIZE {
            return None;
        }

        let tail = self.atomic_load(self.read_tail_idx) as u32;
        let pos = tail % self.read_capacity;

        // Check for gap (writer skipped to next cycle)
        if pos + HEADER_SIZE > self.read_capacity {
            // Skip to next cycle
            self.atomic_store(
                self.read_tail_idx,
                (tail + (self.read_capacity - pos)) as i32,
            );
            return self.read_message(); // retry
        }

        // Read length prefix
        let mut len_bytes = [0u8; 4];
        for i in 0..4 {
            len_bytes[i] = self.read_buf.get_index(pos + i as u32);
        }
        let msg_len = u32::from_le_bytes(len_bytes);

        if msg_len == 0 || self.read_available() < HEADER_SIZE + msg_len {
            return None; // incomplete message
        }

        // Read payload
        let mut data = vec![0u8; msg_len as usize];
        for i in 0..msg_len {
            data[i as usize] = self.read_buf.get_index(pos + HEADER_SIZE + i);
        }

        // Advance tail
        self.atomic_store(
            self.read_tail_idx,
            (tail + HEADER_SIZE + msg_len) as i32,
        );
        self.atomic_add(self.read_msg_count_idx, -1);
        self.atomic_notify(self.write_head_idx); // wake writer

        Some(data)
    }
}

#[cfg(target_arch = "wasm32")]
#[async_trait::async_trait(?Send)]
impl P9Transport for DmaTransport {
    async fn send(&self, data: &[u8]) -> anyhow::Result<()> {
        self.write_message(data);
        Ok(())
    }

    async fn recv(&self) -> anyhow::Result<Vec<u8>> {
        // Poll for message (TODO: use Atomics.waitAsync for non-blocking)
        loop {
            if let Some(msg) = self.read_message() {
                return Ok(msg);
            }
            // Yield to event loop
            wasm_bindgen_futures::JsFuture::from(js_sys::Promise::resolve(&JsValue::NULL))
                .await
                .map_err(|_| anyhow::anyhow!("yield failed"))?;
        }
    }
}
