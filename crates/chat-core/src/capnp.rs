//! Minimal Cap'n Proto builder for ChatCoreOut messages.
//!
//! Builds single-segment Cap'n Proto messages without any external dependencies.
//! Follows the wire format: [segment_count-1: u32] [segment0_size_words: u32] [segment0_bytes...]
//!
//! ChatCoreOut union layout (matching chat_core.capnp):
//!   Data section: 1 word (8 bytes)
//!     - bytes 0-1: union discriminant (u16 LE)
//!   Pointer section: 1 pointer (8 bytes)
//!     - ptr 0: Text for content/thinking/error, or struct for toolCallDetected
//!
//! Union discriminants:
//!   0 = content (Text)
//!   1 = thinking (Text)
//!   2 = toolCallDetected (ToolCallDetectedMsg struct)
//!   3 = complete (Void)
//!   4 = error (Text)

const WORD: usize = 8;

/// Build a ChatCoreOut message with a Text payload (content, thinking, error).
pub fn build_text_event(discriminant: u16, text: &str) -> Vec<u8> {
    let text_bytes = text.as_bytes();
    let text_len = text_bytes.len() + 1; // +1 for NUL terminator
    let text_words = text_len.div_ceil(WORD);

    // Segment: root struct (1 data word + 1 ptr word) + text list
    let segment_words = 2 + text_words;
    let total = 8 + segment_words * WORD; // 8 bytes header + segment

    let mut buf = vec![0u8; total];

    // Message header: segment_count - 1 = 0, segment_size_words
    buf[0..4].copy_from_slice(&0u32.to_le_bytes()); // 1 segment
    buf[4..8].copy_from_slice(&(segment_words as u32).to_le_bytes());

    // Root struct pointer at offset 0 (self-referencing: offset=0)
    // Data offset = 0 words from pointer, data_size = 1 word, ptr_size = 1
    // But in flat format, the struct starts immediately after the header.
    // The root pointer IS the first word of the segment in standard format,
    // but for flat (no root pointer) format used by hyprstream, data starts at offset 0.

    // Data word: union discriminant at bytes 0-1
    let data_offset = 8; // after header
    buf[data_offset..data_offset + 2].copy_from_slice(&discriminant.to_le_bytes());

    // Pointer word: text list pointer
    // List pointer: offset_words | 0b01 (list tag) | element_size=2 (1 byte) | count
    let ptr_offset = data_offset + WORD;
    let list_offset_words: i32 = 0; // text starts right after ptr section
    let ptr_val: u64 = ((list_offset_words as u32 as u64) << 2)
        | 1 // list tag
        | ((2u64) << 32) // element size = byte
        | ((text_len as u64) << 35); // element count
    buf[ptr_offset..ptr_offset + 8].copy_from_slice(&ptr_val.to_le_bytes());

    // Text data (NUL-terminated)
    let text_offset = ptr_offset + WORD;
    buf[text_offset..text_offset + text_bytes.len()].copy_from_slice(text_bytes);
    // NUL terminator is already 0 from vec initialization

    buf
}

/// Build a ChatCoreOut.complete message (Void, no payload).
pub fn build_complete() -> Vec<u8> {
    // Just a struct with discriminant=3, no pointer data
    let segment_words = 2; // 1 data word + 1 ptr word (empty)
    let total = 8 + segment_words * WORD;
    let mut buf = vec![0u8; total];

    buf[0..4].copy_from_slice(&0u32.to_le_bytes());
    buf[4..8].copy_from_slice(&(segment_words as u32).to_le_bytes());

    // discriminant = 3 (complete)
    buf[8..10].copy_from_slice(&3u16.to_le_bytes());

    buf
}

/// Build a ChatCoreOut.toolCallDetected message.
/// The ToolCallDetectedMsg struct has 4 text fields:
///   ptr 0: id, ptr 1: uuid, ptr 2: description, ptr 3: arguments
pub fn build_tool_call_detected(id: &str, uuid: &str, description: &str, arguments: &str) -> Vec<u8> {
    // For simplicity, encode as JSON in a text field rather than nested struct.
    // This avoids complex multi-pointer Cap'n Proto layout.
    // The TS side can parse either native capnp or this JSON-in-text form.
    let json = format!(
        r#"{{"id":"{}","uuid":"{}","description":"{}","arguments":{}}}"#,
        id.replace('"', r#"\""#),
        uuid.replace('"', r#"\""#),
        description.replace('"', r#"\""#),
        arguments, // already JSON
    );
    build_text_event(2, &json)
}

/// Encode multiple ChatCoreOut events into a single buffer.
/// Format: [u32 LE count] [u32 LE len0] [event0 bytes] [u32 LE len1] [event1 bytes] ...
pub fn encode_event_list(events: &[Vec<u8>]) -> Vec<u8> {
    let total_len: usize = 4 + events.iter().map(|e| 4 + e.len()).sum::<usize>();
    let mut buf = Vec::with_capacity(total_len);
    buf.extend_from_slice(&(events.len() as u32).to_le_bytes());
    for event in events {
        buf.extend_from_slice(&(event.len() as u32).to_le_bytes());
        buf.extend_from_slice(event);
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_text_event() {
        let msg = build_text_event(0, "hello");
        // Should be: 8 header + 8 data + 8 ptr + 8 text = 32 bytes
        assert_eq!(msg.len(), 32);
        // Discriminant at offset 8
        assert_eq!(u16::from_le_bytes([msg[8], msg[9]]), 0);
    }

    #[test]
    fn test_build_complete() {
        let msg = build_complete();
        assert_eq!(msg.len(), 24); // 8 header + 16 struct
        assert_eq!(u16::from_le_bytes([msg[8], msg[9]]), 3);
    }

    #[test]
    fn test_encode_event_list() {
        let events = vec![
            build_text_event(0, "hi"),
            build_complete(),
        ];
        let encoded = encode_event_list(&events);
        let count = u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        assert_eq!(count, 2);
    }
}
