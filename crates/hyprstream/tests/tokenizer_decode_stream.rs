//! Tests for DecodeStream behavior with BPE tokenizers
//!
//! These tests verify that DecodeStream correctly handles:
//! - Multi-byte UTF-8 sequences split across tokens
//! - Invalid/incomplete byte sequences
//! - Emoji and special characters

use tokenizers::Tokenizer;

#[test]
fn test_decode_stream_with_qwen_problematic_tokens() {
    // Load Qwen tokenizer
    let tokenizer_path = std::env::var("QWEN_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/home/birdetta/Qwen3/tokenizer.json".to_string());

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    // Test case 1: Token 229 + Token 30543 (the problematic sequence from logs)
    let tokens = vec![229, 30543];
    let decoded_batch = tokenizer.decode(&tokens, false)
        .expect("Failed to decode batch");

    println!("Batch decode [229, 30543]: {:?}", decoded_batch);

    // Test individual tokens
    let token_229 = tokenizer.decode(&[229], false)
        .expect("Failed to decode token 229");
    let token_30543 = tokenizer.decode(&[30543], false)
        .expect("Failed to decode token 30543");

    println!("Token 229 alone: {:?}", token_229);
    println!("Token 30543 alone: {:?}", token_30543);

    // Test with DecodeStream
    let mut decode_stream = tokenizer.decode_stream(false);

    let result_229 = decode_stream.step(229)
        .expect("DecodeStream error on token 229");
    println!("DecodeStream token 229: {:?}", result_229);

    let result_30543 = decode_stream.step(30543)
        .expect("DecodeStream error on token 30543");
    println!("DecodeStream token 30543: {:?}", result_30543);

    // Check if batch decode produces replacement character
    let has_replacement = decoded_batch.contains('�');

    if has_replacement {
        panic!(
            "Batch decode produced replacement character: {:?}\nThis means the model is generating invalid token sequences.",
            decoded_batch
        );
    } else {
        println!("✓ Batch decode is valid: {:?}", decoded_batch);

        // If batch is valid but DecodeStream produced �, that's a DecodeStream bug
        if let Some(text) = result_30543 {
            if text.contains('�') {
                panic!(
                    "DecodeStream produced � but batch decode didn't.\nDecodeStream result: {:?}\nBatch result: {:?}",
                    text, decoded_batch
                );
            }
        }
    }
}

#[test]
fn test_decode_stream_emoji_sequences() {
    let tokenizer_path = std::env::var("QWEN_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/home/birdetta/Qwen3/tokenizer.json".to_string());

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    // Encode some emojis and test streaming decode
    let test_strings = vec![
        "🔥",
        "💥",
        "👑",
        "🎯",
        "🏀",
        "Test 🔥 emoji",
        "Multiple 💥🏀 emojis",
    ];

    for test_str in test_strings {
        println!("\n=== Testing: {:?} ===", test_str);

        let encoding = tokenizer.encode(test_str, false)
            .expect("Failed to encode");
        let tokens = encoding.get_ids();

        println!("Tokens: {:?}", tokens);

        // Batch decode
        let batch_decoded = tokenizer.decode(tokens, false)
            .expect("Failed to batch decode");
        println!("Batch decoded: {:?}", batch_decoded);

        // Stream decode
        let mut decode_stream = tokenizer.decode_stream(false);
        let mut stream_decoded = String::new();

        for &token in tokens {
            match decode_stream.step(token) {
                Ok(Some(text)) => {
                    println!("  Token {} -> {:?}", token, text);
                    stream_decoded.push_str(&text);
                }
                Ok(None) => {
                    println!("  Token {} -> buffering", token);
                }
                Err(e) => {
                    panic!("DecodeStream error on token {}: {}", token, e);
                }
            }
        }

        println!("Stream decoded: {:?}", stream_decoded);

        // Compare batch vs stream
        if batch_decoded != stream_decoded {
            panic!(
                "Mismatch!\nOriginal: {:?}\nBatch:    {:?}\nStream:   {:?}",
                test_str, batch_decoded, stream_decoded
            );
        }

        // Check for replacement characters
        if stream_decoded.contains('�') {
            panic!(
                "Stream decode produced replacement character!\nOriginal: {:?}\nResult:   {:?}",
                test_str, stream_decoded
            );
        }

        println!("✓ Passed");
    }
}

#[test]
fn test_decode_stream_with_token_11162() {
    // Token 11162 from logs: (raw: " �") -> buffering (incomplete UTF-8)
    let tokenizer_path = std::env::var("QWEN_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/home/birdetta/Qwen3/tokenizer.json".to_string());

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    let token_11162 = tokenizer.decode(&[11162], false)
        .expect("Failed to decode token 11162");

    println!("Token 11162 decoded: {:?}", token_11162);
    println!("Token 11162 bytes: {:?}", token_11162.as_bytes());

    // Check if it's actually invalid UTF-8 or just looks like it
    if token_11162.contains('�') {
        println!("⚠ Token 11162 IS a replacement character in the vocabulary");
    } else {
        println!("✓ Token 11162 is valid UTF-8");
    }
}

#[test]
fn test_bpe_byte_level_tokens() {
    let tokenizer_path = std::env::var("QWEN_TOKENIZER_PATH")
        .unwrap_or_else(|_| "/home/birdetta/Qwen3/tokenizer.json".to_string());

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    // Get info about problematic tokens
    let problematic_tokens = vec![229, 11162];

    for token_id in problematic_tokens {
        let decoded = tokenizer.decode(&[token_id], false)
            .expect("Failed to decode");

        println!("\nToken {}: {:?}", token_id, decoded);
        println!("  Bytes: {:?}", decoded.as_bytes());
        println!("  Len: {} bytes, {} chars", decoded.len(), decoded.chars().count());
        println!("  Has replacement char: {}", decoded.contains('�'));

        // Try to get the actual bytes if possible
        if let Some(vocab) = tokenizer.get_vocab(false).get(&token_id.to_string()) {
            println!("  In vocab as: {:?}", vocab);
        }
    }
}
