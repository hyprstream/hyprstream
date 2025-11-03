//! Deep dive into token 229 and what it represents

use tokenizers::Tokenizer;

#[test]
fn investigate_token_229() {
    let tokenizer_path = "/home/birdetta/.local/share/hyprstream/models/qwen3-0.6b/worktrees/main/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .expect("Failed to load tokenizer");

    println!("\n=== Token 229 Investigation ===\n");

    // Decode token 229
    let decoded = tokenizer.decode(&[229], false)
        .expect("Failed to decode");
    println!("Token 229 decoded (skip_special=false): {:?}", decoded);
    println!("  Bytes: {:?}", decoded.as_bytes());
    println!("  Len: {} bytes", decoded.len());
    println!("  Char count: {}", decoded.chars().count());

    let decoded_skip = tokenizer.decode(&[229], true)
        .expect("Failed to decode");
    println!("Token 229 decoded (skip_special=true): {:?}", decoded_skip);

    // Check if it's in added tokens (special tokens)
    if let Some(added_tokens) = tokenizer.get_added_tokens_decoder().get(&229) {
        println!("\n✓ Token 229 IS an added/special token:");
        println!("  Content: {:?}", added_tokens.content);
        println!("  Special: {}", added_tokens.special);
        println!("  Single word: {}", added_tokens.single_word);
        println!("  Lstrip: {}", added_tokens.lstrip);
        println!("  Rstrip: {}", added_tokens.rstrip);
        println!("  Normalized: {}", added_tokens.normalized);
    } else {
        println!("\n✗ Token 229 is NOT an added/special token (it's a regular vocab token)");
    }

    // Check token 30543 too
    println!("\n=== Token 30543 Investigation ===\n");
    let decoded_30543 = tokenizer.decode(&[30543], false)
        .expect("Failed to decode");
    println!("Token 30543 decoded: {:?}", decoded_30543);
    println!("  Bytes: {:?}", decoded_30543.as_bytes());

    if let Some(added_tokens) = tokenizer.get_added_tokens_decoder().get(&30543) {
        println!("\n✓ Token 30543 IS an added/special token:");
        println!("  Content: {:?}", added_tokens.content);
        println!("  Special: {}", added_tokens.special);
    } else {
        println!("\n✗ Token 30543 is NOT an added/special token");
    }

    // Test encoding some text with replacement character
    println!("\n=== Encoding Test ===\n");
    let test_texts = vec![
        "�",
        "\u{fe0f}",
        "�\u{fe0f}",
        "🏀",
        "🔥",
    ];

    for text in test_texts {
        let encoding = tokenizer.encode(text, false)
            .expect("Failed to encode");
        let tokens = encoding.get_ids();
        println!("Text {:?} -> tokens: {:?}", text, tokens);

        // Try to decode back
        let decoded = tokenizer.decode(tokens, false)
            .expect("Failed to decode");
        println!("  Decoded back: {:?}", decoded);
        if decoded != text {
            println!("  ⚠️  MISMATCH!");
        }
    }

    // Check what tokens are around 229 in vocab
    println!("\n=== Tokens around 229 ===\n");
    for id in 227..=231 {
        let decoded = tokenizer.decode(&[id], false)
            .unwrap_or_else(|_| format!("<error>"));
        let is_special = tokenizer.get_added_tokens_decoder().contains_key(&id);
        println!("Token {}: {:?} (special: {})", id, decoded, is_special);
    }
}
