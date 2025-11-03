//! Debug token 229 - compare raw vocab vs decoded output

use tokenizers::Tokenizer;

#[test]
fn debug_token_229_deeply() {
    let tokenizer_path = "/home/birdetta/.local/share/hyprstream/models/qwen3-0.6b/worktrees/main/tokenizer.json";
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .expect("Failed to load tokenizer");

    // Get vocab to see what token 229 SHOULD be
    let vocab = tokenizer.get_vocab(false);

    // Find token 229 in vocab
    for (token_str, &token_id) in vocab.iter() {
        if token_id == 229 {
            println!("Token 229 in vocab: {:?}", token_str);
            println!("  UTF-8 bytes: {:?}", token_str.as_bytes());
            println!("  Char: U+{:04X}", token_str.chars().next().unwrap() as u32);
            break;
        }
    }

    // Now decode it
    let decoded = tokenizer.decode(&[229], false).unwrap();
    println!("\nToken 229 decoded: {:?}", decoded);
    println!("  UTF-8 bytes: {:?}", decoded.as_bytes());
    if let Some(c) = decoded.chars().next() {
        println!("  Char: U+{:04X}", c as u32);
    }

    // Are they the same?
    println!("\n=== Comparison ===");
    println!("Vocab contains 'ĩ' (U+0129): {}", vocab.contains_key("ĩ"));
    println!("Decoded == '�': {}", decoded == "�");
    println!("U+FFFD in bytes: {:?}", "�".as_bytes());
}
