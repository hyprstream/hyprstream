use molt::Interp;

fn main() {
    let mut interp = Interp::new();
    
    println!("Testing puts only:");
    match interp.eval(r#"puts "Hello World""#) {
        Ok(value) => println!("Return value: '{}'", value),
        Err(e) => println!("Error: {:?}", e),
    }
    
    println!("\nTesting return only:");
    match interp.eval(r#"return "Hello World""#) {
        Ok(value) => println!("Return value: '{}'", value),
        Err(e) => println!("Error: {:?}", e),
    }
    
    println!("\nTesting puts followed by return:");
    match interp.eval(r#"puts "Hello World"; return "Goodbye""#) {
        Ok(value) => println!("Return value: '{}'", value),
        Err(e) => println!("Error: {:?}", e),
    }
    
    println!("\nTesting just puts with no return:");
    match interp.eval(r#"puts "Hello World"; set x 42"#) {
        Ok(value) => println!("Return value: '{}'", value),
        Err(e) => println!("Error: {:?}", e),
    }
}