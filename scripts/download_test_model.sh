#!/bin/bash
# Download a small GGUF model for testing MistralEngine

set -e

MODEL_DIR="models"
MODEL_URL="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
MODEL_FILE="phi-3-mini-4k-instruct-q4.gguf"

echo "📦 Downloading small test model for MistralEngine..."
echo "URL: $MODEL_URL"
echo "Destination: $MODEL_DIR/$MODEL_FILE"

# Create models directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Download the model if it doesn't exist
if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "⬇️  Downloading model..."
    wget -q --show-progress "$MODEL_URL" -O "$MODEL_DIR/$MODEL_FILE"
    echo "✅ Model downloaded successfully"
else
    echo "✅ Model already exists"
fi

echo "📊 Model info:"
ls -lh "$MODEL_DIR/$MODEL_FILE"

echo ""
echo "🚀 To test the model, run:"
echo "   cargo run --example mistral_engine_basic"
echo ""
echo "💡 Or use it in code:"
echo "   let model_path = Path::new(\"$MODEL_DIR/$MODEL_FILE\");"