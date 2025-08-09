#!/bin/bash

# Script to create properly formatted metadata.json with correct types

MODELS_DIR="/private/birdetta/.local/share/hyprstream/models"
METADATA_FILE="$MODELS_DIR/metadata.json"
MODEL_FILE="$MODELS_DIR/Qwen2-1.5B-Instruct-GGUF_qwen2-1_5b-instruct-fp16.gguf"

echo "📝 Creating metadata with proper types..."

# Check if model file exists  
if [[ ! -f "$MODEL_FILE" ]]; then
    echo "❌ Model file not found: $MODEL_FILE"
    exit 1
fi

# Get file size
FILE_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null)
CURRENT_TIMESTAMP=$(date +%s)

# Create metadata JSON ensuring all field types match ModelMetadata struct
cat > "$METADATA_FILE" << EOF
{
  "hf://Qwen/Qwen2-1.5B-Instruct-GGUF:fp16": {
    "uri": {
      "registry": "hf",
      "org": "Qwen", 
      "name": "Qwen2-1.5B-Instruct-GGUF",
      "revision": "fp16",
      "uri": "hf://Qwen/Qwen2-1.5B-Instruct-GGUF:fp16"
    },
    "size_bytes": $FILE_SIZE,
    "files": ["qwen2-1_5b-instruct-fp16.gguf"],
    "model_type": "gguf", 
    "architecture": "qwen",
    "created_at": $CURRENT_TIMESTAMP,
    "last_accessed": $CURRENT_TIMESTAMP,
    "parameters": 1500000000,
    "tokenizer_type": "tiktoken",
    "tags": ["instruct"]
  }
}
EOF

echo "✅ Created metadata: $METADATA_FILE"

# Test that the target file can be accessed through the symlink
echo ""
echo "🔍 Testing file access..."
EXPECTED_PATH="/private/birdetta/.local/share/hyprstream/models/hf/Qwen/Qwen2-1.5B-Instruct-GGUF/fp16/qwen2-1_5b-instruct-fp16.gguf"
if [[ -f "$EXPECTED_PATH" ]]; then
    echo "✅ Model file accessible at expected path: $EXPECTED_PATH"
    echo "📊 File size through symlink: $(stat -f%z "$EXPECTED_PATH" 2>/dev/null || stat -c%s "$EXPECTED_PATH" 2>/dev/null) bytes"
else
    echo "❌ Model file not accessible at expected path: $EXPECTED_PATH"
fi