#!/bin/bash

# Script to create properly formatted metadata.json

MODELS_DIR="/private/birdetta/.local/share/hyprstream/models"
METADATA_FILE="$MODELS_DIR/metadata.json"
MODEL_FILE="$MODELS_DIR/Qwen2-1.5B-Instruct-GGUF_qwen2-1_5b-instruct-fp16.gguf"

echo "📝 Creating properly formatted metadata..."

# Check if model file exists
if [[ ! -f "$MODEL_FILE" ]]; then
    echo "❌ Model file not found: $MODEL_FILE"
    exit 1
fi

# Get file size
FILE_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null)
CURRENT_TIMESTAMP=$(date +%s)

# Create metadata JSON with correct structure
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

echo "✅ Created metadata file: $METADATA_FILE"
echo "📊 Model size: $FILE_SIZE bytes"
echo "⏰ Timestamp: $CURRENT_TIMESTAMP"

# Verify the JSON is valid
if command -v jq &> /dev/null; then
    echo ""
    echo "✅ JSON validation:"
    jq . "$METADATA_FILE" > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
else
    echo "💡 Install jq to validate JSON format"
fi