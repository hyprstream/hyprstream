#!/bin/bash

MODELS_DIR="/private/birdetta/.local/share/hyprstream/models"
METADATA_FILE="$MODELS_DIR/metadata.json"

echo "🔍 Debugging metadata file..."
echo "📁 Models dir: $MODELS_DIR"
echo "📄 Metadata file: $METADATA_FILE"
echo ""

if [[ -f "$METADATA_FILE" ]]; then
    echo "✅ Metadata file exists"
    echo "📊 File size: $(wc -c < "$METADATA_FILE") bytes"
    echo ""
    echo "📄 Content:"
    cat "$METADATA_FILE"
    echo ""
    echo ""
    echo "🔍 Expected directory structure check:"
    EXPECTED_DIR="$MODELS_DIR/hf/Qwen/Qwen2-1.5B-Instruct-GGUF/fp16"
    if [[ -d "$EXPECTED_DIR" ]]; then
        echo "✅ Expected directory exists: $EXPECTED_DIR"
        echo "📂 Contents:"
        ls -la "$EXPECTED_DIR"
    else
        echo "❌ Expected directory missing: $EXPECTED_DIR" 
    fi
else
    echo "❌ Metadata file does not exist"
fi