#!/usr/bin/env bash
# collect-hashes.sh - Collect cargo git dependency hashes for Nix build
#
# Runs `nix build` repeatedly, extracting correct hashes from error messages
# and patching nix/hyprstream.nix automatically.
#
# Usage: ./nix/collect-hashes.sh
#
set -euo pipefail

NIX_FILE="nix/hyprstream.nix"
MAX_ITERATIONS=50

for i in $(seq 1 $MAX_ITERATIONS); do
  echo "=== Attempt $i ==="
  
  # Try to build, capture stderr (where Nix prints hash mismatches)
  output=$(nix build .#hyprstream-cpu --no-link 2>&1) || true
  
  # Look for the hash mismatch pattern:
  # "hash mismatch in fixed-output derivation '/nix/store/...':
  #   specified: sha256-AAAAAAAA...
  #      got:    sha256-<actual>..."
  specified=$(echo "$output" | grep -oP 'specified: sha256-\K[A-Za-z0-9+/=]+' | head -1)
  got=$(echo "$output" | grep -oP 'got:    sha256-\K[A-Za-z0-9+/=]+' | head -1)
  
  if [ -z "$specified" ] || [ -z "$got" ]; then
    # Check for other error types
    if echo "$output" | grep -q "error:"; then
      echo "Build failed with non-hash error:"
      echo "$output" | grep "error:" | head -5
      exit 1
    fi
    echo "No hash mismatch found. Build may have succeeded!"
    exit 0
  fi
  
  echo "Replacing hash: $specified -> $got"
  # Replace the placeholder hash with the real one
  sed -i "s|$specified|$got|" "$NIX_FILE"
  
  # Also handle the lib.fakeHash pattern (base32 format)
  fake_pattern="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
  if grep -q "$fake_pattern" "$NIX_FILE"; then
    echo "Still have fakeHash entries, continuing..."
  fi
done

echo "Reached max iterations ($MAX_ITERATIONS). There may be more hashes to collect."
