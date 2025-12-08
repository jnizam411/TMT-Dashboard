#!/bin/bash
# Build script for TMT Rust Core
# Optimized for Apple Silicon

set -e

echo "=========================================="
echo "Building TMT Rust Core for Apple Silicon"
echo "=========================================="

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Check for maturin
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Set optimization flags for Apple Silicon
if [ "$ARCH" = "arm64" ]; then
    echo "Configuring for Apple Silicon (ARM64)..."
    export RUSTFLAGS="-C target-cpu=apple-m1 -C opt-level=3"
fi

# Build the Rust library
echo ""
echo "Building Rust library..."
cd "$(dirname "$0")"

# Development build
if [ "$1" = "--dev" ]; then
    echo "Development build..."
    maturin develop
else
    # Release build
    echo "Release build..."
    maturin build --release

    # Install the wheel
    echo ""
    echo "Installing wheel..."
    pip install ../target/wheels/*.whl --force-reinstall
fi

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Test the installation with:"
echo "  python -c \"import tmt_rust_core; print('Success!')\""
