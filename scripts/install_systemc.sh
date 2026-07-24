#!/usr/bin/env bash

set -euo pipefail

# Configuration
VERSION="2.3.3"
ARCHIVE="systemc-${VERSION}.tar.gz"
URL="https://github.com/accellera-official/systemc/archive/refs/tags/${VERSION}.tar.gz"
PREFIX="/opt/tools/systemc-${VERSION}"
BUILD_DIR="/tmp/systemc-build"

echo "Installing SystemC ${VERSION}..."

# Install required tools (Debian/Ubuntu)
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake wget
fi

# Clean previous build
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Download source
echo "Downloading SystemC..."
wget -O "${ARCHIVE}" "${URL}"

# Extract
tar -xzf "${ARCHIVE}"
cd "systemc-${VERSION}"

# Configure
mkdir build
cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5

# Build
echo "Building..."
make -j"$(nproc)"

# Install
echo "Installing..."
sudo make install

# Add environment variables to ~/.bashrc
if ! grep -q "^export SYSTEMC_HOME=${PREFIX}$" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "# SystemC ${VERSION}"
        echo "export SYSTEMC_HOME=${PREFIX}"
        echo 'export LD_LIBRARY_PATH=$SYSTEMC_HOME/lib-linux64:$LD_LIBRARY_PATH'
        echo 'export CPLUS_INCLUDE_PATH=$SYSTEMC_HOME/include:$CPLUS_INCLUDE_PATH'
    } >> "$HOME/.bashrc"
fi

echo ""
echo "SystemC ${VERSION} installed successfully."
echo "Installation directory: ${PREFIX}"
echo ""
echo "To use it:"
echo "  export SYSTEMC_HOME=${PREFIX}"
echo '  export LD_LIBRARY_PATH=$SYSTEMC_HOME/lib-linux64:$LD_LIBRARY_PATH'
echo '  export CPLUS_INCLUDE_PATH=$SYSTEMC_HOME/include:$CPLUS_INCLUDE_PATH'

