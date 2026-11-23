#!/bin/sh
# build.sh -- build libgfx9probe.so for gfx900 against a radiance checkout's abi/ headers.
#
#   ./build.sh /path/to/radiance
#   RADIANCE=/path/to/radiance ./build.sh
#
# The headers are the only thing needed: a plugin links nothing of the engine's, which is what
# makes an out-of-tree card library possible at all.
set -eu

RADIANCE="${1:-${RADIANCE:-}}"
if [ -z "$RADIANCE" ] || [ ! -f "$RADIANCE/abi/rad_abi.h" ]; then
    echo "build.sh: give the path to a radiance checkout (or set RADIANCE)" >&2
    exit 2
fi

HIPCC="${HIPCC:-hipcc}"
ARCH="${ARCH:-gfx900}"

command -v "$HIPCC" >/dev/null 2>&1 || { echo "build.sh: no $HIPCC on PATH" >&2; exit 2; }

set -x
"$HIPCC" -x hip --offload-arch="$ARCH" -std=c++20 -O2 -fPIC -shared \
    -I"$RADIANCE/abi" -o libgfx9probe.so gfx9probe.hip
set +x

echo "built libgfx9probe.so for $ARCH"
echo "next: cp libgfx9probe.so \"\${RADIANCE_HOME:-$RADIANCE/build/radiance_home}/kernels/\""
