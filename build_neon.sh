#!/usr/bin/env bash
set -euo pipefail
CC="${CC:-gcc}"
ARCH="$(uname -m)"
CFLAGS=(-O3 -ffast-math -fomit-frame-pointer -fPIC -shared)
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
  CFLAGS+=( -mcpu=cortex-a53 -mtune=cortex-a53 )
  echo "[+] Building ARM64 Cortex-A53/NEON library"
else
  echo "[!] Host is $ARCH; building portable scalar library for smoke tests"
fi
"$CC" "${CFLAGS[@]}" -o libneon_kernels.so neon_kernels.c -lm
chmod +x libneon_kernels.so
printf '[+] built %s\n' "$(pwd)/libneon_kernels.so"
