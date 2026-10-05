#!/usr/bin/env bash
set -euo pipefail
python3 benchmark_sdr_synthetic.py \
  --fft-sizes 4096,8192,16384,32768,65536 \
  --workers 1,2,3,4 \
  --modes auto,full-fft \
  --waveforms tone,multitone,ofdm,wideband-noise,dsss \
  --sample-rates 2000000,5000000,10000000,20000000 \
  --snrs -6,0,6,12,18,24 \
  --freqs 800000000,2437000000,3000000000,6000000000 \
  --angles=-60,-30,0,30,60 \
  --trials 20 \
  --warmup 3 \
  --repeats 25 \
  --multitarget-trials 20 \
  --json benchmark_results.json
