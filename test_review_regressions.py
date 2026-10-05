#!/usr/bin/env python3
"""Focused regression tests for the SDR review fixes.

Runs without HackRF.  On ARM the native library is exercised automatically when built.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from doa_h618_core import SDRCore
import drone_doa_h618 as driver
from drone_doa_h618 import FixedBatchReader, ReadyFrame, stop_event, GainGovernor, GainChangeRequest


def test_exact_bins_matches_fft() -> None:
    rng = np.random.default_rng(10)
    n = 8192
    core = SDRCore(n, 5e6, 2.437e9, 0.05, batch_frames=1, fft_workers=1,
                   slave_mode="exact-bins", use_neon=True)
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    core.slave_td[0] = x
    bins = np.asarray([0, 1, 7, 123, 2047, 4096, 8191], dtype=np.int64)
    got = core._exact_bins(0, bins)
    ref = np.fft.fft(x.astype(np.complex128))[bins]
    rel = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-20))
    assert rel < 2e-5, rel
    print(f"[PASS] exact-bin/root-table equivalence rel_err={rel:.3e}")


def test_cfar_double_accumulator() -> None:
    rng = np.random.default_rng(11)
    n = 65536
    core = SDRCore(n, 5e6, 2.437e9, 0.05, batch_frames=1, fft_workers=1,
                   slave_mode="full-fft", use_neon=True)
    p = np.exp(rng.normal(loc=0.0, scale=1.0, size=n)).astype(np.float32)
    out = np.empty(n, np.float32)
    core.neon.lib.cfar_noise_circular_f32(core.neon._f32_ptr(p), n, 128, 8,
                                          core.neon._f32_ptr(out))
    ref = np.empty(n, np.float64)
    for i in range(n):
        idx_l = [(i - d) % n for d in range(9, 137)]
        idx_r = [(i + d) % n for d in range(9, 137)]
        ref[i] = (np.sum(p[idx_l], dtype=np.float64) + np.sum(p[idx_r], dtype=np.float64)) / 256.0
    err = float(np.max(np.abs(out.astype(np.float64) - ref)))
    assert err < 3e-5, err
    print(f"[PASS] CFAR double accumulator max_abs_err={err:.3e}")


def test_reader_soft_skew_no_restart() -> None:
    stop_event.clear()
    r = FixedBatchReader(-1, -1, 16, slots=8, max_gap=2, hard_gap=64)
    # Simulate the same state the producers create: remove occupied slots from free queues.
    for _ in range(6):
        r.master_free.get_nowait()
    for _ in range(6):
        r.slave_free.get_nowait()
    for i in (100, 101, 102):
        r.master_ready.put_nowait(ReadyFrame(i, i - 100, 1000 + i))
    for i in (97, 98, 99, 100, 101, 102):
        r.slave_ready.put_nowait(ReadyFrame(i, i - 98, 2000 + i))
    pair = r.get_pair()
    assert pair[0] == 100
    assert r.soft_gap_events >= 1
    print(f"[PASS] soft frame skew recovered pair={pair[0]} events={r.soft_gap_events}")


def test_calibration_success_path() -> None:
    """Exercise the real calibrate_alignment success branch, including its print path."""
    rng = np.random.default_rng(123)
    n = 131072
    decim = 4
    true_delta = 37
    master = (rng.normal(0, 0.35, n) + 1j * rng.normal(0, 0.35, n)).astype(np.complex64)
    start0 = 20000
    burst_len = 12000
    t = np.arange(burst_len)
    burst = (4.0 * np.exp(2j * np.pi * (0.017 * t + 0.000001 * t * t))).astype(np.complex64)
    burst *= (0.25 + 0.75 * np.sin(np.pi * np.arange(burst_len) / burst_len) ** 2).astype(np.float32)
    master[start0:start0 + burst_len] += burst

    slave = (rng.normal(0, 0.35, n) + 1j * rng.normal(0, 0.35, n)).astype(np.complex64)
    slave[start0 + true_delta:start0 + true_delta + burst_len] += burst

    def to_raw(z: np.ndarray) -> bytes:
        raw = np.empty(z.size * 2, dtype=np.int8)
        raw[0::2] = np.clip(np.rint(z.real * 8), -120, 120).astype(np.int8)
        raw[1::2] = np.clip(np.rint(z.imag * 8), -120, 120).astype(np.int8)
        return raw.tobytes()

    bm, bs = to_raw(master), to_raw(slave)

    class _DummyStdout:
        def __init__(self, fd: int):
            self.fd = fd
        def fileno(self) -> int:
            return self.fd

    class _DummyProc:
        def __init__(self, fd: int):
            self.stdout = _DummyStdout(fd)

    original_read = driver.read_exact_fd
    original_discard = driver.discard_exact_fd
    data = {101: bm, 102: bs}
    driver.stop_event.clear()

    def fake_read(fd: int, buf: bytearray) -> int:
        payload = data[fd]
        buf[:len(payload)] = payload
        return len(payload)

    driver.read_exact_fd = fake_read
    driver.discard_exact_fd = lambda fd, count: None
    try:
        got = driver.calibrate_alignment(_DummyProc(101), _DummyProc(102), 5e6, timeout_s=1.0)
    finally:
        driver.read_exact_fd = original_read
        driver.discard_exact_fd = original_discard
        driver.stop_event.clear()
    assert got == true_delta, got
    print(f"[PASS] real calibration success path lag={got:+d} samples")


def test_gain_governor() -> None:
    quiet = np.zeros(4096, dtype=np.int8).tobytes()
    g = GainGovernor(32, 30, enabled=True, sustain_frames=2)
    g.observe(quiet, quiet)
    clipped = np.full(4096, 127, dtype=np.int8).tobytes()
    try:
        g.observe(clipped, clipped)
        g.observe(clipped, clipped)
    except GainChangeRequest as exc:
        assert (exc.lna, exc.vga) == (32, 24), (exc.lna, exc.vga)
        print(f"[PASS] gain governor requests safe reduction {exc.lna}/{exc.vga} dB")
        return
    raise AssertionError("gain governor did not trigger")


def main() -> int:
    tests = [
        test_exact_bins_matches_fft,
        test_cfar_double_accumulator,
        test_reader_soft_skew_no_restart,
        test_calibration_success_path,
        test_gain_governor,
    ]
    for fn in tests:
        fn()
    print("ALL REVIEW REGRESSION TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
