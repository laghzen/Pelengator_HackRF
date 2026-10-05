from __future__ import annotations

import ctypes
import math
import os
import platform
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional, Sequence

import numpy as np
import scipy.fft

C = 299_792_458.0


@dataclass(slots=True)
class SpectralCandidate:
    frame: int
    peak_idx: int
    start_idx: int
    end_idx: int
    snr_db: float
    effective_freq_hz: float
    occupied_bins: int
    metric: str = "peak-CFAR"


@dataclass(slots=True)
class Detection:
    frame: int
    peak_idx: int
    start_idx: int
    end_idx: int
    freq_offset_hz: float
    target_freq_hz: float
    snr_db: float
    phase_deg: float
    angle_deg: float
    coherence: float
    occupied_bins: int
    slave_compute_mode: str
    detection_metric: str
    baseline_unambiguous: bool
    kinematic_angle_deg: float | None = None
    kinematic_score: float | None = None


@dataclass(slots=True)
class KinematicEstimate:
    angle_deg: float
    score: float
    samples: int
    span_deg: float


def wrap_phase(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


class NeonKernels:
    """Native ARM64 kernels; on aarch64 the supplied C library uses ARM NEON."""

    def __init__(self, library_path: Optional[str | os.PathLike[str]] = None):
        self.lib = None
        self.available = False
        self.neon_enabled = False
        if library_path is None:
            library_path = Path(__file__).with_name("libneon_kernels.so")
        path = Path(library_path)
        if not path.exists():
            return
        try:
            self.lib = ctypes.CDLL(str(path))
            self._bind()
            self.available = True
            self.neon_enabled = bool(self.lib.neon_enabled())
        except OSError:
            self.lib = None
            self.available = False

    @staticmethod
    def _f32_ptr(a: np.ndarray):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    @staticmethod
    def _i8_ptr(a: np.ndarray):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))

    def near_rail_fraction(self, raw: bytes | bytearray | memoryview, rail: int = 126) -> float:
        a = np.frombuffer(raw, dtype=np.int8)
        if a.size == 0:
            return 0.0
        if self.available and hasattr(self.lib, "iq_near_rail_fraction"):
            return float(self.lib.iq_near_rail_fraction(self._i8_ptr(a), a.size, int(rail)))
        # Portable fallback samples every 8th byte to keep the watchdog cheap.
        aa = np.abs(a.astype(np.int16))[::8]
        return float(np.mean(aa >= int(rail))) if aa.size else 0.0

    def _bind(self) -> None:
        lib = self.lib
        lib.neon_enabled.argtypes = []
        lib.neon_enabled.restype = ctypes.c_int

        lib.iq_batch_to_c64_hann_split.argtypes = [
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int8),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t,
        ]
        lib.iq_batch_to_c64_hann_split.restype = None

        lib.iq_near_rail_fraction.argtypes = [
            ctypes.POINTER(ctypes.c_int8), ctypes.c_size_t, ctypes.c_int8,
        ]
        lib.iq_near_rail_fraction.restype = ctypes.c_float

        lib.power_c64_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ]
        lib.power_c64_f32.restype = None

        lib.cross_sum_range_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ]
        lib.cross_sum_range_f32.restype = None

        lib.cfar_noise_circular_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.cfar_noise_circular_f32.restype = None
        lib.windowed_ratio_best_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_float,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.windowed_ratio_best_f32.restype = None

        lib.dft_bins_c64_root.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
        ]
        lib.dft_bins_c64_root.restype = None


class SDRCore:
    """H618 DSP core with correctness-first automatic Slave processing.

    Processing policy:
      * Master: full FFT for detection.
      * Detection: local CA-CFAR-like threshold + spectral connected components.
      * Slave in `auto`: exact DFT only when the measured occupied band is genuinely
        narrow and the total sparse workload is bounded; otherwise full Slave FFT.
      * DOA phase is integrated over the entire detected occupied band, not 5 fixed bins.
      * Coherence is checked before emitting an angle.
    """

    def __init__(
        self,
        n_fft: int,
        sample_rate: float,
        center_freq_hz: float,
        baseline_m: float,
        threshold_snr_db: float = 12.0,
        phase_calibration_deg: float = 0.0,
        batch_frames: int = 4,
        fft_workers: int = 3,
        phase_bins: int = 5,
        use_neon: bool = True,
        neon_library: Optional[str] = None,
        slave_mode: str = "auto",
        cfar_train: int = 24,
        cfar_guard: int = 4,
        cfar_pfa: float = 1e-4,
        phase_occupancy_db: float = 3.0,
        sparse_max_bins: int = 32,
        coherence_min: float = 0.20,
        peak_cluster_gap_bins: int = 4,
    ):
        if n_fft < 16 or n_fft & (n_fft - 1):
            raise ValueError("n_fft must be a power of two >= 16")
        if batch_frames < 1:
            raise ValueError("batch_frames must be >= 1")
        if slave_mode not in {"auto", "exact-bins", "full-fft"}:
            raise ValueError("slave_mode must be auto, exact-bins or full-fft")
        if cfar_train < 4 or cfar_guard < 1:
            raise ValueError("cfar_train/guard too small")
        if not (0.0 < cfar_pfa < 1.0):
            raise ValueError("cfar_pfa must be in (0,1)")

        self.n_fft = int(n_fft)
        self.sample_rate = float(sample_rate)
        self.center_freq_hz = float(center_freq_hz)
        self.baseline_m = float(baseline_m)
        self.threshold_snr_db = float(threshold_snr_db)
        self.phase_calibration = math.radians(float(phase_calibration_deg))
        self.batch_frames = int(batch_frames)
        self.fft_workers = max(1, int(fft_workers))
        self.phase_bins = max(1, int(phase_bins) | 1)
        self.slave_mode = slave_mode
        self.cfar_train = int(cfar_train)
        self.cfar_guard = int(cfar_guard)
        self.cfar_pfa = float(cfar_pfa)
        self.phase_occupancy_db = float(phase_occupancy_db)
        self.sparse_max_bins = max(self.phase_bins, int(sparse_max_bins))
        self.coherence_min = float(coherence_min)
        self.peak_cluster_gap_bins = max(1, int(peak_cluster_gap_bins))

        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.master_td = np.empty((self.batch_frames, self.n_fft), dtype=np.complex64)
        self.slave_td = np.empty((self.batch_frames, self.n_fft), dtype=np.complex64)
        self.power = np.empty((self.batch_frames, self.n_fft), dtype=np.float32)
        self.master_fft: Optional[np.ndarray] = None
        self.slave_fft: Optional[np.ndarray] = None
        self.last_candidates: list[SpectralCandidate] = []
        self.last_slave_mode_by_frame: dict[int, str] = {}
        self.last_incoherent_candidates = 0
        self.last_coherence_min = 0.0

        # One compact N-point unit-circle root table shared by all sparse-bin DFTs.
        # This is O(N) memory (about 256 KiB for N=32768 in complex64) and completely
        # removes the pathological O(N*nbins) Python trig-table generation on cache misses.
        k = np.arange(self.n_fft, dtype=np.float64)
        self._roots = np.exp((-2j * np.pi / self.n_fft) * k).astype(np.complex64)
        self.neon = NeonKernels(neon_library) if use_neon else NeonKernels("")

    @property
    def raw_batch_bytes(self) -> int:
        return self.batch_frames * self.n_fft * 2

    @property
    def raw_pair_bytes(self) -> int:
        return self.raw_batch_bytes * 2

    @property
    def fft_bin_hz(self) -> float:
        return self.sample_rate / self.n_fft

    @property
    def baseline_unambiguous_at_center(self) -> bool:
        return self.baseline_m <= (C / self.center_freq_hz) / 2.0

    def preprocess_fast(self, raw_master, raw_slave) -> None:
        rm = np.frombuffer(raw_master, dtype=np.int8)[: self.raw_batch_bytes]
        rs = np.frombuffer(raw_slave, dtype=np.int8)[: self.raw_batch_bytes]
        if self.neon.available:
            self.neon.lib.iq_batch_to_c64_hann_split(
                self.neon._i8_ptr(rm), self.neon._i8_ptr(rs),
                self.neon._f32_ptr(self.master_td), self.neon._f32_ptr(self.slave_td),
                self.neon._f32_ptr(self.window), self.batch_frames, self.n_fft,
            )
            return

        m = rm.reshape(self.batch_frames, self.n_fft, 2)
        s = rs.reshape(self.batch_frames, self.n_fft, 2)
        self.master_td.real[...] = m[..., 0]
        self.master_td.imag[...] = m[..., 1]
        self.slave_td.real[...] = s[..., 0]
        self.slave_td.imag[...] = s[..., 1]
        self.master_td -= self.master_td.mean(axis=1, keepdims=True)
        self.slave_td -= self.slave_td.mean(axis=1, keepdims=True)
        self.master_td *= self.window[None, :]
        self.slave_td *= self.window[None, :]

    def fft_master_all(self) -> np.ndarray:
        self.master_fft = scipy.fft.fft(self.master_td, axis=1, workers=self.fft_workers, overwrite_x=True)
        return self.master_fft

    def fft_slave_all(self) -> np.ndarray:
        self.slave_fft = scipy.fft.fft(self.slave_td, axis=1, workers=self.fft_workers, overwrite_x=True)
        return self.slave_fft

    def _power_master(self) -> np.ndarray:
        if self.master_fft is None:
            raise RuntimeError("master FFT not computed")
        if self.neon.available:
            flat = np.ascontiguousarray(self.master_fft, dtype=np.complex64).view(np.float32)
            self.neon.lib.power_c64_f32(
                flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.power.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.batch_frames * self.n_fft,
            )
        else:
            self.power[...] = self.master_fft.real * self.master_fft.real + self.master_fft.imag * self.master_fft.imag
        return self.power

    def _cfar_noise(self, p: np.ndarray) -> np.ndarray:
        # Circular CA-CFAR. The moving-sum kernel is native C (and compiled together
        # with the NEON kernels), avoiding Python/scipy image-filter overhead.
        out = np.empty_like(p, dtype=np.float32)
        if self.neon.available and hasattr(self.neon.lib, "cfar_noise_circular_f32"):
            for b in range(self.batch_frames):
                self.neon.lib.cfar_noise_circular_f32(
                    self.neon._f32_ptr(np.ascontiguousarray(p[b])), self.n_fft,
                    self.cfar_train, self.cfar_guard, self.neon._f32_ptr(out[b])
                )
        else:
            # Portable fallback; only used off-target hardware.
            t = self.cfar_train; g = self.cfar_guard; n = self.n_fft
            for b in range(self.batch_frames):
                row = p[b]
                acc = np.zeros(n, dtype=np.float64)
                for d in range(g + 1, g + t + 1):
                    acc += np.roll(row, d) + np.roll(row, -d)
                out[b] = (acc / (2.0 * t)).astype(np.float32)
        global_med = np.median(p, axis=-1, keepdims=True)
        return np.maximum(out, global_med * 0.25 + 1e-20)

    def _best_window_energy(self, row: np.ndarray, noise: float) -> tuple[np.ndarray, np.ndarray]:
        ws = [4]
        w = 8
        while w <= max(4, self.n_fft // 2):
            ws.append(w); w *= 2
        widths = np.asarray(sorted(set(min(x, self.n_fft) for x in ws)), dtype=np.uintp)
        best_ratio = np.zeros(self.n_fft, dtype=np.float32)
        best_width = np.ones(self.n_fft, dtype=np.uintp)
        if self.neon.available and hasattr(self.neon.lib, "windowed_ratio_best_f32"):
            self.neon.lib.windowed_ratio_best_f32(
                self.neon._f32_ptr(np.ascontiguousarray(row)), self.n_fft, float(noise),
                widths.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)), len(widths),
                self.neon._f32_ptr(best_ratio), best_width.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
            )
        else:
            for w in widths.tolist():
                kernel = np.ones(int(w), dtype=np.float32) / float(w)
                avg = np.convolve(np.r_[row[-w//2:], row, row[:w//2]], kernel, mode="valid")[:self.n_fft]
                ratio = avg / max(noise, 1e-20)
                mask = ratio > best_ratio
                best_ratio[mask] = ratio[mask].astype(np.float32)
                best_width[mask] = w
        return best_ratio, best_width

    @staticmethod
    def _wideband_threshold_ratio(width: int, z: float = 4.5) -> float:
        # Conservative gamma-tail approximation for the sum of width exponential noise bins.
        return max(1.05, 1.0 + z / math.sqrt(max(1, width)))

    def _component_ranges(self, mask_1d: np.ndarray) -> list[tuple[int, int]]:
        n = mask_1d.size
        idx = np.flatnonzero(mask_1d)
        if idx.size == 0:
            return []
        breaks = np.flatnonzero(np.diff(idx) > 1)
        groups = np.split(idx, breaks + 1)
        ranges = [(int(g[0]), int(g[-1])) for g in groups if g.size]
        if len(ranges) > 1 and ranges[0][0] == 0 and ranges[-1][1] == n - 1:
            ranges = [(ranges[-1][0], ranges[0][1])] + ranges[1:-1]
        return ranges

    def _expand_band(self, p: np.ndarray, noise: np.ndarray, peak_idx: int) -> tuple[int, int]:
        n = p.size
        thr = noise * (10.0 ** (self.phase_occupancy_db / 10.0))
        if p[peak_idx] <= thr[peak_idx]:
            return peak_idx, peak_idx

        lo = peak_idx
        hi = peak_idx
        # Bounded circular expansion. The bound prevents an extremely strong carrier from
        # swallowing the whole Nyquist interval.
        max_width = max(1, n // 2 - 1)
        for _ in range(max_width):
            nxt = (lo - 1) % n
            if p[nxt] <= thr[nxt]:
                break
            lo = nxt
        for _ in range(max_width):
            nxt = (hi + 1) % n
            if p[nxt] <= thr[nxt]:
                break
            hi = nxt
        return lo, hi

    def detect_candidates(self) -> list[SpectralCandidate]:
        p = self._power_master().copy()
        if self.n_fft >= 32:
            z = max(4, self.n_fft // 4096)
            p[:, :z] = 0.0; p[:, -z:] = 0.0
        noise = self._cfar_noise(p)
        n_train = 2 * self.cfar_train
        alpha = n_train * (self.cfar_pfa ** (-1.0 / n_train) - 1.0)
        alpha = max(alpha, 10.0 ** (self.threshold_snr_db / 10.0))
        candidates: list[SpectralCandidate] = []
        self.last_slave_mode_by_frame.clear()

        for b in range(self.batch_frames):
            fb = p[b]; fn = noise[b]
            thr = fn * alpha
            local_max = (fb >= np.roll(fb, 1)) & (fb > np.roll(fb, -1)) & (fb > thr)
            peaks = np.flatnonzero(local_max)
            kept: list[int] = []
            if peaks.size:
                order = peaks[np.argsort(fb[peaks])[::-1]]
                # Suppression distance protects against many FFT-bin maxima from one emitter.
                suppression = max(2, min(32, int(round(2000.0 / self.fft_bin_hz))))
                for pk in order:
                    if all(min((int(pk)-q) % self.n_fft, (q-int(pk)) % self.n_fft) > suppression for q in kept):
                        kept.append(int(pk))

            for group in self._cluster_peaks(np.asarray(kept, dtype=np.int64)):
                # One coherent wideband emitter can have hundreds of above-CFAR local
                # maxima (OFDM, DSSS, chirps). Treat a dense peak cluster as one candidate.
                pk = max(group, key=lambda q: fb[q])
                if len(group) >= 3 and (group[-1] - group[0] + 1) > self.peak_cluster_gap_bins:
                    lo = group[0]; hi = group[-1]
                    # Expand by the measured low-level occupancy threshold around the cluster.
                    while (pk-lo) < self.n_fft//2:
                        nxt=(lo-1)%self.n_fft
                        if fb[nxt] <= fn[nxt]*(10.0**(self.phase_occupancy_db/10.0)): break
                        lo=nxt
                    while (hi-pk) < self.n_fft//2:
                        nxt=(hi+1)%self.n_fft
                        if fb[nxt] <= fn[nxt]*(10.0**(self.phase_occupancy_db/10.0)): break
                        hi=nxt
                else:
                    lo, hi = self._expand_band(fb, fn, pk)
                band = self._band_indices(int(lo), int(hi))
                if band.size == 0: band = np.asarray([pk], dtype=np.int64)
                weights = fb[band]
                signed = self._signed_fft_offsets(band)
                eff_offset = float(np.sum(weights*signed)/(np.sum(weights)+1e-20))*self.fft_bin_hz
                candidates.append(SpectralCandidate(
                    frame=b, peak_idx=pk, start_idx=int(lo), end_idx=int(hi),
                    snr_db=float(10.0*np.log10((fb[pk]+1e-20)/(fn[pk]+1e-20))),
                    effective_freq_hz=self.center_freq_hz+eff_offset, occupied_bins=int(band.size)))

            # Multiscale energy detector is mandatory for wideband signals: a 12 dB per-bin
            # threshold is not a valid detector for OFDM/DSSS when energy is distributed over hundreds of bins.
            global_noise = float(np.median(fb))
            # Remove already-detected narrow emitters from the wideband energy search;
            # otherwise a powerful CW tone can dominate every large integration window
            # and hide a weaker wideband emitter elsewhere in the spectrum.
            wide_row = fb.copy()
            for c in candidates:
                if c.frame != b: continue
                wide_row[self._band_indices(c.start_idx, c.end_idx)] = global_noise
            ratios, widths = self._best_window_energy(wide_row, global_noise + 1e-20)
            wide_score = (ratios - 1.0) * np.sqrt(np.maximum(widths.astype(np.float32), 1.0))
            wide_peaks = np.flatnonzero((wide_score >= np.roll(wide_score,1)) & (wide_score > np.roll(wide_score,-1)))
            wide_peaks = wide_peaks[np.argsort(wide_score[wide_peaks])[::-1]] if wide_peaks.size else wide_peaks
            wide_min_bins = max(32, 2 * self.sparse_max_bins)
            for pk in wide_peaks[:32]:
                w = int(widths[pk])
                if w < wide_min_bins or wide_score[pk] < 4.5 or ratios[pk] <= 1.0:
                    continue
                # Do not duplicate a stronger narrowband candidate already covering this center.
                duplicate = False
                for c in candidates:
                    if c.frame != b: continue
                    dist = min((int(pk)-c.peak_idx)%self.n_fft, (c.peak_idx-int(pk))%self.n_fft)
                    if dist <= max(w//2, c.occupied_bins//2, 4):
                        duplicate = True; break
                if duplicate: continue
                lo = int(pk - w//2) % self.n_fft
                hi = int(lo + w - 1) % self.n_fft
                band = self._band_indices(lo, hi)
                # Wideband qualification: a real occupied band should contain a dense
                # population of elevated bins. This prevents a single strong CW tone
                # from being represented as an artificially wide integration window.
                occ_frac = float(np.mean(fb[band] > global_noise * (10.0 ** (self.phase_occupancy_db / 10.0))))
                if occ_frac < 0.65:
                    continue
                weights = fb[band]
                signed = self._signed_fft_offsets(band)
                eff_offset = float(np.sum(weights*signed)/(np.sum(weights)+1e-20))*self.fft_bin_hz
                candidates.append(SpectralCandidate(
                    frame=b, peak_idx=int(pk), start_idx=lo, end_idx=hi,
                    snr_db=float(10.0*np.log10(max(ratios[pk],1e-20))),
                    effective_freq_hz=self.center_freq_hz+eff_offset, occupied_bins=int(band.size), metric="band-energy"))

        # Overlap-aware merge: a dense OFDM/DSSS band must not turn into dozens of
        # separate phase estimates. Prefer the widest coherent candidate for overlapping bands.
        merged: list[SpectralCandidate] = []
        for frame in range(self.batch_frames):
            fc = [c for c in candidates if c.frame == frame]
            fc.sort(key=lambda c:(-c.occupied_bins, -c.snr_db))
            for c in fc:
                bset = self._band_indices(c.start_idx, c.end_idx)
                keep = True
                for e in merged:
                    if e.frame != frame: continue
                    eset = self._band_indices(e.start_idx, e.end_idx)
                    overlap = np.intersect1d(bset, eset, assume_unique=False).size
                    if overlap / max(1, min(bset.size, eset.size)) >= 0.40:
                        keep = False; break
                if keep: merged.append(c)
        merged.sort(key=lambda c:(c.frame,-c.snr_db)); self.last_candidates=merged
        return merged

    def _cluster_peaks(self, peaks: np.ndarray) -> list[list[int]]:
        if peaks.size == 0:
            return []
        vals = sorted(int(x) for x in peaks)
        groups: list[list[int]] = [[vals[0]]]
        gap = self.peak_cluster_gap_bins
        for v in vals[1:]:
            if v - groups[-1][-1] <= gap:
                groups[-1].append(v)
            else:
                groups.append([v])
        # Join FFT-edge groups because the spectrum is circular.
        if len(groups) > 1 and (groups[0][0] == 0 or self.n_fft - 1 - groups[-1][-1] < gap):
            if self.n_fft - groups[-1][-1] + groups[0][0] <= gap:
                groups[0] = groups[-1] + groups[0]
                groups.pop()
        return groups

    def _signed_fft_offsets(self, bins: np.ndarray) -> np.ndarray:
        d = ((bins.astype(np.int64) + self.n_fft // 2) % self.n_fft) - self.n_fft // 2
        return d.astype(np.float64)

    def selected_bins(self, peak_idx: int) -> np.ndarray:
        half = self.phase_bins // 2
        offsets = np.arange(-half, half + 1, dtype=np.int64)
        return (int(peak_idx) + offsets) % self.n_fft

    def _signed_bin_offsets(self, bins: np.ndarray, center: int) -> np.ndarray:
        d = ((bins.astype(np.int64) - int(center) + self.n_fft // 2) % self.n_fft) - self.n_fft // 2
        return d.astype(np.float64)

    def _band_indices(self, start: int, end: int) -> np.ndarray:
        if start <= end:
            return np.arange(start, end + 1, dtype=np.int64)
        return np.concatenate((np.arange(start, self.n_fft, dtype=np.int64), np.arange(0, end + 1, dtype=np.int64)))

    def _exact_bins(self, frame_idx: int, bins: np.ndarray) -> np.ndarray:
        """Exact DFT coefficients for selected FFT bins without per-bin trig tables.

        Each coefficient uses the same roots as the full FFT convention. The native
        kernel walks the precomputed N-point unit-circle table with modular phase
        stepping, so a changing/FHSS bin set has no cache-miss trig-generation penalty.
        """
        bins_arr = np.ascontiguousarray(np.asarray(bins, dtype=np.uintp))
        out = np.empty(bins_arr.size, dtype=np.complex64)
        x = self.slave_td[frame_idx]
        if bins_arr.size == 0:
            return out
        if self.neon.available and hasattr(self.neon.lib, 'dft_bins_c64_root'):
            self.neon.lib.dft_bins_c64_root(
                self.neon._f32_ptr(np.ascontiguousarray(x).view(np.float32)),
                self.n_fft,
                self.neon._f32_ptr(self._roots.view(np.float32)),
                bins_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
                bins_arr.size,
                self.neon._f32_ptr(out.view(np.float32)),
            )
            return out
        # Portable fallback. It intentionally uses the same root table, so results
        # remain convention-identical to the native path.
        xx = np.ascontiguousarray(x)
        for j, b in enumerate(bins_arr.tolist()):
            idx = (int(b) * np.arange(self.n_fft, dtype=np.int64)) % self.n_fft
            out[j] = np.dot(xx, self._roots[idx])
        return out

    def _cross_range_neon_safe(self, xm: np.ndarray, xs: np.ndarray, bins: np.ndarray) -> tuple[float, float, float, float]:
        if bins.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        c_re = c_im = pm = ps = 0.0
        ranges: list[tuple[int, int]] = []
        st = int(bins[0]); prev = st
        for val in bins[1:]:
            val = int(val)
            if val != prev + 1:
                ranges.append((st, prev)); st = val
            prev = val
        ranges.append((st, prev))
        if self.neon.available:
            for lo, hi in ranges:
                mm = np.ascontiguousarray(xm[lo:hi + 1], dtype=np.complex64)
                ss = np.ascontiguousarray(xs[lo:hi + 1], dtype=np.complex64)
                rr = np.zeros(1, dtype=np.float32)
                ii = np.zeros(1, dtype=np.float32)
                pp_m = np.zeros(1, dtype=np.float32)
                pp_s = np.zeros(1, dtype=np.float32)
                self.neon.lib.cross_sum_range_f32(
                    self.neon._f32_ptr(mm.view(np.float32)), self.neon._f32_ptr(ss.view(np.float32)),
                    hi - lo + 1,
                    self.neon._f32_ptr(rr), self.neon._f32_ptr(ii),
                    self.neon._f32_ptr(pp_m), self.neon._f32_ptr(pp_s),
                )
                c_re += float(rr[0]); c_im += float(ii[0]); pm += float(pp_m[0]); ps += float(pp_s[0])
            return c_re, c_im, pm, ps
        for lo, hi in ranges:
            mm = xm[lo:hi + 1]
            ss = xs[lo:hi + 1]
            cross = np.vdot(ss, mm)
            c_re += float(cross.real); c_im += float(cross.imag)
            pm += float(np.vdot(mm, mm).real); ps += float(np.vdot(ss, ss).real)
        return c_re, c_im, pm, ps

    def _full_slave_required(self, candidates_by_frame: dict[int, list[SpectralCandidate]], mode: str) -> bool:
        if mode == "full-fft":
            return True
        if mode == "exact-bins":
            return False
        for cands in candidates_by_frame.values():
            total = sum(c.occupied_bins for c in cands)
            if total > self.sparse_max_bins or any(c.occupied_bins > self.sparse_max_bins for c in cands):
                return True
        return False

    def detect_and_localize(self) -> list[Detection]:
        self.fft_master_all()
        self.last_incoherent_candidates = 0
        self.last_coherence_min = 1.0
        candidates = self.detect_candidates()
        if not candidates:
            self.slave_fft = None
            return []

        by_frame: dict[int, list[SpectralCandidate]] = {}
        for c in candidates:
            by_frame.setdefault(c.frame, []).append(c)

        frame_mode_by_id: dict[int, str] = {}
        for frame_idx, cands in by_frame.items():
            unsafe_sparse = (
                sum(c.occupied_bins for c in cands) > self.sparse_max_bins
                or any(c.occupied_bins > self.sparse_max_bins for c in cands)
            )
            # `exact-bins` is never allowed to become an accuracy regression: when
            # the measured occupied band exceeds the sparse budget, the implementation
            # transparently falls back to a full Slave FFT.
            use_full = self.slave_mode == "full-fft" or unsafe_sparse
            frame_mode_by_id[frame_idx] = "full-fft" if use_full else "exact-bins"

        full_frames = [f for f, m in frame_mode_by_id.items() if m == "full-fft"]
        sparse_frames = [f for f, m in frame_mode_by_id.items() if m == "exact-bins"]
        self.slave_fft = None
        full_fft_map: dict[int, np.ndarray] = {}
        if self.slave_mode == "full-fft":
            Xs = self.fft_slave_all()
            full_fft_map = {f: Xs[f] for f in full_frames}
        elif full_frames:
            if len(full_frames) == self.batch_frames:
                Xs = self.fft_slave_all(); full_fft_map = {f: Xs[f] for f in full_frames}
            elif len(full_frames) > 1:
                sub = scipy.fft.fft(self.slave_td[full_frames], axis=1, workers=self.fft_workers, overwrite_x=True)
                full_fft_map = {f: sub[i] for i, f in enumerate(full_frames)}
            else:
                f = full_frames[0]
                full_fft_map[f] = scipy.fft.fft(self.slave_td[f], workers=self.fft_workers, overwrite_x=True)

        detections: list[Detection] = []
        self.last_slave_mode_by_frame = dict(frame_mode_by_id)
        for frame_idx, cands in by_frame.items():
            frame_mode = frame_mode_by_id[frame_idx]
            self.last_slave_mode_by_frame[frame_idx] = frame_mode
            for c in cands:
                bins = self._band_indices(c.start_idx, c.end_idx)
                xm = self.master_fft[frame_idx]
                if frame_mode == "full-fft":
                    xs = full_fft_map[frame_idx]
                    cr, ci, pm, ps = self._cross_range_neon_safe(xm, xs, bins)
                else:
                    xs_sparse = self._exact_bins(frame_idx, bins)
                    xm_sparse = xm[bins]
                    cross = np.vdot(xs_sparse, xm_sparse)
                    cr = float(cross.real); ci = float(cross.imag)
                    pm = float(np.vdot(xm_sparse, xm_sparse).real)
                    ps = float(np.vdot(xs_sparse, xs_sparse).real)

                cross_mag = math.hypot(cr, ci)
                coherence = cross_mag / math.sqrt(max(pm * ps, 1e-30))
                self.last_coherence_min = min(self.last_coherence_min, float(coherence))
                if coherence < self.coherence_min:
                    self.last_incoherent_candidates += 1
                    continue

                d_phi = math.atan2(ci, cr) + self.phase_calibration
                d_phi = float(wrap_phase(d_phi))
                freq = float(c.effective_freq_hz)
                wavelength = C / max(1.0, freq)
                sin_t = d_phi * wavelength / (2.0 * np.pi * self.baseline_m)
                angle_deg = float(np.degrees(np.arcsin(np.clip(sin_t, -1.0, 1.0))) )
                unambiguous = self.baseline_m <= wavelength / 2.0
                detections.append(
                    Detection(
                        frame=frame_idx,
                        peak_idx=c.peak_idx,
                        start_idx=c.start_idx,
                        end_idx=c.end_idx,
                        freq_offset_hz=freq - self.center_freq_hz,
                        target_freq_hz=freq,
                        snr_db=c.snr_db,
                        phase_deg=math.degrees(d_phi),
                        angle_deg=angle_deg,
                        coherence=float(coherence),
                        occupied_bins=c.occupied_bins,
                        slave_compute_mode=frame_mode,
                        detection_metric=c.metric,
                        baseline_unambiguous=unambiguous,
                    )
                )
        return detections


class KinematicDOAResolver:
    """Model-based rotating-baseline ambiguity resolver.

    It does not rely on heuristic phase-derivative continuity. For a static emitter and
    known platform heading psi(t), it maximizes the circular phase likelihood

        phi(t) = 2*pi*f*d/c * sin(theta_world - psi(t)) + phi_cal.

    Thus spatial aliases are selected by global model consistency across the rotation arc.
    """

    def __init__(self, baseline_m: float, min_samples: int = 8, min_span_deg: float = 25.0,
                 grid_step_deg: float = 0.25, history: int = 64):
        self.baseline_m = float(baseline_m)
        self.min_samples = int(min_samples)
        self.min_span_deg = float(min_span_deg)
        self.grid_step_deg = float(grid_step_deg)
        self._obs: Deque[tuple[float, float, float, float]] = deque(maxlen=int(history))

    def add(self, phase_deg: float, target_freq_hz: float, platform_angle_deg: float, weight: float = 1.0) -> KinematicEstimate | None:
        self._obs.append((math.radians(phase_deg), float(target_freq_hz), math.radians(platform_angle_deg), max(1e-3, float(weight))))
        if len(self._obs) < self.min_samples:
            return None
        psi = np.asarray([x[2] for x in self._obs], dtype=np.float64)
        # unwrap only the encoder angle to measure physical rotation span
        psi_u = np.unwrap(psi)
        span = math.degrees(float(psi_u.max() - psi_u.min()))
        if span < self.min_span_deg:
            return None

        phi = np.asarray([x[0] for x in self._obs], dtype=np.float64)
        freq = np.asarray([x[1] for x in self._obs], dtype=np.float64)
        weights = np.asarray([x[3] for x in self._obs], dtype=np.float64)
        theta_grid = np.deg2rad(np.arange(-180.0, 180.0, self.grid_step_deg, dtype=np.float64))
        k = 2.0 * np.pi * freq * self.baseline_m / C
        pred = k[:, None] * np.sin(theta_grid[None, :] - psi[:, None])
        residual = wrap_phase(phi[:, None] - pred)
        score = (weights[:, None] * np.cos(residual)).sum(axis=0) / weights.sum()
        best = int(np.argmax(score))
        # Parabolic refinement on the periodic score grid.
        left = score[(best - 1) % score.size]
        mid = score[best]
        right = score[(best + 1) % score.size]
        denom = left - 2.0 * mid + right
        frac = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
        refined = theta_grid[best] + frac * math.radians(self.grid_step_deg)
        refined_deg = ((math.degrees(refined) + 180.0) % 360.0) - 180.0
        return KinematicEstimate(float(refined_deg), float(mid), len(self._obs), span)


class KinematicTrackerBank:
    def __init__(self, baseline_m: float, freq_tolerance_hz: float = 250_000.0):
        self.baseline_m = float(baseline_m)
        self.freq_tol = float(freq_tolerance_hz)
        self.tracks: list[tuple[float, KinematicDOAResolver, float]] = []

    def update(self, det: Detection, platform_angle_deg: float, now_s: float) -> KinematicEstimate | None:
        best_i = -1; best_df = float("inf")
        for i, (freq, _, last_s) in enumerate(self.tracks):
            df = abs(freq - det.target_freq_hz)
            if df < best_df and df <= self.freq_tol:
                best_i = i; best_df = df
        if best_i < 0:
            tr = KinematicDOAResolver(self.baseline_m)
            self.tracks.append((det.target_freq_hz, tr, now_s))
            best_i = len(self.tracks) - 1
        freq, tr, _ = self.tracks[best_i]
        est = tr.add(det.phase_deg, det.target_freq_hz, platform_angle_deg, weight=max(1.0, 10.0 ** (0.05 * det.snr_db)))
        self.tracks[best_i] = (0.8 * freq + 0.2 * det.target_freq_hz, tr, now_s)
        self.tracks = [x for x in self.tracks if now_s - x[2] <= 10.0]
        return est


def set_process_affinity(cpus: Optional[Iterable[int]]) -> None:
    if cpus is None or not hasattr(os, "sched_setaffinity"):
        return
    mask = {int(c) for c in cpus}
    if mask:
        os.sched_setaffinity(0, mask)


def choose_default_fft_workers() -> int:
    n = os.cpu_count() or 1
    if platform.machine().lower() in {"aarch64", "arm64"}:
        return min(3, n)
    return min(4, n)
