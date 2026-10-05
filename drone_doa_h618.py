#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Full, Queue

import numpy as np

from doa_h618_core import SDRCore, KinematicTrackerBank, choose_default_fft_workers, set_process_affinity, NeonKernels
from encoder_source import EncoderSource, configure_encoder_low_latency

SERIAL_MASTER_DEFAULT = "0000000000000000010961dc2976b74f"
SERIAL_SLAVE_DEFAULT = "0000000000000000010961dc2a24574f"
SPEED_OF_LIGHT = 299_792_458.0
HACKRF_MIN_FREQ_HZ = 1_000_000.0
HACKRF_MAX_FREQ_HZ = 6_000_000_000.0

proc_master = None
proc_slave = None
reader = None
encoder = None
stop_event = threading.Event()


@dataclass(slots=True)
class ReadyFrame:
    frame_id: int
    slot: int
    ready_ns: int


def cleanup_and_exit(signum=None, frame=None):
    stop_event.set()
    if encoder is not None:
        encoder.stop()
    if reader is not None:
        reader.stop()
    for p in (proc_master, proc_slave):
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    for p in (proc_master, proc_slave):
        if p is not None:
            try:
                p.wait(timeout=1.0)
            except Exception:
                pass
    raise SystemExit(0)


signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)


def enable_master_clock(serial: str) -> None:
    print("[*] Активация 10 МГц reference на Master...")
    try:
        subprocess.run(["hackrf_clock", "-d", serial, "-o", "1"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[+] Master CLKOUT включен")
    except Exception as exc:
        print(f"[-] Предупреждение по clock: {exc}")


def start_streaming(serial_m: str, serial_s: str, freq_hz: float, sample_rate: int, lna: int, vga: int):
    global proc_master, proc_slave
    cmd_base = ["hackrf_transfer", "-r", "-", "-f", str(int(freq_hz)), "-s", str(sample_rate), "-l", str(lna), "-g", str(vga)]
    proc_master = subprocess.Popen(cmd_base[:1] + ["-d", serial_m] + cmd_base[1:],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    proc_slave = subprocess.Popen(cmd_base[:1] + ["-d", serial_s] + cmd_base[1:],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    time.sleep(0.5)


def stop_streaming() -> None:
    global proc_master, proc_slave
    for p in (proc_master, proc_slave):
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
    for p in (proc_master, proc_slave):
        if p is not None:
            try:
                p.wait(timeout=1.0)
            except Exception:
                pass
    proc_master = proc_slave = None


def read_exact_fd(fd: int, buf: bytearray) -> int:
    mv = memoryview(buf)
    off = 0
    while off < len(buf) and not stop_event.is_set():
        n = os.readv(fd, [mv[off:]])
        if n == 0:
            return off
        off += n
    return off


def discard_exact_fd(fd: int, count: int) -> None:
    scratch = bytearray(min(256 * 1024, max(4096, count)))
    left = count
    while left > 0 and not stop_event.is_set():
        n = os.readv(fd, [memoryview(scratch)[:min(left, len(scratch))]])
        if n == 0:
            raise RuntimeError("EOF while discarding synchronization offset")
        left -= n


class FixedBatchReader:
    """Two independent producers with sequence-numbered reusable buffers.

    Queue order is not trusted as synchronization. Each complete batch has an explicit
    frame_id. If one side gets ahead, old frames are released until IDs match. A larger
    gap triggers a hard resynchronization request instead of feeding mismatched data to DSP.
    """

    def __init__(self, fd_master: int, fd_slave: int, chunk_bytes: int, slots: int = 6, max_gap: int = 2, hard_gap: int = 64):
        self.fd_master = fd_master
        self.fd_slave = fd_slave
        self.chunk_bytes = int(chunk_bytes)
        self.slots = max(3, int(slots))
        # max_gap is an observability threshold, not a reason to restart hardware.
        # Normal scheduling/USB queue skew is repaired by dropping only stale leading
        # batches.  hard_gap is reserved for a genuinely pathological stream state.
        self.max_gap = max(1, int(max_gap))
        self.hard_gap = max(self.max_gap + 1, int(hard_gap))
        self.master_bufs = [bytearray(chunk_bytes) for _ in range(self.slots)]
        self.slave_bufs = [bytearray(chunk_bytes) for _ in range(self.slots)]
        self.master_free: Queue[int] = Queue(self.slots)
        self.slave_free: Queue[int] = Queue(self.slots)
        self.master_ready: Queue[ReadyFrame] = Queue(self.slots)
        self.slave_ready: Queue[ReadyFrame] = Queue(self.slots)
        for i in range(self.slots):
            self.master_free.put_nowait(i); self.slave_free.put_nowait(i)
        self.threads: list[threading.Thread] = []
        self.error: BaseException | None = None
        self._pending_m: ReadyFrame | None = None
        self._pending_s: ReadyFrame | None = None
        self.resync_count = 0
        self.soft_gap_events = 0
        self.dropped_frame_pairs = 0

    def _producer(self, fd: int, bufs, free_q: Queue[int], ready_q: Queue[ReadyFrame], name: str):
        seq = 0
        try:
            while not stop_event.is_set():
                idx = free_q.get()
                n = read_exact_fd(fd, bufs[idx])
                if n != self.chunk_bytes:
                    self.error = RuntimeError(f"{name}: EOF/short stream {n}/{self.chunk_bytes}")
                    stop_event.set()
                    return
                ready_q.put(ReadyFrame(seq, idx, time.monotonic_ns()))
                seq += 1
        except BaseException as exc:
            self.error = exc
            stop_event.set()

    def start(self):
        self.threads = [
            threading.Thread(target=self._producer, args=(self.fd_master, self.master_bufs, self.master_free, self.master_ready, "master"), daemon=True),
            threading.Thread(target=self._producer, args=(self.fd_slave, self.slave_bufs, self.slave_free, self.slave_ready, "slave"), daemon=True),
        ]
        for t in self.threads: t.start()

    def _get_m(self):
        if self._pending_m is None:
            self._pending_m = self.master_ready.get(timeout=0.5)
        return self._pending_m

    def _get_s(self):
        if self._pending_s is None:
            self._pending_s = self.slave_ready.get(timeout=0.5)
        return self._pending_s

    def _drop_m(self):
        f = self._pending_m; self._pending_m = None
        if f is not None:
            self.master_free.put(f.slot); self.dropped_frame_pairs += 1

    def _drop_s(self):
        f = self._pending_s; self._pending_s = None
        if f is not None:
            self.slave_free.put(f.slot); self.dropped_frame_pairs += 1

    def _flush_queue(self, q: Queue[ReadyFrame], free_q: Queue[int]):
        while True:
            try:
                f = q.get_nowait()
            except Empty:
                return
            free_q.put(f.slot)

    def resync(self):
        """Soft queue resynchronization; hardware producers remain alive."""
        self._drop_m(); self._drop_s()
        self._flush_queue(self.master_ready, self.master_free)
        self._flush_queue(self.slave_ready, self.slave_free)
        self.resync_count += 1

    def get_pair(self):
        while not stop_event.is_set():
            try:
                m = self._get_m(); s = self._get_s()
            except Empty:
                if self.error: raise self.error
                continue
            if m.frame_id == s.frame_id:
                self._pending_m = self._pending_s = None
                ready_ns = (m.ready_ns + s.ready_ns) // 2
                return m.frame_id, m.slot, s.slot, self.master_bufs[m.slot], self.slave_bufs[s.slot], ready_ns

            gap = m.frame_id - s.frame_id
            # The queues are timestamp-equivalent streams; the only safe repair is
            # to discard batches from the side that is ahead until the frame IDs meet.
            # Do NOT flush both queues: doing so preserves the sequence skew and can
            # produce a self-sustaining resync loop.
            if abs(gap) > self.max_gap:
                self.soft_gap_events += 1
            if abs(gap) > self.hard_gap:
                raise RuntimeError(
                    f"pathological stream divergence: master={m.frame_id}, slave={s.frame_id}"
                )
            if gap < 0:
                self._drop_m()
            else:
                self._drop_s()
        raise RuntimeError("reader stopped")

    def release_pair(self, im: int, is_: int):
        self.master_free.put(im); self.slave_free.put(is_)

    def stop(self):
        stop_event.set()

    def join(self, timeout: float = 1.0):
        for t in self.threads:
            t.join(timeout=timeout)


def _normalized_corr_peak(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float]:
    from scipy.signal import correlate
    a = np.asarray(a, dtype=np.complex64); b = np.asarray(b, dtype=np.complex64)
    a = a - np.mean(a); b = b - np.mean(b)
    na = np.linalg.norm(a) + 1e-20; nb = np.linalg.norm(b) + 1e-20
    c = correlate(a / na, b / nb, mode="full", method="fft")
    mag = np.abs(c)
    best = int(np.argmax(mag))
    # lag convention: c index = len(b)-1 => zero lag.
    lag = best - (len(b) - 1)
    # Do not compare the peak against its own immediate correlation sidelobes:
    # adjacent lags can be almost as large for a long coherent burst. Suppress a
    # small neighborhood and report the strongest *independent* peak instead.
    exclude = max(8, min(64, len(b) // 64))
    alt = mag.copy()
    alt[max(0, best - exclude):min(len(alt), best + exclude + 1)] = 0.0
    second = float(np.max(alt)) if alt.size else 0.0
    return int(lag), float(mag[best]), second


def calibrate_alignment(proc_m, proc_s, sample_rate: float, timeout_s: float = 0.0) -> int:
    print("\n" + "=" * 72)
    print("  КАЛИБРОВКА СИНХРОННОГО SAMPLE-TIMELINE: ждём общий сигнал")
    print("  Слабый/отсутствующий эфир НЕ завершает программу; калибровка повторяется.")
    print("=" * 72)

    n = 131072
    decim = 4
    snippet_len = 12288
    max_wait = None if timeout_s <= 0 else time.monotonic() + timeout_s
    while not stop_event.is_set():
        if max_wait is not None and time.monotonic() > max_wait:
            raise TimeoutError("calibration timeout")
        bm = bytearray(n * 2); bs = bytearray(n * 2)
        errors: list[BaseException] = []
        def rr(fd, buf):
            try:
                if read_exact_fd(fd, buf) != len(buf):
                    raise RuntimeError("calibration EOF/short read")
            except BaseException as exc:
                errors.append(exc)
        tm = threading.Thread(target=rr, args=(proc_m.stdout.fileno(), bm), daemon=True)
        ts = threading.Thread(target=rr, args=(proc_s.stdout.fileno(), bs), daemon=True)
        tm.start(); ts.start(); tm.join(); ts.join()
        if errors: raise errors[0]

        am = np.frombuffer(bm, dtype=np.int8); ass = np.frombuffer(bs, dtype=np.int8)
        xm = am[0::2].astype(np.float32) + 1j * am[1::2].astype(np.float32)
        xs = ass[0::2].astype(np.float32) + 1j * ass[1::2].astype(np.float32)
        xm -= xm.mean(); xs -= xs.mean()
        rms_m = float(np.sqrt(np.mean(np.abs(xm) ** 2))); rms_s = float(np.sqrt(np.mean(np.abs(xs) ** 2)))
        if max(rms_m, rms_s) < 3.0:
            print(f"[*] калибровочный сигнал слабый: RMS M={rms_m:.2f} S={rms_s:.2f}; ждём...", end="\r")
            continue

        md = xm[::decim]; sd = xs[::decim]
        # Pick an energetic master snippet. This is robust to arbitrary signal placement in the window.
        env = np.convolve(np.abs(md) ** 2, np.ones(256, dtype=np.float32), mode="same")
        start = int(np.argmax(env)) - snippet_len // (2 * decim)
        start = max(0, min(len(md) - snippet_len // decim - 1, start))
        snip = md[start:start + snippet_len // decim]
        lag_d, peak, second = _normalized_corr_peak(sd, snip)
        margin_db = 20.0 * np.log10(max(peak, 1e-12) / max(second, 1e-12))
        # `peak` is a normalized correlation coefficient in [0,1]. The old
        # peak/median(amplitude) ratio mixed incompatible normalizations and could
        # reject a perfectly good coherent calibration burst. Use the coefficient
        # itself plus a sidelobe margin instead.
        if peak < 0.20 or margin_db < 3.0:
            print(f"[*] неоднозначный calibration peak: corr={peak:.3f}, margin={margin_db:.1f} dB; повтор...", end="\r")
            continue

        # `_normalized_corr_peak(sd, snip)` returns the absolute location of the
        # snippet inside the decimated slave array. The physical inter-channel
        # displacement is therefore relative to `start`, not relative to sample 0.
        # With the decimation restored, delta is directly the number of raw samples
        # by which Slave is displaced with respect to the Master reference.
        delta_coarse = (lag_d - start) * decim

        # Refine around the RELATIVE lag while keeping the Master reference fixed at
        # its original absolute position. This avoids the old 2*start addressing bug.
        center = max(0, start * decim)
        ref = xm[center: center + min(16384, n - center)]
        best = None
        ref_n = np.linalg.norm(ref) + 1e-20
        for delta_try in range(delta_coarse - 8, delta_coarse + 9):
            ss = center + delta_try
            se = ss + len(ref)
            if ss < 0 or se > len(xs):
                continue
            val = abs(np.vdot(xs[ss:se], ref)) / (np.linalg.norm(xs[ss:se]) * ref_n + 1e-20)
            if best is None or val > best[1]:
                best = (delta_try, float(val))
        if best is None or best[1] < 0.12:
            continue
        delta = int(best[0])
        print(f"\n[+] alignment: lag={delta:+d} samples, corr={best[1]:.3f}, coarse={delta_coarse:+d}")
        # Positive delta means the matching event occurs later in Slave; discarding
        # the corresponding Slave prefix advances it onto the Master timeline.
        if delta > 0:
            discard_exact_fd(proc_s.stdout.fileno(), delta * 2)
            print(f"[+] сброшено со Slave: {delta} samples")
        elif delta < 0:
            discard_exact_fd(proc_m.stdout.fileno(), (-delta) * 2)
            print(f"[+] сброшено с Master: {-delta} samples")
        else:
            print("[+] sample offset = 0")
        return delta
    raise RuntimeError("stopped during calibration")


def draw_radar(angle: float) -> str:
    width = 31
    norm = int((angle + 90.0) / 180.0 * (width - 1))
    norm = max(0, min(width - 1, norm))
    bar = ["-"] * width; bar[width // 2] = "|"; bar[norm] = "▲"
    return "[" + "".join(bar) + "]"


def validate_frequency(f: float):
    if not (HACKRF_MIN_FREQ_HZ <= f <= HACKRF_MAX_FREQ_HZ):
        raise ValueError("HackRF center frequency must be within 1 MHz .. 6 GHz")



@dataclass(slots=True)
class GainChangeRequest(Exception):
    lna: int
    vga: int
    reason: str


def _clipping_stats(raw: bytes | bytearray | memoryview) -> float:
    """Near-rail fraction only; the hot path is native NEON when available."""
    a = np.frombuffer(raw, dtype=np.int8)
    if a.size == 0:
        return 0.0
    global _gain_neon
    if _gain_neon is None:
        _gain_neon = NeonKernels()
    return _gain_neon.near_rail_fraction(a, rail=126)


_gain_neon = None


class GainGovernor:
    """Runtime overload guard for HackRF gain.

    hackrf_transfer programs RX gain when the stream starts; it does not provide a
    portable per-sample gain API. Therefore the governor never invents an in-place
    gain change: after sustained ADC near-rail activity it requests a controlled
    stream restart with a lower gain pair. Quiet signals are left untouched, so weak
    signal sensitivity is not traded away pre-emptively.
    """

    def __init__(self, lna: int, vga: int, enabled: bool = True,
                 clip_fraction: float = 5e-4, sustain_frames: int = 4):
        self.lna = int(lna); self.vga = int(vga); self.enabled = bool(enabled)
        self.clip_fraction = float(clip_fraction)
        self.sustain_frames = max(1, int(sustain_frames))
        self.hot_frames = 0

    def observe(self, raw_master: bytes, raw_slave: bytes) -> None:
        if not self.enabled:
            return
        fm = _clipping_stats(raw_master)
        fs = _clipping_stats(raw_slave)
        hot = max(fm, fs) >= self.clip_fraction
        self.hot_frames = self.hot_frames + 1 if hot else max(0, self.hot_frames - 1)
        if self.hot_frames < self.sustain_frames:
            return
        # First remove VGA in fine 2 dB steps. Once VGA is already low, remove one
        # full LNA step (8 dB). This protects the ADC while preserving as much gain as
        # possible and respecting HackRF's discrete gain steps.
        if self.vga >= 8:
            nvga = max(0, self.vga - 6)
            nlna = self.lna
        elif self.lna >= 8:
            nlna = max(0, self.lna - 8)
            nvga = self.vga
        else:
            self.hot_frames = 0
            return
        self.hot_frames = 0
        raise GainChangeRequest(nlna, nvga,
                                f"ADC headroom low: master/slave near-rail; gain {self.lna}/{self.vga} -> {nlna}/{nvga} dB")

def main():
    ap = argparse.ArgumentParser(description="H618 adaptive multi-target drone DOA")
    ap.add_argument("--freq", type=float, default=2_437_000_000.0)
    ap.add_argument("--sample-rate", type=int, default=5_000_000)
    ap.add_argument("--fft", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fft-workers", type=int, default=choose_default_fft_workers())
    ap.add_argument("--slave-mode", choices=["auto", "exact-bins", "full-fft"], default="auto")
    ap.add_argument("--sparse-max-bins", type=int, default=32)
    ap.add_argument("--baseline-cm", type=float, default=5.0)
    ap.add_argument("--threshold-snr", type=float, default=12.0)
    ap.add_argument("--phase-occupancy-db", type=float, default=1.5)
    ap.add_argument("--peak-cluster-gap-bins", type=int, default=4)
    ap.add_argument("--coherence-min", type=float, default=0.20)
    ap.add_argument("--cfar-train", type=int, default=24)
    ap.add_argument("--cfar-guard", type=int, default=4)
    ap.add_argument("--cfar-pfa", type=float, default=1e-4)
    ap.add_argument("--lna", type=int, default=16, help="RX LNA/IF gain, 0..40 dB in 8 dB steps")
    ap.add_argument("--vga", type=int, default=16, help="RX VGA/baseband gain, 0..62 dB in 2 dB steps")
    ap.add_argument("--gain-auto", action="store_true", help="restart stream at lower gain after sustained ADC near-rail activity")
    ap.add_argument("--gain-clip-fraction", type=float, default=5e-4, help="near-rail sample fraction that starts overload protection")
    ap.add_argument("--gain-sustain-frames", type=int, default=4, help="consecutive hot batches before gain reduction")
    ap.add_argument("--phase-cal-deg", type=float, default=0.0)
    ap.add_argument("--serial-master", default=SERIAL_MASTER_DEFAULT)
    ap.add_argument("--serial-slave", default=SERIAL_SLAVE_DEFAULT)
    ap.add_argument("--queue-slots", type=int, default=6)
    ap.add_argument("--max-frame-gap", type=int, default=2, help="soft skew warning threshold; does not restart hardware")
    ap.add_argument("--hard-frame-gap", type=int, default=64, help="pathological skew threshold that triggers stream recovery")
    ap.add_argument("--coherence-resync-frames", type=int, default=12)
    ap.add_argument("--max-restarts", type=int, default=3)
    ap.add_argument("--encoder-path", default="", help="serial/FIFO/text source; lines: angle or timestamp_ns,angle_deg")
    ap.add_argument("--encoder-freq-tol-hz", type=float, default=250_000.0)
    ap.add_argument("--encoder-zero-deg", type=float, default=0.0, help="encoder zero -> platform heading offset")
    ap.add_argument("--encoder-latency-ms", type=float, default=0.0, help="measured transport delay correction in ms; use measured value, not a blind 16 ms assumption")
    ap.add_argument("--encoder-low-latency", choices=["auto", "force", "off"], default="auto", help="reduce kernel/USB-serial buffering where the driver exposes a low-latency control")
    ap.add_argument("--cpus", default="")
    ap.add_argument("--no-neon", action="store_true")
    args = ap.parse_args()

    global reader, encoder

    validate_frequency(args.freq)
    if not (0 <= args.lna <= 40 and args.lna % 8 == 0):
        ap.error("--lna must be 0..40 in 8 dB steps")
    if not (0 <= args.vga <= 62 and args.vga % 2 == 0):
        ap.error("--vga must be 0..62 in 2 dB steps")
    if args.cpus:
        set_process_affinity(int(x) for x in args.cpus.split(",") if x.strip())

    core = SDRCore(
        n_fft=args.fft, sample_rate=args.sample_rate, center_freq_hz=args.freq,
        baseline_m=args.baseline_cm / 100.0, threshold_snr_db=args.threshold_snr,
        phase_calibration_deg=args.phase_cal_deg, batch_frames=args.batch,
        fft_workers=args.fft_workers, use_neon=not args.no_neon,
        slave_mode=args.slave_mode, sparse_max_bins=args.sparse_max_bins,
        cfar_train=args.cfar_train, cfar_guard=args.cfar_guard, cfar_pfa=args.cfar_pfa,
        phase_occupancy_db=args.phase_occupancy_db, coherence_min=args.coherence_min,
        peak_cluster_gap_bins=args.peak_cluster_gap_bins,
    )

    wavelength = SPEED_OF_LIGHT / args.freq
    gain = GainGovernor(args.lna, args.vga, enabled=args.gain_auto,
                        clip_fraction=args.gain_clip_fraction,
                        sustain_frames=args.gain_sustain_frames)
    print("=" * 86)
    print(" H618 DRONE DOA | CFAR MULTI-TARGET + BAND INTEGRATION + ADAPTIVE SLAVE + NEON")
    print("=" * 86)
    print(f"RF={args.freq/1e6:.3f} MHz | Fs={args.sample_rate/1e6:.3f} MSPS | FFT={args.fft} | batch={args.batch}")
    print(f"slave={args.slave_mode} | sparse cap={args.sparse_max_bins} bins | FFT workers={args.fft_workers}")
    print(f"baseline={args.baseline_cm:.2f} cm | lambda/2={wavelength*50:.2f} cm | NEON={'ON' if core.neon.neon_enabled else 'OFF'}")
    print(f"pair raw rate={4.0*args.sample_rate/1e6:.1f} MB/s | CFAR train={args.cfar_train} guard={args.cfar_guard} Pfa~{args.cfar_pfa:g}")
    print(f"gain={gain.lna}/{gain.vga} dB (LNA/VGA) | overload governor={"ON" if gain.enabled else "OFF"}")

    def new_reader():
        global reader
        reader = FixedBatchReader(proc_master.stdout.fileno(), proc_slave.stdout.fileno(), core.raw_batch_bytes,
                                  slots=args.queue_slots, max_gap=args.max_frame_gap, hard_gap=args.hard_frame_gap)
        reader.start()
        return reader

    trackers = KinematicTrackerBank(args.baseline_cm / 100.0, args.encoder_freq_tol_hz)
    if args.encoder_path:
        encoder = EncoderSource(args.encoder_path)
        if args.encoder_low_latency != "off":
            status = configure_encoder_low_latency(args.encoder_path, strict=(args.encoder_low_latency == "force"))
            print(f"[+] encoder latency setup: {status}")
        encoder.start()
        print(f"[+] encoder source: {args.encoder_path}")

    restart_count = 0
    while not stop_event.is_set():
        stop_event.clear()
        try:
            enable_master_clock(args.serial_master)
            start_streaming(args.serial_master, args.serial_slave, args.freq, args.sample_rate, gain.lna, gain.vga)
            calibrate_alignment(proc_master, proc_slave, args.sample_rate)
            new_reader()

            wall_anchor_ns = time.time_ns()
            mono_anchor_ns = time.monotonic_ns()
            batch_duration_ns = int(args.batch * args.fft / args.sample_rate * 1e9)
            strong_low_coh = 0

            while not stop_event.is_set():
                frame_id, im, is_, raw_m, raw_s, ready_ns = reader.get_pair()
                released = False
                try:
                    gain.observe(raw_m, raw_s)
                    capture_ns = int(ready_ns - batch_duration_ns)
                    core.preprocess_fast(raw_m, raw_s)
                    detections = core.detect_and_localize()

                    high_snr_bad = bool(
                        core.last_incoherent_candidates
                        and any(c.snr_db >= args.threshold_snr + 6.0 for c in core.last_candidates)
                    )
                    for d in detections:
                        det_ns = capture_ns + int(d.frame * args.fft / args.sample_rate * 1e9)
                        wall_ns = wall_anchor_ns + (det_ns - mono_anchor_ns)
                        dt = datetime.fromtimestamp(wall_ns / 1e9, tz=timezone.utc)
                        ts_str = dt.strftime("%H:%M:%S.%f")[:-3]

                        kin_angle = None
                        kin_score = None
                        if encoder is not None:
                            enc = encoder.angle_at(det_ns + int(args.encoder_latency_ms * 1e6))
                            if enc is not None:
                                platform_angle = enc + args.encoder_zero_deg
                                est = trackers.update(d, platform_angle, det_ns / 1e9)
                                if est is not None:
                                    kin_angle = est.angle_deg; kin_score = est.score
                                    d.kinematic_angle_deg = kin_angle; d.kinematic_score = kin_score

                        angle_text = f"DOA={d.angle_deg:+6.1f}°"
                        if kin_angle is not None:
                            angle_text += f" | WORLD={kin_angle:+6.1f}° Kscore={kin_score:+.3f}"
                        ambiguity = "" if d.baseline_unambiguous else " [AMBIGUOUS-STATIC]"
                        print(
                            f"\n🚨 [{ts_str}] id={frame_id} {d.target_freq_hz/1e6:9.3f} MHz | "
                            f"SNR={d.snr_db:5.1f} dB {d.detection_metric} | band={d.occupied_bins:5d} bins | "
                            f"coh={d.coherence:.3f} | phase={d.phase_deg:+7.2f}° | {angle_text}{ambiguity}"
                        )
                        if kin_angle is not None:
                            print(f"   ↳ rotation-resolved world bearing: {kin_angle:+.2f}°")
                        elif not d.baseline_unambiguous and encoder is not None:
                            print(f"   ↳ encoder span={encoder.rotation_span_deg(det_ns + int(args.encoder_latency_ms * 1e6)):.1f}°; waiting for sufficient rotation arc")

                    if not detections:
                        print(f"[{frame_id}] clean | FFT={args.fft} batch={args.batch} slave={','.join(core.last_slave_mode_by_frame.values()) or '-'} NEON={'Y' if core.neon.neon_enabled else 'N'}", end="\r")
                    if high_snr_bad:
                        strong_low_coh += 1
                    else:
                        strong_low_coh = max(0, strong_low_coh - 1)
                    if strong_low_coh >= args.coherence_resync_frames:
                        raise RuntimeError("persistent high-SNR inter-channel incoherence")
                finally:
                    if not released:
                        reader.release_pair(im, is_)

        except GainChangeRequest as exc:
            gain.lna, gain.vga = exc.lna, exc.vga
            restart_count = 0  # gain recovery is a controlled mode change, not a stream-failure budget hit
            print(f"\n[!] gain governor: {exc.reason}; restarting RX at {gain.lna}/{gain.vga} dB")
            if reader is not None:
                reader.stop(); reader.join(timeout=0.2)
            stop_streaming()
            stop_event.clear()
            time.sleep(0.03)
            continue
        except (RuntimeError, TimeoutError) as exc:
            restart_count += 1
            print(f"\n[!] stream recovery {restart_count}/{args.max_restarts}: {exc}")
            if restart_count > args.max_restarts:
                print("[-] maximum automatic recoveries exhausted")
                break
            stop_event.set()
            if reader is not None:
                reader.stop(); reader.join(timeout=0.2)
            stop_streaming()
            stop_event.clear()
            time.sleep(0.05)
            continue
        finally:
            if stop_event.is_set():
                if reader is not None:
                    reader.stop(); reader.join(timeout=0.2)
                stop_streaming()

        # Normal exit only occurs when SIGINT/SIGTERM set stop_event.
        break

    stop_event.set()
    if encoder is not None:
        encoder.stop()
    if reader is not None:
        reader.stop(); reader.join(timeout=0.2)
    stop_streaming()


if __name__ == "__main__":
    main()
