from __future__ import annotations

import os
import select
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path



def configure_encoder_low_latency(path: str, strict: bool = False) -> str:
    """Best-effort low-latency setup, only where the OS/driver exposes a knob.

    Important: a built-in UART such as /dev/ttyS1 does NOT inherently have a generic
    16 ms USB latency timer. The 16 ms timer is a characteristic of some USB-serial
    adapters/drivers (FTDI is a common example). We therefore inspect the actual tty
    driver and only touch a supported control. Failure is non-fatal unless strict=True.
    """
    p = Path(path)
    if not p.is_char_device() or not p.name.startswith("tty"):
        return "not-a-serial-device"

    tty_name = p.name
    latency_attr = Path("/sys/class/tty") / tty_name / "device" / "latency_timer"
    if latency_attr.exists():
        try:
            latency_attr.write_text("1\n")
            value = latency_attr.read_text().strip()
            if value == "1":
                return f"latency_timer=1ms ({tty_name})"
            msg = f"latency_timer write not confirmed ({tty_name}: {value!r})"
        except Exception as exc:
            msg = f"latency_timer unavailable ({tty_name}: {exc})"
        if strict:
            raise RuntimeError(msg)
        return msg + "; use --encoder-latency-ms with a measured value"

    setserial = shutil.which("setserial")
    if setserial:
        try:
            subprocess.run([setserial, str(p), "low_latency"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            probe = subprocess.run([setserial, "-a", str(p)], check=True,
                                  capture_output=True, text=True)
            if "low_latency" in probe.stdout:
                return f"setserial low_latency ({tty_name})"
            msg = f"setserial ran but low_latency was not reported ({tty_name})"
        except Exception as exc:
            msg = f"setserial low_latency unavailable ({tty_name}: {exc})"
        if strict:
            raise RuntimeError(msg)
        return msg + "; use --encoder-latency-ms with a measured value"

    msg = f"no driver latency control found for {tty_name}; use measured --encoder-latency-ms"
    if strict:
        raise RuntimeError(msg)
    return msg


class EncoderSource:
    """Low-latency encoder reader for serial device, FIFO, pipe, or text file.

    Accepted lines:
      angle_deg
      timestamp_ns,angle_deg
      timestamp_ns angle_deg

    For serial character devices we avoid Python's buffered text iterator and read
    through select()+os.read(), minimizing host-side delivery latency. Serial port
    baud/parity/etc. remain under the OS/MCU configuration. Any transport delay is
    compensated explicitly with --encoder-latency-ms from the main program.
    """

    def __init__(self, path: str, max_history: int = 256, poll_ms: float = 2.0):
        self.path = path
        self.samples = deque(maxlen=max_history)  # (monotonic_ns, unwrapped_angle_deg)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self._last_raw: float | None = None
        self._last_unwrapped: float | None = None
        self.poll_s = max(0.0005, float(poll_ms) / 1000.0)

    @staticmethod
    def _unwrap_step(prev_raw: float | None, prev_unwrapped: float | None, raw: float) -> float:
        if prev_raw is None or prev_unwrapped is None:
            return raw
        delta = (raw - prev_raw + 180.0) % 360.0 - 180.0
        return prev_unwrapped + delta

    def _parse(self, line: str) -> tuple[int, float]:
        parts = line.replace(",", " ").split()
        if not parts:
            raise ValueError("empty encoder line")
        if len(parts) == 1:
            return time.monotonic_ns(), float(parts[0])
        return int(float(parts[0])), float(parts[1])

    def _store_line(self, line: str) -> None:
        try:
            ts_ns, raw = self._parse(line.strip())
            unwrapped = self._unwrap_step(self._last_raw, self._last_unwrapped, raw)
            self._last_raw = raw
            self._last_unwrapped = unwrapped
            with self.lock:
                self.samples.append((ts_ns, unwrapped))
        except Exception:
            return

    def _run_char_device(self) -> None:
        fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
        try:
            buf = bytearray()
            while not self.stop_event.is_set():
                ready, _, _ = select.select([fd], [], [], self.poll_s)
                if not ready:
                    continue
                chunk = os.read(fd, 4096)
                if not chunk:
                    time.sleep(self.poll_s)
                    continue
                buf.extend(chunk)
                while True:
                    try:
                        pos = buf.index(10)  # '\n'
                    except ValueError:
                        break
                    line = bytes(buf[:pos]).rstrip(b"\r")
                    del buf[:pos + 1]
                    if line:
                        self._store_line(line.decode("ascii", errors="ignore"))
        finally:
            os.close(fd)

    def _run_stream(self) -> None:
        if self.path == "-":
            import sys
            for line in sys.stdin:
                if self.stop_event.is_set():
                    break
                self._store_line(line)
            return
        with open(self.path, "r", buffering=1, encoding="utf-8", errors="replace") as f:
            for line in f:
                if self.stop_event.is_set():
                    break
                self._store_line(line)

    def _run(self):
        try:
            p = Path(self.path)
            if p.is_char_device():
                self._run_char_device()
            else:
                self._run_stream()
        except BaseException as exc:
            self.error = exc

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, name="encoder-reader", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def angle_at(self, timestamp_ns: int) -> float | None:
        with self.lock:
            s = list(self.samples)
        if not s:
            return None
        if timestamp_ns <= s[0][0]:
            return s[0][1]
        if timestamp_ns >= s[-1][0]:
            return s[-1][1]
        for i in range(1, len(s)):
            t0, a0 = s[i - 1]
            t1, a1 = s[i]
            if t0 <= timestamp_ns <= t1:
                if t1 == t0:
                    return a1
                u = (timestamp_ns - t0) / (t1 - t0)
                return a0 + u * (a1 - a0)
        return s[-1][1]

    def rotation_span_deg(self, timestamp_ns: int, window_s: float = 2.0) -> float:
        cutoff = timestamp_ns - int(window_s * 1e9)
        with self.lock:
            a = [x[1] for x in self.samples if x[0] >= cutoff]
        if len(a) < 2:
            return 0.0
        return float(max(a) - min(a))
