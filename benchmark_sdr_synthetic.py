#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import scipy.fft

from doa_h618_core import KinematicDOAResolver, SDRCore, choose_default_fft_workers

C = 299_792_458.0


def parse_csv(s, cast=float):
    return [cast(x.strip()) for x in s.split(",") if x.strip()]


def wrap_deg(x):
    return ((np.asarray(x) + 180.0) % 360.0) - 180.0


def make_baseband_waveform(n: int, fs: float, rng: np.random.Generator, waveform: str, bandwidth_hz: float):
    if waveform == "tone":
        k = max(1, int(round(0.071 * n)))
        x = np.exp(2j * np.pi * k * np.arange(n) / n).astype(np.complex64)
        occupied = np.asarray([k], dtype=np.int64)
    elif waveform == "multitone":
        x = np.zeros(n, dtype=np.complex64)
        half = max(1, int(round(bandwidth_hz * n / fs / 2.0)))
        center = int(round(0.071 * n))
        bins = np.arange(center - half, center + half + 1)
        bins = bins[(bins >= 1) & (bins < n - 1)]
        for k in bins:
            ph = rng.uniform(0.0, 2.0 * np.pi)
            x += np.exp(2j * np.pi * k * np.arange(n) / n + 1j * ph).astype(np.complex64)
        x /= math.sqrt(max(1, len(bins)))
        occupied = bins.astype(np.int64)
    elif waveform in {"ofdm", "wideband-noise"}:
        width = max(2, int(round(bandwidth_hz * n / fs)))
        center = int(round(0.071 * n))
        lo = max(1, center - width // 2)
        hi = min(n // 2 - 1, center + width // 2)
        spec = np.zeros(n, dtype=np.complex64)
        if waveform == "ofdm":
            q = rng.integers(0, 4, hi - lo + 1)
            symbols = ((2 * (q & 1) - 1) + 1j * (2 * ((q >> 1) & 1) - 1)).astype(np.complex64)
            spec[lo:hi + 1] = symbols
        else:
            spec[lo:hi + 1] = (rng.normal(size=hi - lo + 1) + 1j * rng.normal(size=hi - lo + 1)).astype(np.complex64)
        spec /= math.sqrt(float(np.mean(np.abs(spec[lo:hi + 1]) ** 2)) + 1e-20)
        x = scipy.fft.ifft(spec).astype(np.complex64) * math.sqrt(n)
        occupied = np.arange(lo, hi + 1, dtype=np.int64)
    elif waveform == "dsss":
        chips = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n)
        x = (chips + 1j * rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n)).astype(np.complex64)
        # Band-limit by a rectangular spectral mask.
        spec = scipy.fft.fft(x)
        width = max(2, int(round(bandwidth_hz * n / fs)))
        center = int(round(0.071 * n))
        lo = max(1, center - width // 2); hi = min(n // 2 - 1, center + width // 2)
        mask = np.zeros(n, dtype=bool); mask[lo:hi + 1] = True
        x = scipy.fft.ifft(spec * mask).astype(np.complex64)
        occupied = np.arange(lo, hi + 1, dtype=np.int64)
    else:
        raise ValueError(f"unknown waveform={waveform}")
    x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-20)
    return x.astype(np.complex64), occupied


def make_signal_batch(
    *, n_fft: int, batch: int, sample_rate: float, center_freq_hz: float,
    baseline_m: float, angle_deg: float, snr_db: float, rng: np.random.Generator,
    waveform: str = "tone", bandwidth_hz: float = 1000.0,
    targets: list[dict] | None = None,
):
    t = np.arange(n_fft, dtype=np.float64) / sample_rate
    if targets is None:
        targets = [{"offset_hz": 0.0, "angle_deg": angle_deg, "snr_db": snr_db,
                    "waveform": waveform, "bandwidth_hz": bandwidth_hz}]
    master_frames = []
    slave_frames = []
    gt_bands = []
    for _frame in range(batch):
        xm = np.zeros(n_fft, dtype=np.complex128)
        xs = np.zeros(n_fft, dtype=np.complex128)
        bands = []
        for target in targets:
            bw = float(target.get("bandwidth_hz", bandwidth_hz))
            wf, occ = make_baseband_waveform(n_fft, sample_rate, rng, target.get("waveform", waveform), bw)
            freq_offset = float(target.get("offset_hz", 0.0))
            # Frequency translate in time domain. For finite N this uses a complex oscillator.
            x = wf.astype(np.complex128) * np.exp(2j * np.pi * freq_offset * t)
            theta = math.radians(float(target.get("angle_deg", angle_deg)))
            tau = baseline_m * math.sin(theta) / C
            X = scipy.fft.fft(x)
            fbin = scipy.fft.fftfreq(n_fft, d=1.0 / sample_rate)
            # Exact wideband geometric delay plus RF carrier phase.
            H = np.exp(-2j * np.pi * fbin * tau) * np.exp(-2j * np.pi * center_freq_hz * tau)
            xs_clean = scipy.fft.ifft(X * H)
            snr = float(target.get("snr_db", snr_db))
            sig_p = float(np.mean(np.abs(x) ** 2))
            npow = sig_p / (10.0 ** (snr / 10.0))
            xm += x
            xs += xs_clean
            bands.append({"offset_hz": freq_offset, "angle_deg": math.degrees(theta), "snr_db": snr,
                          "occupied_bins": int(len(occ))})
        # Independent receiver noise with the desired reference power.
        total_sig = float(np.mean(np.abs(xm) ** 2))
        # Use the weakest target SNR as the stress level for the aggregate.
        ref_snr = min(float(tg.get("snr_db", snr_db)) for tg in targets)
        noise_power = total_sig / (10.0 ** (ref_snr / 10.0))
        sigma = math.sqrt(noise_power / 2.0)
        xm += sigma * (rng.normal(size=n_fft) + 1j * rng.normal(size=n_fft))
        xs += sigma * (rng.normal(size=n_fft) + 1j * rng.normal(size=n_fft))
        master_frames.append(xm.astype(np.complex64)); slave_frames.append(xs.astype(np.complex64)); gt_bands.append(bands)

    m = np.asarray(master_frames)
    s = np.asarray(slave_frames)
    # Fixed headroom; quantization remains far below the tested SNR floor.
    peak = max(float(np.max(np.abs(m))), float(np.max(np.abs(s))), 1.0)
    scale = min(52.0, 92.0 / peak)
    def pack(x):
        iq = np.empty((batch, n_fft, 2), dtype=np.int8)
        iq[..., 0] = np.clip(np.rint(x.real * scale), -127, 127).astype(np.int8)
        iq[..., 1] = np.clip(np.rint(x.imag * scale), -127, 127).astype(np.int8)
        return iq.tobytes()
    return pack(m), pack(s), gt_bands


def build_core(args, n_fft, batch, mode, sample_rate, center_freq):
    return SDRCore(
        n_fft=n_fft, sample_rate=sample_rate, center_freq_hz=center_freq,
        baseline_m=args.baseline_cm / 100.0, threshold_snr_db=args.threshold_db,
        batch_frames=batch, fft_workers=args.eval_workers, slave_mode=mode,
        sparse_max_bins=args.sparse_max_bins, phase_occupancy_db=args.phase_occupancy_db,
        coherence_min=args.coherence_min, cfar_train=args.cfar_train, cfar_guard=args.cfar_guard,
        cfar_pfa=args.cfar_pfa, use_neon=not args.no_neon,
    )


def benchmark_processing(args):
    results = []
    fft_sizes = parse_csv(args.fft_sizes, int); workers_list = parse_csv(args.workers, int)
    sample_rates = parse_csv(args.sample_rates, float)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    waveforms = [x.strip() for x in args.waveforms.split(",") if x.strip()]
    for sample_rate in sample_rates:
        for n_fft in fft_sizes:
            for wf in waveforms:
                for workers in workers_list:
                    for mode in modes:
                        old = args.eval_workers; args.eval_workers = workers
                        core = build_core(args, n_fft, args.batch, mode, sample_rate, args.center_freq)
                        args.eval_workers = old
                        rng = np.random.default_rng(args.seed)
                        raw_m, raw_s, _ = make_signal_batch(
                            n_fft=n_fft, batch=args.batch, sample_rate=sample_rate,
                            center_freq_hz=args.center_freq, baseline_m=args.baseline_cm/100.0,
                            angle_deg=17.0, snr_db=args.benchmark_snr, rng=rng, waveform=wf,
                            bandwidth_hz=args.bandwidth_hz,
                        )
                        for _ in range(args.warmup):
                            core.preprocess_fast(raw_m, raw_s); core.detect_and_localize()
                        t0 = time.perf_counter(); mode_seen = {}
                        dets = 0
                        for _ in range(args.repeats):
                            core.preprocess_fast(raw_m, raw_s)
                            ds = core.detect_and_localize(); dets += len(ds)
                            mode_seen.update(core.last_slave_mode_by_frame)
                        dt = time.perf_counter() - t0
                        batches_s = args.repeats / dt; frames_s = batches_s * args.batch
                        req_pair = 4.0 * sample_rate / 1e6
                        eq_pair = frames_s * n_fft * 4.0 / 1e6
                        results.append({
                            "sample_rate": sample_rate, "fft": n_fft, "waveform": wf,
                            "workers": workers, "slave_mode_request": mode,
                            "slave_modes_seen": sorted(set(core.last_slave_mode_by_frame.values())),
                            "neon": bool(core.neon.neon_enabled), "batch": args.batch,
                            "ms_per_batch": 1000.0 * dt / args.repeats, "frames_per_s": frames_s,
                            "equiv_raw_pair_mbps": eq_pair, "required_pair_mbps": req_pair,
                            "realtime_factor": frames_s * n_fft / sample_rate,
                            "detections": dets,
                        })
    return results


def benchmark_pipe_read(args):
    chunk = args.pipe_chunk_kb * 1024
    total_bytes = int(args.pipe_mb * 1024 * 1024)
    def one_pipe():
        rfd, wfd = os.pipe(); pattern = os.urandom(chunk)
        writer_err = []
        def writer():
            try:
                sent = 0
                while sent < total_bytes:
                    piece = pattern[:min(chunk, total_bytes-sent)]
                    off = 0
                    while off < len(piece):
                        off += os.write(wfd, piece[off:])
                    sent += len(piece)
            except BaseException as e:
                writer_err.append(e)
            finally:
                os.close(wfd)
        tw = threading.Thread(target=writer, daemon=True); tw.start()
        buf = bytearray(chunk); got = 0; t0 = time.perf_counter()
        while got < total_bytes:
            n = os.readv(rfd, [buf]);
            if n == 0: break
            got += n
        dt = time.perf_counter()-t0; os.close(rfd); tw.join()
        if writer_err: raise writer_err[0]
        return got, dt
    g0, dt0 = one_pipe()
    # Concurrent dual-pipe throughput approximates two HackRF streams.
    pipes=[]
    for _ in range(2): pipes.append((os.pipe(), os.urandom(chunk)))
    errs=[]
    def writer2(wfd, pattern):
        try:
            sent=0
            while sent<total_bytes:
                piece=pattern[:min(chunk,total_bytes-sent)]; off=0
                while off<len(piece): off += os.write(wfd,piece[off:])
                sent += len(piece)
        except BaseException as e: errs.append(e)
        finally: os.close(wfd)
    ts=[]
    for (rfd,wfd),pat in pipes:
        t=threading.Thread(target=writer2,args=(wfd,pat),daemon=True); t.start(); ts.append(t)
    bufs=[bytearray(chunk),bytearray(chunk)]; got=[0,0]
    t0=time.perf_counter()
    # Nonblocking not required; alternate blocking reads fairly.
    while got[0]<total_bytes or got[1]<total_bytes:
        for i in range(2):
            (rfd, _), _pat = pipes[i]
            buf = bufs[i]
            if got[i]>=total_bytes: continue
            n=os.readv(rfd,[buf]);
            if n==0: got[i]=total_bytes; continue
            got[i]+=n
    dt=time.perf_counter()-t0
    for ((rfd,_),_pat),t in zip(pipes,ts): os.close(rfd); t.join()
    if errs: raise errs[0]
    return {"single_MBps":g0/dt0/1e6,"dual_MBps":sum(got)/dt/1e6,"total_bytes_each":total_bytes}


def run_core_once(core, raw_m, raw_s):
    core.preprocess_fast(raw_m, raw_s)
    return core.detect_and_localize()


def benchmark_detection(args):
    snrs=parse_csv(args.snrs,float); freqs=parse_csv(args.freqs,float); angles=parse_csv(args.angles,float)
    waveforms=[x.strip() for x in args.waveforms.split(",") if x.strip()]
    bandwidths=parse_csv(args.bandwidths_hz,float)
    modes=[x.strip() for x in args.modes.split(",") if x.strip()]
    rng=np.random.default_rng(args.seed+10); out=[]
    for wf in waveforms:
        for bw_req in bandwidths:
            bw=min(float(bw_req), 0.45*args.sample_rate)
            for freq in freqs:
                cores={mode:build_core(args,args.eval_fft,1,mode,args.sample_rate,freq) for mode in modes}
                for snr in snrs:
                    stat={mode:{"det":0,"errs":[],"coh":[],"full":0} for mode in modes}
                    trials=0
                    for angle in angles:
                        for _ in range(args.trials):
                            # Generate one identical RF realization, then feed it to every mode.
                            rm,rs,_=make_signal_batch(
                                n_fft=args.eval_fft,batch=1,sample_rate=args.sample_rate,
                                center_freq_hz=freq,baseline_m=args.baseline_cm/100.0,
                                angle_deg=angle,snr_db=snr,rng=rng,waveform=wf,bandwidth_hz=bw,
                            )
                            trials+=1
                            gt_offset=0.071*args.sample_rate
                            for mode,core in cores.items():
                                ds=run_core_once(core,rm,rs)
                                if ds:
                                    stat[mode]["det"]+=1
                                    # Pick the closest detected frequency to the known synthetic emitter.
                                    d=min(ds,key=lambda x:abs((x.target_freq_hz-freq)-(gt_offset)))
                                    stat[mode]["errs"].append(float(wrap_deg(d.angle_deg-angle)))
                                    stat[mode]["coh"].append(d.coherence)
                                    if d.slave_compute_mode=="full-fft": stat[mode]["full"]+=1
                    for mode in modes:
                        st=stat[mode]
                        out.append({
                            "waveform":wf,"bandwidth_hz":bw,"freq_hz":freq,"snr_db":snr,"slave_mode":mode,
                            "P_detect":st["det"]/max(1,trials),
                            "angle_rmse_deg":float(np.sqrt(np.mean(np.square(st["errs"])))) if st["errs"] else None,
                            "median_coherence":float(np.median(st["coh"])) if st["coh"] else None,
                            "full_slave_fraction":st["full"]/max(1,st["det"]),
                            "baseline_lambda_over_2_safe":(args.baseline_cm/100.0)<=C/(2*freq),
                        })
    return out

def benchmark_multi_target(args):
    rng=np.random.default_rng(args.seed+20)
    targets=[
        {"offset_hz":-0.26*args.sample_rate,"angle_deg":-21.0,"snr_db":args.multitarget_snr,"waveform":"tone","bandwidth_hz":1000.0},
        {"offset_hz":+0.22*args.sample_rate,"angle_deg":+31.0,"snr_db":args.multitarget_snr,"waveform":"ofdm","bandwidth_hz":0.06*args.sample_rate},
        {"offset_hz":+0.06*args.sample_rate,"angle_deg":+7.0,"snr_db":args.multitarget_snr-3.0,"waveform":"tone","bandwidth_hz":1000.0},
    ]
    core=build_core(args,args.eval_fft,1,"auto",args.sample_rate,args.center_freq)
    found=[]
    for _ in range(args.multitarget_trials):
        rm,rs,gt=make_signal_batch(
            n_fft=args.eval_fft,batch=1,sample_rate=args.sample_rate,center_freq_hz=args.center_freq,
            baseline_m=args.baseline_cm/100.0,angle_deg=0,snr_db=args.multitarget_snr,rng=rng,targets=targets,
        )
        ds=run_core_once(core,rm,rs)
        found.extend(ds)
    expected=[0.071*args.sample_rate + float(t["offset_hz"]) for t in targets]
    matched=[0]*len(expected); errors=[]
    for d in found:
        j=min(range(len(expected)),key=lambda i:abs(d.freq_offset_hz-expected[i]))
        if abs(d.freq_offset_hz-expected[j]) < 0.04*args.sample_rate and not matched[j]:
            matched[j]=1; errors.append(abs(float(wrap_deg(d.angle_deg-targets[j]["angle_deg"]))))
    return {"targets":targets,"detections_total":len(found),"targets_recovered":sum(matched),
            "target_recall":sum(matched)/len(matched),"matched_angle_abs_error_deg":errors}


def benchmark_sparse_equivalence(args):
    rng=np.random.default_rng(args.seed+30)
    core=build_core(args,args.eval_fft,1,"auto",args.sample_rate,args.center_freq)
    x=(rng.normal(size=args.eval_fft)+1j*rng.normal(size=args.eval_fft)).astype(np.complex64)
    core.slave_td[0]=x*core.window
    full=scipy.fft.fft(core.slave_td[0],workers=args.eval_workers)
    bins=core.selected_bins(args.eval_fft//7)
    sparse=core._exact_bins(0,bins)
    err=np.abs(sparse.astype(np.complex128)-full[bins].astype(np.complex128))
    # Narrowband coherent energy preservation: exact selected bins vs corresponding full FFT bins.
    ratio=float(np.abs(np.vdot(sparse,sparse))/max(1e-20,np.abs(np.vdot(full[bins],full[bins]))))
    return {"max_abs_error":float(np.max(err)),"max_rel_error":float(np.max(err)/(np.max(np.abs(full[bins]))+1e-20)),
            "energy_ratio":ratio,"bins":len(bins),
            "note":"Sparse path uses exact DFT coefficients; auto chooses it only while measured occupied-band workload <= sparse_max_bins."}


def benchmark_rotation(args):
    rng=np.random.default_rng(args.seed+40)
    resolver=KinematicDOAResolver(args.baseline_cm/100.0,min_samples=12,min_span_deg=35.0,grid_step_deg=0.5,history=96)
    f=args.rotation_freq; theta=args.rotation_target_angle; psi=np.linspace(-70.0,70.0,args.rotation_samples)
    tau=args.baseline_cm/100.0*np.sin(np.deg2rad(theta-psi))/C
    phi=np.rad2deg(2*np.pi*f*tau)
    phi += rng.normal(0.0,args.rotation_phase_noise_deg,size=phi.size)
    estimates=[]
    for p,ph in zip(psi,phi):
        e=resolver.add(ph,f,float(p),weight=1.0)
        if e: estimates.append(e)
    final=estimates[-1] if estimates else None
    return {"freq_hz":f,"target_angle_deg":theta,"rotation_span_deg":float(psi[-1]-psi[0]),
            "static_ambiguity":(args.baseline_cm/100.0)>C/(2*f),
            "resolved_angle_deg":None if final is None else final.angle_deg,
            "abs_error_deg":None if final is None else abs(float(wrap_deg(final.angle_deg-theta))),
            "score":None if final is None else final.score,
            "samples_used":None if final is None else final.samples}


def main():
    ap=argparse.ArgumentParser(description="Synthetic H618/NEON SDR benchmark: speed, RF bandwidth, SNR, multi-target and rotating-baseline ambiguity")
    ap.add_argument("--fft-sizes",default="4096,8192,16384,32768,65536")
    ap.add_argument("--workers",default="1,2,3,4")
    ap.add_argument("--modes",default="auto,exact-bins,full-fft")
    ap.add_argument("--waveforms",default="tone,multitone,ofdm,wideband-noise,dsss")
    ap.add_argument("--sample-rates",default="2000000,5000000,10000000,20000000")
    ap.add_argument("--eval-fft",type=int,default=32768)
    ap.add_argument("--eval-workers",type=int,default=3)
    ap.add_argument("--batch",type=int,default=4)
    ap.add_argument("--sample-rate",type=float,default=5_000_000.0)
    ap.add_argument("--center-freq",type=float,default=2_437_000_000.0)
    ap.add_argument("--baseline-cm",type=float,default=5.0)
    ap.add_argument("--threshold-db",type=float,default=12.0)
    ap.add_argument("--phase-occupancy-db",type=float,default=1.5)
    ap.add_argument("--sparse-max-bins",type=int,default=32)
    ap.add_argument("--coherence-min",type=float,default=0.20)
    ap.add_argument("--cfar-train",type=int,default=24)
    ap.add_argument("--cfar-guard",type=int,default=4)
    ap.add_argument("--cfar-pfa",type=float,default=1e-4)
    ap.add_argument("--bandwidth-hz",type=float,default=100_000.0)
    ap.add_argument("--bandwidths-hz",default="1000,100000,1000000,5000000", help="detection bandwidth matrix; clipped by sample rate")
    ap.add_argument("--benchmark-snr",type=float,default=18.0)
    ap.add_argument("--snrs",default="-6,0,6,12,18,24")
    ap.add_argument("--freqs",default="800000000,2437000000,3000000000,6000000000")
    ap.add_argument("--angles",default="-60,-30,0,30,60")
    ap.add_argument("--trials",type=int,default=12)
    ap.add_argument("--warmup",type=int,default=2)
    ap.add_argument("--repeats",type=int,default=12)
    ap.add_argument("--pipe-mb",type=int,default=64)
    ap.add_argument("--pipe-chunk-kb",type=int,default=256)
    ap.add_argument("--multitarget-snr",type=float,default=15.0)
    ap.add_argument("--multitarget-trials",type=int,default=20)
    ap.add_argument("--rotation-freq",type=float,default=6_000_000_000.0)
    ap.add_argument("--rotation-target-angle",type=float,default=37.0)
    ap.add_argument("--rotation-samples",type=int,default=80)
    ap.add_argument("--rotation-phase-noise-deg",type=float,default=3.0)
    ap.add_argument("--seed",type=int,default=20260929)
    ap.add_argument("--no-neon",action="store_true")
    ap.add_argument("--skip-processing",action="store_true")
    ap.add_argument("--json",default="benchmark_results.json")
    args=ap.parse_args()

    print("="*104)
    print(" SYNTHETIC SDR BENCHMARK | H618 | NEON | CFAR | WIDEBAND PHASE INTEGRATION | MULTI-TARGET | KINEMATIC DOA")
    print("="*104)
    print(f"machine={platform.machine()} cpu_count={os.cpu_count()} | native library={'OFF' if args.no_neon else 'AUTO'}")

    report={"platform":{"machine":platform.machine(),"cpu_count":os.cpu_count()},"settings":vars(args)}
    if not args.skip_processing:
        proc=benchmark_processing(args); report["processing"]=proc
        print("\n[1] DSP THROUGHPUT")
        print(f"{'Fs':>6} {'FFT':>7} {'WF':>13} {'W':>2} {'REQ':>7} {'MEAS':>7} {'ms/b':>8} {'RTx':>7} {'slave seen':>14}")
        for r in proc:
            print(f"{r['sample_rate']/1e6:6.1f} {r['fft']:7d} {r['waveform']:>13} {r['workers']:2d} {r['required_pair_mbps']:7.1f} {r['equiv_raw_pair_mbps']:7.1f} {r['ms_per_batch']:8.2f} {r['realtime_factor']:7.2f} {','.join(r['slave_modes_seen']):>14}")

    pipe=benchmark_pipe_read(args); report["pipe_read"]=pipe
    print("\n[2] SYNTHETIC PIPE INGEST")
    print(f"single={pipe['single_MBps']:.1f} MB/s | dual={pipe['dual_MBps']:.1f} MB/s aggregate | target at 5 MSPS = 20 MB/s pair")

    eq=benchmark_sparse_equivalence(args); report["sparse_equivalence"]=eq
    print("\n[3] EXACT SPARSE DFT EQUIVALENCE")
    print(f"bins={eq['bins']} max_abs={eq['max_abs_error']:.3g} max_rel={eq['max_rel_error']:.3g} energy_ratio={eq['energy_ratio']:.9f}")

    det=benchmark_detection(args); report["detection"]=det
    print("\n[4] DETECTION / DOA MATRIX")
    print(f"{'WF':>13} {'BW kHz':>8} {'RF':>8} {'SNR':>6} {'MODE':>10} {'Pdet':>7} {'RMSE':>9} {'coh':>7} {'fullS':>7} {'safe':>5}")
    for r in det:
        rm='-' if r['angle_rmse_deg'] is None else f"{r['angle_rmse_deg']:.2f}"; co='-' if r['median_coherence'] is None else f"{r['median_coherence']:.3f}"
        print(f"{r['waveform']:>13} {r['bandwidth_hz']/1e3:8.1f} {r['freq_hz']/1e6:8.0f} {r['snr_db']:6.1f} {r['slave_mode']:>10} {r['P_detect']:7.3f} {rm:>9} {co:>7} {r['full_slave_fraction']:7.3f} {str(r['baseline_lambda_over_2_safe'])[0]:>5}")

    mt=benchmark_multi_target(args); report["multi_target"]=mt
    print("\n[5] MULTI-TARGET")
    print(f"recovered={mt['targets_recovered']}/{len(mt['targets'])} recall={mt['target_recall']:.3f} detections={mt['detections_total']} angle errors={mt['matched_angle_abs_error_deg']}")

    rot=benchmark_rotation(args); report["rotation"]=rot
    print("\n[6] ROTATING-BASELINE AMBIGUITY RESOLUTION")
    print(json.dumps(rot,indent=2))

    Path(args.json).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\n[+] JSON report: {Path(args.json).resolve()}")


if __name__=="__main__":
    main()
