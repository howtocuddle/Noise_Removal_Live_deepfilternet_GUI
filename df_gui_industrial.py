#!/usr/bin/env python3
"""
Groundbreakin — Live Mic Noise Filtering (DeepFilterNet)

Install (CPU):
  pip install numpy sounddevice soundfile
  pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
  pip install deepfilternet

Run:
  python df_gui_industrial.py
"""

from __future__ import annotations

import queue
import time
from collections import deque
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import colorsys
import sys
import os

import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sounddevice as sd
import soundfile as sf
import warnings
# (resampling helpers moved to module dfgui.utils)
# VST/Pedalboard removed (AI-only pipeline)

# Echo cancellation removed for lightweight (AI-only) pipeline
WEBRTC_AEC_AVAILABLE = False

DF_AVAILABLE = True
try:
    warnings.filterwarnings("ignore", message="`torchaudio.backend.common.AudioMetaData` has been moved")
    import df.enhance  # type: ignore  # noqa: F401
except Exception:
    DF_AVAILABLE = False


# ---- Theme (Groundbreak-inspired) ----
THEME = {
    # OLED black design with Groundbreak accent
    "bg": "#000000",
    "panel": "#0b0b0b",
    "sunken": "#0a0a0a",
    "stroke": "#1a1a1a",
    "text": "#ffffff",
    "muted": "#d0d0d0",
    "accent": "#ffd319",
}

SR = 48_000
BLOCK = 1024
CHANNELS_IN = 1
CHANNELS_OUT = 1
# Target maximum noise attenuation in dB (higher -> stronger suppression)
ATTN_DB = 100.0
MAX_RETUNE_ATTEMPTS = 2


from dfgui.engine import DFEngine
from dfgui.utils import resample_best as _resample_best_mod


class AudioWorker(threading.Thread):
    def __init__(self, engine, in_q, out_q, writer, level_callback=None, error_callback=None, stop_event=None, writer_lock: threading.Lock | None = None, writer_closed: threading.Event | None = None, echo_enabled: bool = False, ref_q: queue.Queue | None = None, sample_rate: int = 48000, spec_callback=None, aec_mode: str | None = None, gain_db: float = 6.0, vst_board=None, vst_after_ai: bool = True, low_latency_mode: bool = False, vst_bypass: bool = False, vst_wet: float = 1.0,
                 comp_enabled: bool = True, comp_target_db: float = -20.0, comp_max_gain_db: float = 12.0, comp_floor_db: float = -55.0, comp_floor_max_gain_db: float = 6.0, comp_knee_db: float = 10.0, comp_step_up_db: float = 0.1, comp_step_down_db: float = 0.2):
        super().__init__(daemon=True)
        self.engine, self.in_q, self.out_q = engine, in_q, out_q
        self.writer, self.level_callback = writer, level_callback
        self.error_callback = error_callback
        self.stop_event = stop_event or threading.Event()
        self.writer_lock = writer_lock or threading.Lock()
        self.writer_closed = writer_closed or threading.Event()
        self._last_in_time = time.perf_counter()
        # Echo/AEC disabled
        self.echo_enabled = False
        self.aec_mode = "Off"
        self.ref_q = None
        self.sample_rate = int(sample_rate) if sample_rate else 48000
        self.spec_callback = spec_callback
        self._slow_count = 0
        self.low_latency_mode = bool(low_latency_mode)

        # Optional WebRTC AEC state
        self._apm = None
        self._webrtc_sr = 16000
        self._webrtc_frame = int(self._webrtc_sr // 100)  # 10 ms
        self._rev_buf = np.zeros(0, dtype=np.float32)
        # Static output gain with limiter (not a normalizer)
        try:
            self.gain_lin = float(10 ** (float(gain_db) / 20.0))
        except Exception:
            self.gain_lin = 1.0
        # VST disabled
        self.vst_board = None
        self.vst_after_ai = False
        self.vst_bypass = True
        self.vst_wet = 1.0
        # AEC backend not initialized (disabled)
        # Upward compressor (pre-AI) state
        self._comp_gain_db: float = 0.0
        self._comp_enabled: bool = bool(comp_enabled)
        self._comp_target_db: float = float(comp_target_db)
        self._comp_max_gain_db: float = float(comp_max_gain_db)
        self._comp_floor_db: float = float(comp_floor_db)
        self._comp_max_gain_floor_db: float = float(comp_floor_max_gain_db)
        self._comp_knee_db: float = float(comp_knee_db)
        self._comp_step_up_db: float = float(comp_step_up_db)
        self._comp_step_down_db: float = float(comp_step_down_db)
        # Hold to prevent sudden gain drop while speaking
        self._comp_hold_ms: float = 150.0
        self._comp_hold_frames_left: int = 0
        # Glue preset extras
        self._glue_amount: float = 0.5
        self._bright_amount: float = 0.0
        self._ott_amount: float = 0.0
        self._glue_mix: float = 0.8
        self._tone_lp_state: float = 0.0

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x))) if x.size else 0.0)

    def run(self):
        while not self.stop_event.is_set():
            try:
                chunk = self.in_q.get(timeout=0.03)
            except queue.Empty:
                continue
            self._last_in_time = time.perf_counter()

            if chunk.ndim == 2:
                mono = chunk.mean(axis=1, dtype=np.float32)
            else:
                mono = chunk.reshape(-1).astype(np.float32)

            # Echo processing removed

            # Direct processing (no pre-normalization)
            try:
                # Direct path (no VST pre-processing)
                x = mono
                # Pre-AI upward compression to raise quiet parts + glue tone
                try:
                    if getattr(self, '_comp_enabled', True):
                        # Parallel glue compression then gentle upward lift
                        x = self._parallel_glue(x, int(self.sample_rate))
                        x = self._compress_upward_chunk(x, int(self.sample_rate))
                except Exception:
                    pass
                if self.engine is None:
                    enhanced = x
                else:
                    df_sr = getattr(self.engine, 'df_sr', 48000)
                    t0 = time.perf_counter()
                    if int(self.sample_rate) != int(df_sr):
                        x_df = App._resample_best(x, int(self.sample_rate), int(df_sr))  # type: ignore[attr-defined]
                        y_df = self.engine.process(x_df)
                        enhanced = App._resample_best(y_df, int(df_sr), int(self.sample_rate))  # type: ignore[attr-defined]
                    else:
                        enhanced = self.engine.process(x)
                    # Watchdog timing
                    dt = time.perf_counter() - t0
                    blk = max(1, mono.size) / max(1, float(self.sample_rate))
                    if dt > 2.0 * blk:
                        self._slow_count += 1
                    else:
                        self._slow_count = max(0, self._slow_count - 1)
                    if self._slow_count >= 6:
                        self.engine = None
                        if self.error_callback:
                            try:
                                self.error_callback("AI too slow; switching to pass-through.")
                            except Exception:
                                pass
                # No VST post-processing
                # Minimal safety: clip to [-1, 1]
                try:
                    enhanced = np.clip(enhanced.astype(np.float32, copy=False), -1.0, 1.0)
                except Exception:
                    pass
            except Exception as e:
                if self.error_callback:
                    try:
                        self.error_callback(f"Enhance error: {e}")
                    except Exception:
                        pass
                break

            if self.level_callback:
                try:
                    self.level_callback(self._rms(enhanced))
                except Exception:
                    pass
            # Move spectrogram work off the audio callback thread
            if self.spec_callback is not None:
                try:
                    self.spec_callback(mono)
                except Exception:
                    pass

            # Safe write to file if recording
            if self.writer is not None:
                try:
                    with self.writer_lock:
                        if self.writer_closed.is_set() or self.writer is None:
                            break
                        self.writer.write(enhanced)
                except Exception as e:
                    if self.error_callback:
                        try:
                            self.error_callback(f"Write error: {e}")
                        except Exception:
                            pass
                    break

            # Feed live output queue (non-blocking)
            try:
                if self.out_q is not None:
                    self.out_q.put_nowait(enhanced.reshape(-1, 1))
            except Exception:
                pass

    def _compress_upward_chunk(self, x: np.ndarray, sr: int) -> np.ndarray:
        if x.size == 0:
            return x
        # Process in short frames (≈10 ms) to avoid large per-chunk jumps
        frame = max(64, int(sr // 100))
        n = x.size
        y = np.empty_like(x, dtype=np.float32)
        gain_db = float(self._comp_gain_db)
        i = 0
        while i < n:
            j = min(n, i + frame)
            seg = x[i:j]
            # Frame RMS
            rms = float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0
            lvl_db = -120.0 if rms <= 1e-9 else float(20.0 * np.log10(max(1e-9, rms)))
            desired_db = 0.0
            if lvl_db < self._comp_target_db:
                below = float(self._comp_target_db - lvl_db)
                if below < self._comp_knee_db:
                    below *= (below / max(1e-6, self._comp_knee_db))
                max_gain = self._comp_max_gain_db
                if lvl_db < self._comp_floor_db:
                    max_gain = min(max_gain, self._comp_max_gain_floor_db)
                desired_db = min(max_gain, below)
            # Per-frame rate limiting
            delta = desired_db - gain_db
            # Update hold: if near speaking level, refresh hold timer
            frame_ms = float(1000.0 * (j - i) / max(1, int(sr)))
            if lvl_db > (self._comp_target_db - 12.0):
                # speaking region, hold release for a short time
                self._comp_hold_frames_left = int(max(1.0, self._comp_hold_ms / max(1.0, frame_ms)))
            else:
                if self._comp_hold_frames_left > 0:
                    self._comp_hold_frames_left -= 1
            # Prevent downward steps while hold is active
            if self._comp_hold_frames_left > 0 and delta < 0.0:
                delta = 0.0
            if delta > self._comp_step_up_db:
                delta = self._comp_step_up_db
            elif delta < -self._comp_step_down_db:
                delta = -self._comp_step_down_db
            gain_db = float(gain_db + delta)
            # Smooth gain change across the frame to avoid steps
            g_prev = float(10.0 ** (self._comp_gain_db / 20.0))
            g_cur = float(10.0 ** (gain_db / 20.0))
            L = j - i
            if L > 1:
                g_vec = np.linspace(g_prev, g_cur, num=L, dtype=np.float32)
                y[i:j] = seg.astype(np.float32, copy=False) * g_vec
            else:
                y[i:j] = seg.astype(np.float32, copy=False) * g_cur
            self._comp_gain_db = gain_db
            i = j
        # Apply brightness (simple high-shelf via 1st-order HP boost)
        b = float(max(0.0, min(1.0, getattr(self, '_bright_amount', 0.0))))
        if b > 0.0:
            fc = float(1000.0 + 5000.0 * b)
            alpha = float(1.0 - np.exp(-2.0 * np.pi * fc / max(1.0, float(sr))))
            lp = float(self._tone_lp_state)
            yy = y.copy()
            for k in range(n):
                lp = lp + alpha * (yy[k] - lp)
                hp = yy[k] - lp
                yy[k] = yy[k] + (0.5 * b) * hp
            self._tone_lp_state = lp
            y = yy
        # Soft safety
        # Gentle soft clip when needed
        if float(np.max(np.abs(y))) > 1.0:
            y = (np.tanh(1.2 * y) / np.tanh(1.2)).astype(np.float32, copy=False)
        # Dry/wet glue mix
        w = float(max(0.0, min(1.0, getattr(self, '_glue_mix', 0.8))))
        if w < 1.0:
            y = (w * y + (1.0 - w) * x).astype(np.float32, copy=False)
        self._comp_gain_db = float(gain_db)
        return y

    def _parallel_glue(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Simple parallel glue compression: mild downward compression on a parallel path,
        smoothed per ~10 ms, with optional brightness on the compressed path, then mixed.
        Controlled by _glue_amount (0..1), _bright_amount (0..1), _glue_mix (0..1).
        """
        if x.size == 0:
            return x
        g_amt = float(max(0.0, min(1.0, getattr(self, '_glue_amount', 0.5))))
        mix = float(max(0.0, min(1.0, getattr(self, '_glue_mix', 0.8))))
        b_amt = float(max(0.0, min(1.0, getattr(self, '_bright_amount', 0.0))))
        # Static curve parameters from glue amount
        thr_db = -24.0 + 8.0 * (1.0 - g_amt)     # -24..-16
        ratio = 2.0 + 4.0 * g_amt                # 2..6:1
        atk_ms = 5.0 + 15.0 * g_amt              # 5..20 ms
        rel_ms = 80.0 + 120.0 * g_amt            # 80..200 ms
        step_up = 0.05 + 0.20 * (1.0 - g_amt)    # 0.05..0.25 dB/frame
        step_dn = 0.10 + 0.40 * (1.0 - g_amt)    # 0.10..0.50 dB/frame
        # Frame grid
        frame = max(32, int(sr // 100))
        n = x.size
        y = np.empty_like(x, dtype=np.float32)
        g_db = getattr(self, '_glue_gain_db', 0.0)
        i = 0
        # optional brightness shelf on compressed path only
        if b_amt > 0.0:
            fc = float(1500.0 + 6000.0 * b_amt)
            alpha = float(1.0 - np.exp(-2.0 * np.pi * fc / max(1.0, float(sr))))
            lp = float(getattr(self, '_glue_lp_state', 0.0))
        else:
            alpha = 0.0
            lp = 0.0
        while i < n:
            j = min(n, i + frame)
            seg = x[i:j]
            # RMS level in dBFS
            rms = float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0
            lvl_db = -120.0 if rms <= 1e-9 else float(20.0 * np.log10(max(1e-9, rms)))
            # Gain computer (downward compression above threshold)
            over = max(0.0, lvl_db - thr_db)
            gr_db = -over * (1.0 - 1.0/ratio)  # negative gain reduction
            target_db = g_db + gr_db
            # Rate limit per frame with hold against downward dips
            delta = target_db - g_db
            frame_ms = float(1000.0 * (j - i) / max(1, int(sr)))
            if lvl_db > (thr_db - 6.0):
                # near/above threshold (voice active) -> refresh hold
                setattr(self, '_glue_hold_frames_left', int(max(1.0, 120.0 / max(1.0, frame_ms))))
            else:
                left = int(getattr(self, '_glue_hold_frames_left', 0))
                if left > 0:
                    setattr(self, '_glue_hold_frames_left', left - 1)
                    if delta < 0.0:
                        delta = 0.0
            if delta > step_up:
                delta = step_up
            elif delta < -step_dn:
                delta = -step_dn
            g_db = float(g_db + delta)
            # Ramp gain within frame
            L = j - i
            g_prev = float(10.0 ** (getattr(self, '_glue_gain_db', 0.0) / 20.0))
            g_cur = float(10.0 ** (g_db / 20.0))
            if L > 1:
                g_vec = np.linspace(g_prev, g_cur, num=L, dtype=np.float32)
                comp = seg.astype(np.float32, copy=False) * g_vec
            else:
                comp = seg.astype(np.float32, copy=False) * g_cur
            # Brightness on compressed path only
            if b_amt > 0.0:
                for k in range(L):
                    lp = lp + alpha * (comp[k] - lp)
                    hp = comp[k] - lp
                    comp[k] = comp[k] + 0.4 * b_amt * hp
            y[i:j] = (mix * comp + (1.0 - mix) * seg).astype(np.float32, copy=False)
            setattr(self, '_glue_gain_db', g_db)
            i = j
        if b_amt > 0.0:
            setattr(self, '_glue_lp_state', lp)
        # Gentle limiter safeguard
        if float(np.max(np.abs(y))) > 1.0:
            y = (np.tanh(1.2 * y) / np.tanh(1.2)).astype(np.float32, copy=False)
        return y

    def _init_webrtc_apm(self, sr: int) -> None:
        if not WEBRTC_AEC_AVAILABLE:
            return
        # Choose internal AEC processing rate (16 kHz is typical and fast)
        self._webrtc_sr = 16000
        self._webrtc_frame = int(self._webrtc_sr // 100)
        # Try common constructors across wrappers
        apm = None
        ctor_candidates = [
            getattr(ap, 'AudioProcessing', None),
            getattr(ap, 'WebRtcAudioProcessing', None),
            getattr(ap, 'Apm', None),
            getattr(ap, 'APM', None),
        ]
        for ctor in ctor_candidates:
            if ctor is None:
                continue
            try:
                # Try common kwargs
                try:
                    apm = ctor(enable_aec=True, use_ns=True, use_agc=False, use_hpf=True)
                except Exception:
                    try:
                        apm = ctor(aec=True, ns=True, agc=False, hpf=True)
                    except Exception:
                        apm = ctor()
                break
            except Exception:
                continue
        if apm is None:
            raise RuntimeError('Could not construct WebRTC AudioProcessing')
        # Configure formats if supported
        try:
            if hasattr(apm, 'set_stream_format'):
                apm.set_stream_format(self._webrtc_sr, 1)
            if hasattr(apm, 'set_reverse_stream_format'):
                apm.set_reverse_stream_format(self._webrtc_sr, 1)
        except Exception:
            pass
        # Enable features via attribute flags if present
        for name in ('echo_cancellation', 'aec', 'echo_canceller'):
            try:
                if hasattr(apm, name):
                    setattr(apm, name, True)
            except Exception:
                pass
        self._apm = apm

    def _webrtc_process(self, mono: np.ndarray) -> np.ndarray:
        """Apply WebRTC AEC using far-end ref from ref_q. Keeps length equal to input."""
        if self._apm is None or mono.size == 0:
            return mono
        # Resample near-end to APM rate and convert to int16
        x = App._resample_best(mono, int(self.sample_rate), int(self._webrtc_sr))  # type: ignore[attr-defined]
        x = np.clip(x, -1.0, 1.0)
        x_i16 = (x * 32767.0).astype(np.int16, copy=False)
        # Gather far-end reference and resample/convert
        try:
            ref = None
            while True:
                ref = self.ref_q.get_nowait()
        except queue.Empty:
            ref = None
        if isinstance(ref, np.ndarray) and ref.size:
            r = App._resample_best(ref.astype(np.float32, copy=False), int(self.sample_rate), int(self._webrtc_sr))  # type: ignore[attr-defined]
            r = np.clip(r, -1.0, 1.0)
            r_i16 = (r * 32767.0).astype(np.int16, copy=False)
            self._rev_buf = np.concatenate((self._rev_buf.astype(np.float32, copy=False), r.astype(np.float32, copy=False)))
        # Process in 10ms frames
        n = x_i16.size
        out_i16 = np.empty(n, dtype=np.int16)
        i = 0
        rev_i16_buf = None
        if self._rev_buf.size:
            # _rev_buf is already at _webrtc_sr; avoid no-op resample
            rev = np.clip(self._rev_buf, -1.0, 1.0).astype(np.float32, copy=False)
            rev_i16_buf = (rev * 32767.0).astype(np.int16, copy=False)
        rev_pos = 0
        while i < n:
            frame = x_i16[i:i+self._webrtc_frame]
            if frame.size < self._webrtc_frame:
                pad = np.zeros(self._webrtc_frame - frame.size, dtype=np.int16)
                frame = np.concatenate((frame, pad))
            # Feed reverse frame if available
            if rev_i16_buf is not None:
                rframe = rev_i16_buf[rev_pos:rev_pos+self._webrtc_frame]
                if rframe.size < self._webrtc_frame:
                    rpad = np.zeros(self._webrtc_frame - rframe.size, dtype=np.int16)
                    rframe = np.concatenate((rframe, rpad))
                try:
                    if hasattr(self._apm, 'process_reverse_stream'):
                        self._apm.process_reverse_stream(rframe)
                    elif hasattr(self._apm, 'process_reverse'):
                        self._apm.process_reverse(rframe)
                except Exception:
                    pass
                rev_pos += self._webrtc_frame
            # Process capture frame
            y = None
            try:
                if hasattr(self._apm, 'process_stream'):
                    y = self._apm.process_stream(frame)
                elif hasattr(self._apm, 'process_capture_stream'):
                    y = self._apm.process_capture_stream(frame)
            except Exception:
                y = None
            if y is None:
                y = frame
            out_i16[i:i+self._webrtc_frame] = y[:self._webrtc_frame]
            i += self._webrtc_frame
        # Save unused reverse samples
        if rev_i16_buf is not None and rev_pos < rev_i16_buf.size:
            leftover = rev_i16_buf[rev_pos:].astype(np.float32) / 32767.0
            self._rev_buf = leftover.astype(np.float32, copy=False)
        else:
            self._rev_buf = np.zeros(0, dtype=np.float32)
        # Convert back to float32 and resample to original rate
        y_f32 = (out_i16.astype(np.float32) / 32767.0)
        y_out = App._resample_best(y_f32, int(self._webrtc_sr), int(self.sample_rate))  # type: ignore[attr-defined]
        # Trim/pad to match original length
        if y_out.size != mono.size:
            if y_out.size > mono.size:
                y_out = y_out[:mono.size]
            else:
                pad = np.zeros(mono.size - y_out.size, dtype=np.float32)
                y_out = np.concatenate((y_out, pad))
        return y_out

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Denoiser X")
        self.root.geometry("640x360")
        self.root.minsize(560, 320)
        # Open maximized by default (Windows/Linux); ignore on platforms that don't support
        try:
            self.root.state('zoomed')
        except Exception:
            try:
                self.root.attributes('-zoomed', True)
            except Exception:
                pass

        # State: 'live' (monitor) or 'record' (write WAV)
        self.mode = tk.StringVar(value="live")
        # Processing: 'AI' (DeepFilterNet) or 'Normal' (pass-through)
        self.proc_mode = tk.StringVar(value=("AI" if DF_AVAILABLE else "Normal"))
        # DeepFilterNet model choice
        self.model_choice = tk.StringVar(value="DF3 (default)")
        self.model_dir = tk.StringVar(value="")
        # AEC mode
        self.aec_mode = tk.StringVar(value="Off")  # Off | Simple | WebRTC
        # AI stream preferences
        self.force_48k_var = tk.BooleanVar(value=False)
        # Low-latency (WASAPI) hint
        self.low_latency_var = tk.BooleanVar(value=False)
        # Driver/host API selection (Auto, WASAPI, WDM-KS, DirectSound, MME, ASIO)
        self.driver_var = tk.StringVar(value="Auto")
        # Disable EQ by default
        self.eq_enabled_var = tk.BooleanVar(value=False)
        # Config state
        self._config = {}
        self._config_path = os.path.join(os.getcwd(), "settings.json")
        self.file_path = tk.StringVar(value="recording.wav")
        self.level_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Idle")
        self.auto_info_var = tk.StringVar(value="SR: -, Block: -")
        self.stream = None
        self.writer = None
        self.writer_lock = threading.Lock()
        self.writer_closed = threading.Event()
        self.engine = None
        # Larger queues reduce dropouts under transient CPU spikes
        self.in_q = queue.Queue(maxsize=64)
        self.out_q = queue.Queue(maxsize=256)
        self.stop_event = threading.Event()
        self.worker = None
        self._last_out = None  # cache last output block for dropout smoothing
        self._pending_out = None  # leftover processed audio to use next callback
        # Auto-block runtime tuning
        self._auto_blocks: list[int] = []
        self._auto_block_index: int = 0
        self._retune_attempts: int = 0
        self._retune_scheduled: bool = False
        self._xrun_count: int = 0
        self._start_time: float = 0.0
        self._eq_after_id: int | None = None
        # Compressor defaults
        self.comp_enabled_var = tk.BooleanVar(value=True)
        self.comp_target_db_var = tk.DoubleVar(value=-20.0)
        self.comp_max_gain_db_var = tk.DoubleVar(value=12.0)
        self.comp_floor_db_var = tk.DoubleVar(value=-55.0)
        self.comp_floor_gain_db_var = tk.DoubleVar(value=6.0)
        self.comp_knee_db_var = tk.DoubleVar(value=10.0)
        self.comp_step_up_db_var = tk.DoubleVar(value=1.0)
        self.comp_step_down_db_var = tk.DoubleVar(value=2.0)

        # Sounddevice defaults tuned for stability
        try:
            sd.default.dtype = 'float32'
        except Exception:
            pass
        self._build_ui()
        self._refresh_devices()
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            pass

    # ---- Style ----
    def _apply_style(self):
        style = ttk.Style(self.root)
        # Prefer clam so our colors apply consistently, fallback to Windows themes
        for theme in ("clam", "vista", "xpnative"):
            try:
                style.theme_use(theme)
                break
            except Exception:
                continue
        # Containers
        style.configure("Root.TFrame", background=THEME["bg"]) 
        style.configure("Panel.TFrame", background=THEME["panel"], relief="flat")
        style.configure("Sunken.TFrame", background=THEME["sunken"], relief="flat")
        style.configure("Sep.TSeparator", background=THEME["stroke"], troughcolor=THEME["stroke"]) 
        # Typography
        style.configure("TLabel", background=THEME["panel"], foreground=THEME["text"], font=("Segoe UI", 10)) 
        style.configure("Title.TLabel", background=THEME["bg"], foreground=THEME["text"], font=("Segoe UI", 26, "bold"))
        # Make secondary text still readable on OLED
        style.configure("Sub.TLabel", background=THEME["bg"], foreground=THEME["muted"], font=("Segoe UI", 10))
        # Controls
        style.configure("TButton", background=THEME["panel"], foreground=THEME["text"], font=("Segoe UI", 10), relief="flat") 
        style.map("TButton", background=[("active", "#141414"), ("pressed", "#101010")],
                               foreground=[("disabled", "#9a9a9a")])
        style.configure("Accent.TButton", background=THEME["accent"], foreground="#111", font=("Segoe UI", 10, "bold"), relief="flat") 
        style.map("Accent.TButton", background=[("active", "#ffcf33"), ("pressed", "#e6b800")],
                                   foreground=[("disabled", "#9a9a9a")])
        style.configure("Ghost.TButton", background=THEME["panel"], foreground=THEME["muted"], font=("Segoe UI", 10), relief="flat") 
        style.configure("Bar.Horizontal.TProgressbar", troughcolor=THEME["sunken"], background=THEME["accent"], lightcolor=THEME["accent"], darkcolor=THEME["accent"])
        style.configure("Field.TEntry", fieldbackground=THEME["sunken"], background=THEME["sunken"], foreground=THEME["text"], insertcolor=THEME["text"], bordercolor=THEME["stroke"])
        style.configure("Segment.TRadiobutton", background=THEME["panel"], foreground=THEME["text"], font=("Segoe UI", 10)) 
        # Segmented button styles for a modern toggle look
        style.configure("Seg.TButton", background=THEME["panel"], foreground=THEME["text"], padding=(10,6), relief="flat")
        style.map("Seg.TButton", background=[("active", "#141414"), ("pressed", "#101010")])
        style.configure("SegSel.TButton", background=THEME["accent"], foreground="#111", padding=(10,6), relief="flat")
        style.configure("TCombobox", fieldbackground=THEME["sunken"], background=THEME["sunken"], foreground=THEME["text"], arrowcolor=THEME["text"], bordercolor=THEME["stroke"]) 

        # top stripe
        self.top_stripe = tk.Frame(self.root, height=4, bg=THEME["accent"])
        self.top_stripe.pack(fill="x", side="top")

    # ---- UI ----
    def _build_ui(self):
        # Apply theme before building widgets
        try:
            self._apply_style()
        except Exception:
            pass
        rootf = ttk.Frame(self.root, style="Root.TFrame")
        rootf.pack(fill="both", expand=True)

        # Header
        header = ttk.Frame(rootf, style="Root.TFrame")
        header.pack(fill="x", padx=20, pady=(14, 6))
        left_h = ttk.Frame(header, style="Root.TFrame"); left_h.pack(side="left", anchor="w")
        ttk.Label(left_h, text="Denoiser X", style="Title.TLabel").pack(anchor="w")
        right_h = ttk.Frame(header, style="Root.TFrame"); right_h.pack(side="right")
        self._status_dot = tk.Canvas(right_h, width=14, height=14, bg=THEME["bg"], highlightthickness=0)
        self._status_dot.pack(side="right", padx=(8, 0))
        self._status_dot_id = self._status_dot.create_oval(2, 2, 12, 12, fill="#60666f", outline="")
        ttk.Label(right_h, textvariable=self.status_var, style="TLabel").pack(side="right")

        # Main grid
        ttk.Separator(rootf, style="Sep.TSeparator").pack(fill="x", padx=20)
        grid = ttk.Frame(rootf, style="Root.TFrame")
        grid.pack(fill="both", expand=True, padx=20, pady=12)

        # Left panel inside a scrollable container to keep everything visible
        left_wrap = ttk.Frame(grid, style="Panel.TFrame")
        left_wrap.pack(side="left", fill="both", expand=True)
        try:
            self._left_canvas = tk.Canvas(left_wrap, bg=THEME["panel"], highlightthickness=0)
            left_scroll = ttk.Scrollbar(left_wrap, orient="vertical", command=self._left_canvas.yview)
            self._left_canvas.configure(yscrollcommand=left_scroll.set)
            left_scroll.pack(side="right", fill="y")
            self._left_canvas.pack(side="left", fill="both", expand=True)
            left = ttk.Frame(self._left_canvas, style="Panel.TFrame")
            self._left_win = self._left_canvas.create_window((0, 0), window=left, anchor="nw")
            def _on_left_config(event=None):
                try:
                    self._left_canvas.configure(scrollregion=self._left_canvas.bbox("all"))
                except Exception:
                    pass
            left.bind("<Configure>", _on_left_config)
            def _on_canvas_config(event=None):
                try:
                    w = int(self._left_canvas.winfo_width())
                    self._left_canvas.itemconfigure(self._left_win, width=w)
                except Exception:
                    pass
            self._left_canvas.bind("<Configure>", _on_canvas_config)
            # Enable mouse wheel on Windows
            def _on_wheel(e):
                try:
                    self._left_canvas.yview_scroll(-1 if e.delta > 0 else 1, 'units')
                except Exception:
                    pass
            self._left_canvas.bind_all("<MouseWheel>", _on_wheel)
        except Exception:
            # Fallback without scrolling
            left = ttk.Frame(grid, style="Panel.TFrame")
            left.pack(side="left", fill="both", expand=True)
        # Right sidebar (fixed width + scrollable for smaller screens)
        right_wrap = ttk.Frame(grid, style="Panel.TFrame")
        right_wrap.pack(side="left", fill="y", padx=(8, 0))
        try:
            right_wrap.update_idletasks()
            right_wrap.configure(width=380)
            right_wrap.pack_propagate(False)
        except Exception:
            pass
        try:
            self._right_canvas = tk.Canvas(right_wrap, bg=THEME["panel"], highlightthickness=0)
            right_scroll = ttk.Scrollbar(right_wrap, orient="vertical", command=self._right_canvas.yview)
            self._right_canvas.configure(yscrollcommand=right_scroll.set)
            right_scroll.pack(side="right", fill="y")
            self._right_canvas.pack(side="left", fill="both", expand=True)
            right = ttk.Frame(self._right_canvas, style="Panel.TFrame")
            self._right_win = self._right_canvas.create_window((0, 0), window=right, anchor="nw")
            def _on_right_config(event=None):
                try:
                    self._right_canvas.configure(scrollregion=self._right_canvas.bbox("all"))
                except Exception:
                    pass
            def _on_right_canvas(event=None):
                try:
                    w = int(self._right_canvas.winfo_width())
                    self._right_canvas.itemconfigure(self._right_win, width=w)
                except Exception:
                    pass
            right.bind("<Configure>", _on_right_config)
            self._right_canvas.bind("<Configure>", _on_right_canvas)
            # Mouse wheel support on Windows
            def _on_wheel_r(e):
                try:
                    self._right_canvas.yview_scroll(-1 if e.delta > 0 else 1, 'units')
                except Exception:
                    pass
            self._right_canvas.bind_all("<MouseWheel>", _on_wheel_r)
        except Exception:
            # Fallback without scrolling
            right = ttk.Frame(right_wrap, style="Panel.TFrame")
            right.pack(fill="y")

        # Left controls
        pad = {"padx": 12, "pady": 10}

        row0 = ttk.Frame(left, style="Panel.TFrame")
        row0.pack(fill="x", **pad)
        ttk.Label(row0, text="Microphone", style="TLabel").pack(anchor="w")
        self.in_sel = ttk.Combobox(row0, state="readonly")
        self.in_sel.pack(fill="x", pady=(6, 0))

        row1 = ttk.Frame(left, style="Panel.TFrame")
        row1.pack(fill="x", **pad)
        ttk.Label(row1, text="Mode", style="TLabel").pack(anchor="w")
        seg = ttk.Frame(row1, style="Panel.TFrame")
        seg.pack(fill="x", pady=(6, 0))
        self.seg_live = ttk.Button(seg, text="Live Output", style="SegSel.TButton" if self.mode.get()=="live" else "Seg.TButton", command=lambda: self._set_mode("live"))
        self.seg_record = ttk.Button(seg, text="Record to WAV", style="SegSel.TButton" if self.mode.get()=="record" else "Seg.TButton", command=lambda: self._set_mode("record"))
        self.seg_live.pack(side="left")
        self.seg_record.pack(side="left", padx=(6, 0))

        # Processing selection
        row1b = ttk.Frame(left, style="Panel.TFrame")
        row1b.pack(fill="x", **pad)
        ttk.Label(row1b, text="Processing", style="TLabel").pack(anchor="w")
        segp = ttk.Frame(row1b, style="Panel.TFrame")
        segp.pack(fill="x", pady=(6, 0))
        self.seg_ai = ttk.Button(segp, text="AI (DeepFilterNet)", style="SegSel.TButton" if self.proc_mode.get()=="AI" else "Seg.TButton", command=lambda: self._set_proc("AI"))
        self.seg_normal = ttk.Button(segp, text="Normal (Pass-through)", style="SegSel.TButton" if self.proc_mode.get()=="Normal" else "Seg.TButton", command=lambda: self._set_proc("Normal"))
        self.seg_ai.pack(side="left")
        self.seg_normal.pack(side="left", padx=(6, 0))

        # Model selection for AI
        row1c = ttk.Frame(left, style="Panel.TFrame")
        row1c.pack(fill="x", **pad)
        ttk.Label(row1c, text="AI Model", style="TLabel").pack(anchor="w")
        model_wrap = ttk.Frame(row1c, style="Panel.TFrame")
        model_wrap.pack(fill="x", pady=(6, 0))
        self.model_sel = ttk.Combobox(model_wrap, state="readonly", values=[
            "DF3 (default)", "DF2", "DF1", "Custom dir..."
        ])
        self.model_sel.set(self.model_choice.get())
        self.model_sel.pack(side="left", fill="x", expand=True)
        self.model_sel.bind("<<ComboboxSelected>>", lambda e=None: self._on_model_change())
        self.model_entry = ttk.Entry(model_wrap, textvariable=self.model_dir, width=24, style="Field.TEntry")
        self.model_browse = ttk.Button(model_wrap, text="Browse...", style="Ghost.TButton", command=self._browse_model_dir)
        # initial state
        self._on_model_change()

        # Driver selection row
        row1d = ttk.Frame(left, style="Panel.TFrame")
        row1d.pack(fill="x", **pad)
        ttk.Label(row1d, text="Driver", style="TLabel").pack(anchor="w")
        drv_wrap = ttk.Frame(row1d, style="Panel.TFrame")
        drv_wrap.pack(fill="x", pady=(6, 0))
        self.driver_combo = ttk.Combobox(drv_wrap, state="readonly", width=18)
        self._populate_driver_combo()
        self.driver_combo.pack(side="left")
        self.driver_combo.bind("<<ComboboxSelected>>", lambda e=None: self._on_driver_change())

        self.out_wrap = ttk.Frame(left, style="Panel.TFrame")
        self.out_label = ttk.Label(self.out_wrap, text="Output device", style="TLabel")
        self.out_sel = ttk.Combobox(self.out_wrap, state="readonly")

        self.file_wrap = ttk.Frame(left, style="Panel.TFrame")
        self.file_entry = ttk.Entry(self.file_wrap, style="Field.TEntry", textvariable=self.file_path)
        self.file_btn = ttk.Button(self.file_wrap, text="Browse…", style="Ghost.TButton", command=self._browse_file)

        row2 = ttk.Frame(left, style="Panel.TFrame")
        row2.pack(fill="x", **pad)
        self.start_btn = ttk.Button(row2, text="Start", style="Accent.TButton", command=self.start)
        self.stop_btn = ttk.Button(row2, text="Stop", style="Ghost.TButton", command=self.stop, state="disabled")
        self.retune_btn = ttk.Button(row2, text="Retune Now", style="Ghost.TButton", command=self._retune_now, state="disabled")
        self.start_btn.pack(side="left")
        self.stop_btn.pack(side="left", padx=8)
        self.retune_btn.pack(side="left")

        advanced = ttk.Frame(left, style="Panel.TFrame")
        advanced.pack(fill="x", **pad)
        # SR / Block
        ttk.Label(advanced, text="Sample rate (auto)", style="TLabel").pack(side="left")
        self.sr_entry = ttk.Entry(advanced, width=8, style="Field.TEntry")
        # Prefer 'auto' as default for robust device compatibility
        self.sr_entry.insert(0, "auto")
        self.sr_entry.pack(side="left", padx=(6, 16))
        ttk.Label(advanced, text="Block (auto)", style="TLabel").pack(side="left")
        self.block_entry = ttk.Entry(advanced, width=6, style="Field.TEntry")
        # Default to a large block (20000) per request
        self.block_entry.insert(0, "20000")
        self.block_entry.pack(side="left", padx=(6, 16))
        # Force 48k for AI
        self.force_48k_chk = ttk.Checkbutton(advanced, text="Force 48 kHz (AI)", variable=self.force_48k_var)
        self.force_48k_chk.pack(side="left", padx=(0, 12))
        # Low latency mode (WASAPI)
        self.lowlat_chk = ttk.Checkbutton(advanced, text="Low Latency (WASAPI)", variable=self.low_latency_var)
        self.lowlat_chk.pack(side="left", padx=(0, 12))
        # (Removed: No Auto Retune toggle)
        # Attenuation slider
        self.atten_var = tk.DoubleVar(value=ATTN_DB)
        ttk.Label(advanced, text="Attenuation (dB)", style="TLabel").pack(side="left")
        self.atten_scale = ttk.Scale(advanced, from_=0, to=100, orient="horizontal", length=200, variable=self.atten_var, command=lambda _=None: self._on_atten_change())
        self.atten_scale.pack(side="left", padx=(6, 6))
        self.atten_val_lbl = ttk.Label(advanced, text=f"{int(self.atten_var.get())} dB", style="TLabel")
        self.atten_val_lbl.pack(side="left", padx=(0, 12))
        # (Removed: Gain and Echo controls)

        # Right panel: meter + status + tri-bar EQ
        rpad = {"padx": 14, "pady": 12}
        meterf = ttk.Frame(right, style="Panel.TFrame")
        meterf.pack(fill="x", **rpad)
        ttk.Label(meterf, text="Levels", style="TLabel").pack(anchor="w")
        self.meter = ttk.Progressbar(meterf, style="Bar.Horizontal.TProgressbar", orient="horizontal", length=220, mode="determinate", maximum=0.25, variable=self.level_var)
        self.meter.pack(fill="x", pady=(6, 0))

        # Lightweight 3s level history (audio reactive, low CPU)
        histf = ttk.Frame(right, style="Panel.TFrame")
        histf.pack(fill="x", padx=14, pady=(6, 8))
        ttk.Label(histf, text="Level History (3s)", style="TLabel").pack(anchor="w")
        self.hist_height = 70
        self.hist_canvas = tk.Canvas(histf, height=self.hist_height, bg=THEME["sunken"], highlightthickness=0)
        self.hist_canvas.pack(fill="x", pady=(6,0))
        # History buffer and timers
        self._hist_window_sec = 3.0
        self._hist_vals: deque[tuple[float,float]] = deque()  # (timestamp, rms)
        self._hist_after_id = None
        self._hist_last_draw = 0.0
        try:
            self.hist_canvas.bind('<Configure>', self._on_hist_resize)
        except Exception:
            pass
        ttk.Separator(right, style="Sep.TSeparator").pack(fill="x", padx=14, pady=(6,4))
        self.status = ttk.Label(right, textvariable=self.status_var, style="TLabel")
        self.status.pack(fill="x", **rpad)
        self.auto_info = ttk.Label(right, textvariable=self.auto_info_var, style="TLabel")
        self.auto_info.pack(fill="x", padx=14, pady=(0, 8))

        # Diagnostics
        diagbar = ttk.Frame(right, style="Panel.TFrame"); diagbar.pack(fill="x", padx=14)
        ttk.Button(diagbar, text="Diagnostics...", style="Ghost.TButton", command=self._open_diagnostics).pack(side="left")
        # (Glue compressor removed)
        self._diag = []  # list of dicts for attempts/errors

        # (VST section removed for lightweight build)

        # Optional Tri-bar EQ canvas (disabled by default)
        self.eq_canvas = None
        if self.eq_enabled_var.get():
            self.eq_width, self.eq_height = 360, 180
            self.eq_canvas = tk.Canvas(right, height=self.eq_height, bg=THEME["sunken"], highlightthickness=0)
            self.eq_canvas.pack(fill="x", padx=12, pady=(0,12))
            try:
                self.eq_canvas.bind('<Configure>', self._on_eq_resize)
            except Exception:
                pass
            # EQ state
            self._eq_n = 256
            self._eq_win = np.hanning(self._eq_n).astype(np.float32)
            self._eq_vals = {"low": 0.0, "mid": 0.0, "high": 0.0}
            self._eq_max = 1e-6  # for auto scaling
            self._eq_last_t = 0.0

        # Load config and apply
        try:
            self._config = self._load_config()
        except Exception:
            self._config = {}
        try:
            self._apply_loaded_config()
        except Exception:
            pass
        self._on_mode_change()
        # kick redraw loop if EQ enabled
        if self.eq_enabled_var.get() and self.eq_canvas is not None:
            try:
                self.root.after(80, self._redraw_eq)
            except Exception:
                pass

        # Batch processing panel
        batch = ttk.Frame(rootf, style="Root.TFrame")
        batch.pack(fill="x", padx=20, pady=(6, 12))
        ttk.Label(batch, text="Batch AI Processing", style="TLabel").pack(anchor="w")
        self.batch_list = tk.Listbox(batch, height=4, bg=THEME["sunken"], fg=THEME["text"], highlightthickness=0, selectmode=tk.EXTENDED)
        self.batch_list.pack(fill="x", pady=(6, 6))
        bar = ttk.Frame(batch, style="Root.TFrame")
        bar.pack(fill="x")
        ttk.Button(bar, text="Add files...", style="Ghost.TButton", command=self._batch_add).pack(side="left")
        ttk.Button(bar, text="Clear", style="Ghost.TButton", command=self._batch_clear).pack(side="left", padx=(8,0))
        ttk.Button(bar, text="Process AI", style="Accent.TButton", command=self._batch_process).pack(side="right")

        # Footer
        ttk.Separator(rootf, style="Sep.TSeparator").pack(fill="x", padx=20, pady=(6, 0))
        footer = ttk.Frame(rootf, style="Root.TFrame")
        footer.pack(fill="x", padx=20, pady=(6, 12))
        ttk.Label(footer, text="by howtocuddle", style="Sub.TLabel").pack(side="left")

    def _set_running(self, running: bool):
        try:
            self._status_dot.itemconfig(self._status_dot_id, fill=("#29d11e" if running else "#60666f"))
        except Exception:
            pass

    def _on_close(self):
        """Gracefully stop audio/threads and close the window."""
        try:
            self.stop()
        except Exception:
            pass
        # Save config on exit
        try:
            self._save_config()
        except Exception:
            pass
        # Cancel UI timers
        try:
            if getattr(self, '_eq_after_id', None) is not None:
                self.root.after_cancel(self._eq_after_id)  # type: ignore
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # ---- Devices ----
    def _refresh_devices(self):
        def _fmt(d):
            return f"[{d.get('index', -1)}] {d.get('name','Unknown')}"
        try:
            devices = sd.query_devices()
            apis = sd.query_hostapis()
        except Exception:
            devices = []
            apis = []
        in_devs = []
        out_devs = []
        sel_driver = (self.driver_var.get() if hasattr(self, 'driver_var') else 'Auto').strip()
        sel_driver_norm = self._norm_api_name(sel_driver)
        for idx, d in enumerate(devices):
            d = dict(d)
            d["index"] = idx
            d["hostapi_name"] = apis[d["hostapi"]]["name"] if apis else "API"
            d_api_norm = self._norm_api_name(d.get("hostapi_name", ""))
            if sel_driver_norm and sel_driver_norm != 'auto' and d_api_norm != sel_driver_norm:
                continue
            # Input (microphones): must have input channels, pass name filter, and pass settings check
            if d.get("max_input_channels", 0) > 0 and self._looks_like_mic(d.get('name', '')):
                if self._device_is_usable(idx, for_input=True):
                    in_devs.append(d)
            # Output: list only outputs that can be opened
            if d.get("max_output_channels", 0) > 0:
                if self._device_is_usable(idx, for_input=False):
                    out_devs.append(d)
        self.in_devices = in_devs
        self.out_devices = out_devs
        self.in_sel.configure(values=[_fmt(d) for d in self.in_devices])
        if self.in_devices:
            self.in_sel.current(0)
        self.out_sel.configure(values=[_fmt(d) for d in self.out_devices])
        if self.out_devices:
            self.out_sel.current(0)
        # Update driver combo to show available APIs
        try:
            self._populate_driver_combo()
        except Exception:
            pass
        # Set UI sample rate to Windows default for selected devices
        try:
            in_idx = self._parse_device_index(self.in_sel)
            out_idx = self._parse_device_index(self.out_sel)
            def_sr = None
            if self.mode.get() == "live" and out_idx is not None:
                try:
                    def_sr = int(sd.query_devices(out_idx).get('default_samplerate') or 0)
                except Exception:
                    def_sr = None
            if def_sr is None and in_idx is not None:
                try:
                    def_sr = int(sd.query_devices(in_idx).get('default_samplerate') or 0)
                except Exception:
                    def_sr = None
            if isinstance(def_sr, int) and def_sr > 0:
                self.sr_entry.delete(0, tk.END)
                self.sr_entry.insert(0, str(def_sr))
        except Exception:
            pass

    def _parse_device_index(self, combo: ttk.Combobox) -> int | None:
        sel = combo.get().strip()
        if not sel.startswith("["):
            return None
        try:
            return int(sel.split("]")[0][1:])
        except Exception:
            return None

    def _looks_like_mic(self, name: str) -> bool:
        n = (name or '').lower()
        # Exclude common render/loopback/virtual output names appearing as inputs
        bad_tokens = [
            'loopback', 'stereo mix', 'what u hear', 'mix', 'speaker', 'speakers',
            'headphone', 'headphones', 'line out', 'digital audio', 'spdif', 'hdmi',
            'output', 'vb-audio', 'vb cable', 'cable output', 'cable input (vb-audio',
        ]
        if any(tok in n for tok in bad_tokens):
            return False
        # Otherwise treat as usable mic
        return True

    # ---- Config persistence ----
    def _load_config(self) -> dict:
        import json
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _apply_loaded_config(self):
        c = self._config or {}
        try:
            self.mode.set(c.get('mode', self.mode.get()))
            self.proc_mode.set(c.get('proc_mode', self.proc_mode.get()))
            self.driver_var.set(c.get('driver', self.driver_var.get()))
            self.force_48k_var.set(bool(c.get('force_48k', self.force_48k_var.get())))
            self.low_latency_var.set(bool(c.get('low_latency', self.low_latency_var.get())))
            self.eq_enabled_var.set(bool(c.get('eq_enabled', self.eq_enabled_var.get())))
            fp = c.get('file_path');  self.file_path.set(fp if fp else self.file_path.get())
            # Compressor
            self.comp_enabled_var.set(bool(c.get('comp_enabled', self.comp_enabled_var.get())))
            self.comp_target_db_var.set(float(c.get('comp_target_db', self.comp_target_db_var.get())))
            self.comp_max_gain_db_var.set(float(c.get('comp_max_gain_db', self.comp_max_gain_db_var.get())))
            self.comp_floor_db_var.set(float(c.get('comp_floor_db', self.comp_floor_db_var.get())))
            self.comp_floor_gain_db_var.set(float(c.get('comp_floor_gain_db', self.comp_floor_gain_db_var.get())))
            self.comp_knee_db_var.set(float(c.get('comp_knee_db', self.comp_knee_db_var.get())))
            self.comp_step_up_db_var.set(float(c.get('comp_step_up_db', self.comp_step_up_db_var.get())))
            self.comp_step_down_db_var.set(float(c.get('comp_step_down_db', self.comp_step_down_db_var.get())))
            # (Glue simple controls removed)
        except Exception:
            pass
        try:
            sr_text = c.get('sr_text', None)
            if sr_text is not None:
                self.sr_entry.delete(0, tk.END); self.sr_entry.insert(0, str(sr_text))
        except Exception:
            pass
        try:
            block_text = c.get('block_text', None)
            if block_text is not None:
                self.block_entry.delete(0, tk.END); self.block_entry.insert(0, str(block_text))
        except Exception:
            pass
        # Refresh devices and restore selection
        try:
            self._refresh_devices()
            in_idx = c.get('in_device_index'); out_idx = c.get('out_device_index')
            if isinstance(in_idx, int):
                for i, d in enumerate(self.in_devices):
                    if d.get('index') == in_idx:
                        self.in_sel.current(i)
                        break
            if isinstance(out_idx, int):
                for i, d in enumerate(self.out_devices):
                    if d.get('index') == out_idx:
                        self.out_sel.current(i)
                        break
        except Exception:
            pass
        # VST chain loading removed

    # (VST chain UI helpers removed)

    def _save_config(self):
        import json
        cfg = {}
        try:
            cfg['mode'] = self.mode.get()
            cfg['proc_mode'] = self.proc_mode.get()
            cfg['driver'] = self.driver_var.get()
            cfg['force_48k'] = bool(self.force_48k_var.get())
            cfg['low_latency'] = bool(self.low_latency_var.get())
            # removed: no_auto_retune
            cfg['eq_enabled'] = bool(self.eq_enabled_var.get())
            cfg['file_path'] = self.file_path.get()
            cfg['sr_text'] = self.sr_entry.get()
            cfg['block_text'] = self.block_entry.get()
            try:
                in_idx = self._parse_device_index(self.in_sel); out_idx = self._parse_device_index(self.out_sel)
            except Exception:
                in_idx = None; out_idx = None
            cfg['in_device_index'] = in_idx
            cfg['out_device_index'] = out_idx
            # Compressor
            cfg['comp_enabled'] = bool(self.comp_enabled_var.get())
            cfg['comp_target_db'] = float(self.comp_target_db_var.get())
            cfg['comp_max_gain_db'] = float(self.comp_max_gain_db_var.get())
            cfg['comp_floor_db'] = float(self.comp_floor_db_var.get())
            cfg['comp_floor_gain_db'] = float(self.comp_floor_gain_db_var.get())
            cfg['comp_knee_db'] = float(self.comp_knee_db_var.get())
            cfg['comp_step_up_db'] = float(self.comp_step_up_db_var.get())
            cfg['comp_step_down_db'] = float(self.comp_step_down_db_var.get())
            # (Glue simple controls removed)
        # VST chain path removed
        except Exception:
            pass
        try:
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def _device_is_usable(self, index: int, for_input: bool) -> bool:
        try:
            info = sd.query_devices(index)
        except Exception:
            return False
        try:
            sr = int(info.get('default_samplerate') or 48000)
            if sr <= 0:
                sr = 48000
        except Exception:
            sr = 48000
        # Build extra settings only for WASAPI, and use shared for broad compatibility
        extra = None
        try:
            host_name = sd.query_hostapis()[info['hostapi']]['name']
        except Exception:
            host_name = ''
        if self._norm_api_name(self.driver_var.get()) == 'wasapi' or self._norm_api_name(host_name) == 'wasapi':
            try:
                extra = sd.WasapiSettings(exclusive=False, low_latency=False)
            except Exception:
                extra = None
        try:
            if for_input:
                # Use 1 channel for probe
                sd.check_input_settings(device=index, samplerate=sr, channels=1, dtype='float32', extra_settings=extra)
            else:
                # Try 2 channels if available, else 1
                ch = 2 if int(info.get('max_output_channels', 0)) >= 2 else 1
                sd.check_output_settings(device=index, samplerate=sr, channels=ch, dtype='float32', extra_settings=extra)
            return True
        except Exception:
            return False

    def _norm_api_name(self, name: str) -> str:
        n = (name or '').strip().lower()
        if not n:
            return ''
        if 'asio' in n:
            return 'asio'
        if 'wasapi' in n:
            return 'wasapi'
        if 'wdm' in n or 'wdm-ks' in n or 'kernel streaming' in n or 'ks' in n:
            return 'wdm-ks'
        if 'directsound' in n or 'direct sound' in n or 'dx' in n:
            return 'directsound'
        if 'mme' in n or 'wave' in n:
            return 'mme'
        if 'auto' in n:
            return 'auto'
        # strip 'windows '
        return n.replace('windows ', '').strip()

    def _populate_driver_combo(self):
        if not hasattr(self, 'driver_combo'):
            return
        try:
            apis = sd.query_hostapis()
        except Exception:
            apis = []
        seen: set[str] = set()
        order = ['wasapi', 'wdm-ks', 'directsound', 'mme', 'asio']
        names_map = {
            'wasapi': 'WASAPI',
            'wdm-ks': 'WDM-KS',
            'directsound': 'DirectSound (DX)',
            'mme': 'MME (Wave)',
            'asio': 'ASIO',
        }
        avail: list[str] = []
        for a in apis:
            norm = self._norm_api_name(a.get('name', ''))
            if norm and norm in order and norm not in seen:
                seen.add(norm)
        for k in order:
            if k in seen:
                avail.append(names_map.get(k, k.upper()))
        values = ['Auto'] + avail
        try:
            self.driver_combo.configure(values=values)
            cur = self.driver_var.get()
            if cur not in values:
                self.driver_var.set('Auto')
                self.driver_combo.set('Auto')
            else:
                self.driver_combo.set(cur)
        except Exception:
            pass

    def _on_driver_change(self):
        try:
            choice = self.driver_combo.get().strip()
            self.driver_var.set(choice or 'Auto')
        except Exception:
            self.driver_var.set('Auto')
        # Refresh lists to show only devices for this API
        try:
            self._refresh_devices()
        except Exception:
            pass

    def _on_mode_change(self):
        for w in (self.out_wrap, self.file_wrap):
            try:
                w.pack_forget()
            except Exception:
                pass
        if self.mode.get() == "live":
            self.out_wrap.pack(fill="x", padx=12, pady=(4, 0))
            self.out_label.pack(anchor="w")
            self.out_sel.pack(fill="x", pady=(6, 0))
        else:
            self.file_wrap.pack(fill="x", padx=12, pady=(4, 0))
            ttk.Label(self.file_wrap, text="WAV file", style="TLabel").pack(anchor="w")
            self.file_entry.pack(fill="x", pady=(6, 6))
            self.file_btn.pack(anchor="w")

    def _set_mode(self, mode: str):
        if mode not in ("live", "record"):
            return
        try:
            self.mode.set(mode)
            # update segmented styles
            self.seg_live.configure(style="SegSel.TButton" if mode=="live" else "Seg.TButton")
            self.seg_record.configure(style="SegSel.TButton" if mode=="record" else "Seg.TButton")
            self._on_mode_change()
        except Exception:
            pass

    def _set_proc(self, mode: str):
        if mode not in ("AI", "Normal"):
            return
        if mode == "AI" and not DF_AVAILABLE:
            messagebox.showwarning("AI not available", "DeepFilterNet is not installed. Using Normal mode.")
            mode = "Normal"
        try:
            self.proc_mode.set(mode)
            self.seg_ai.configure(style="SegSel.TButton" if mode=="AI" else "Seg.TButton")
            self.seg_normal.configure(style="SegSel.TButton" if mode=="Normal" else "Seg.TButton")
        except Exception:
            pass

    def _schedule_retune(self):
        if (self._retune_scheduled or self._retune_attempts >= MAX_RETUNE_ATTEMPTS):
            return
        self._retune_scheduled = True
        try:
            self.root.after(0, self._retune_block)
        except Exception:
            # Best-effort: try immediate
            self._retune_block()

    def _retune_block(self):
        try:
            # Build candidate list for current SR and context
            sr = int(getattr(self, 'current_sr', 48000))
            user_block = 0
            try:
                t = (self.block_entry.get() or '').strip().lower()
                user_block = int(t) if t and t not in ('auto','0') else 0
            except Exception:
                user_block = 0
            cands = self._build_block_candidates(sr, user_block)
            # Find current block and move to next
            try:
                cur_block = int(self.stream.blocksize) if self.stream is not None else 0
            except Exception:
                cur_block = 0
            if cur_block in cands:
                idx = cands.index(cur_block)
                next_idx = min(len(cands)-1, idx + 1)
            else:
                next_idx = 0
            next_block = cands[next_idx] if cands else 0
            # Update UI entry to lock next block and restart
            try:
                self.block_entry.delete(0, tk.END)
                self.block_entry.insert(0, str(max(1, next_block)))
            except Exception:
                pass
            self._retune_attempts += 1
            self.stop()
            self.start()
        except Exception:
            # Ignore retune failures
            pass
        finally:
            self._retune_scheduled = False

    def _build_block_candidates(self, sr: int, user_block: int) -> list[int]:
        out: list[int] = []
        def _add(x):
            xi = int(max(0, x))
            if xi not in out:
                out.append(xi)
        if user_block and user_block > 0:
            _add(user_block)
        ten = max(1, int(sr // 100))  # 10 ms
        # Prefer more stable sizes first: ~20 ms, then ~10 ms
        _add(max(1, ten*2))  # ~20 ms
        _add(ten)            # ~10 ms
        if int(sr) == 48000:
            _add(960)       # 20 ms at 48k
            _add(480)       # 10 ms at 48k
        _add(1024)
        _add(2048)
        _add(512)
        _add(0)            # driver default auto
        return out

    def _retune_now(self):
        # Manual retune: attempt next candidate immediately
        if self.stream is None:
            try:
                messagebox.showinfo("Retune", "Start audio first, then Retune.")
            except Exception:
                pass
            return
        try:
            self.status_var.set("Retuning block size...")
        except Exception:
            pass
        try:
            # Allow manual retune regardless of auto attempt limits
            self._retune_scheduled = False
            self._retune_block()
        except Exception:
            pass

    # (VST3 post-processing helpers removed)

    def _on_atten_change(self):
        try:
            val = float(self.atten_var.get())
        except Exception:
            val = ATTN_DB
        # Update running engine if present
        try:
            eng = None
            if getattr(self, 'engine', None) is not None:
                eng = self.engine
            elif getattr(self, 'worker', None) is not None and getattr(self.worker, 'engine', None) is not None:
                eng = self.worker.engine
            if eng is not None and hasattr(eng, 'set_atten'):
                eng.set_atten(val)
        except Exception:
            pass
        # Update label
        try:
            if hasattr(self, 'atten_val_lbl') and self.atten_val_lbl is not None:
                self.atten_val_lbl.configure(text=f"{int(round(val))} dB")
        except Exception:
            pass

    def _on_gain_change(self):
        try:
            g = float(self.gain_var.get())
        except Exception:
            g = 0.0
        try:
            if self.worker is not None:
                self.worker.gain_lin = float(10 ** (g / 20.0))
        except Exception:
            pass

    def _on_model_change(self):
        try:
            choice = self.model_sel.get()
        except Exception:
            choice = self.model_choice.get()
        self.model_choice.set(choice)
        # Toggle custom dir widgets
        try:
            if choice == "Custom dir...":
                self.model_entry.pack(side="left", padx=(6, 6))
                self.model_browse.pack(side="left")
            else:
                self.model_entry.pack_forget()
                self.model_browse.pack_forget()
        except Exception:
            pass

    def _browse_model_dir(self):
        try:
            p = filedialog.askdirectory(title="Select DeepFilterNet model folder (contains checkpoints/ and config.ini)")
        except Exception:
            p = ""
        if p:
            self.model_dir.set(p)

    def _on_aec_change(self):
        # Echo/AEC removed; always Off
        self.aec_mode.set("Off")

    def _load_engine_background(self, model_dir: str | None):
        try:
            self.status_var.set("Loading AI model...")
        except Exception:
            pass
        engine = None
        try:
            engine = DFEngine(model_dir=model_dir)
            try:
                engine.set_atten(float(self.atten_var.get()))
            except Exception:
                pass
        except Exception as e:
            def _warn():
                try:
                    messagebox.showerror("DeepFilterNet error", f"Falling back to Normal mode.\n{e}")
                except Exception:
                    pass
            try:
                self.root.after(0, _warn)
            except Exception:
                pass
            engine = None
        # Apply to running worker if still active
        def _apply():
            try:
                if engine is not None:
                    # keep a reference so UI controls (atten) can update it
                    self.engine = engine
                    if self.worker is not None:
                        self.worker.engine = engine
                    self.status_var.set("AI ready")
            except Exception:
                pass
        try:
            self.root.after(0, _apply)
        except Exception:
            _apply()

    def _browse_file(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile=self.file_path.get() or "enhanced.wav",
            title="Save enhanced audio as",
        )
        if p:
            self.file_path.set(p)

    # ---- Start/Stop ----
    def start(self):
        if self.stream is not None:
            return
        # Parse SR/BLOCK with 'auto' support
        sr_text = (self.sr_entry.get() or '').strip().lower()
        block_text = (self.block_entry.get() or '').strip().lower()
        # User-specified values (None/0 means auto)
        user_sr = None
        if sr_text not in ('', 'auto', '0'):
            try:
                t = sr_text.replace('hz','').strip()
                if t.endswith('k'):
                    user_sr = int(float(t[:-1]) * 1000)
                else:
                    user_sr = int(float(t))
            except Exception:
                user_sr = None
        try:
            user_block = 0 if block_text in ('', 'auto', '0') else int(block_text)
        except Exception:
            user_block = 0
        # Fallbacks used only as last resort
        sr = SR
        block = 0

        in_idx = self._parse_device_index(self.in_sel)
        if in_idx is None:
            messagebox.showerror("No input", "Please choose a microphone.")
            return

        self.stop_event.clear()
        self.in_q = queue.Queue(maxsize=16)
        self.out_q = queue.Queue(maxsize=128)
        self.ref_q = None  # echo reference disabled (AI-only)

        # Prepare engine depending on processing mode (async for AI to avoid UI stalls)
        self.engine = None
        if self.proc_mode.get() == "AI":
            # Resolve model selection
            model_dir = None
            try:
                choice = self.model_choice.get()
            except Exception:
                choice = "DF3 (default)"
            if choice == "DF2":
                model_dir = "DeepFilterNet2"
            elif choice == "DF1":
                model_dir = "DeepFilterNet"
            elif choice == "Custom dir...":
                model_dir = (self.model_dir.get() or "").strip() or None
            # Spawn loader thread
            try:
                self._engine_thread = threading.Thread(target=self._load_engine_background, args=(model_dir,), daemon=True)
                self._engine_thread.start()
            except Exception:
                # As last resort, we keep running in Normal
                pass

        self.writer = None
        self.writer_closed.clear()

        def audio_callback(indata, outdata, frames, time_info, status):
            try:
                # Fast exit if stopping
                if self.stop_event.is_set():
                    if outdata is not None:
                        outdata.fill(0)
                    return
                if status:
                    # Count xruns and schedule retune if they persist shortly after start
                    try:
                        self._xrun_count += 1
                        if (
                            (time.perf_counter() - self._start_time) < 5.0
                            and self._xrun_count >= 5
                            and self._retune_attempts < MAX_RETUNE_ATTEMPTS
                        ):
                            self._schedule_retune()
                    except Exception:
                        pass
                # Ensure mono (compute in float32 to avoid upcasting to float64)
                if indata.ndim == 1:
                    in_mono = indata.astype(np.float32, copy=False)
                else:
                    in_mono = indata.mean(axis=1, dtype=np.float32)
                self.in_q.put_nowait(in_mono)
            except queue.Full:
                pass
            except Exception:
                # Never raise from callback
                if outdata is not None:
                    try:
                        outdata.fill(0)
                    except Exception:
                        pass
                return

            if self.mode.get() == "live":
                # Fill output with queued processed audio; handle leftovers across callbacks
                if outdata is not None:
                    outdata.fill(0)
                    remain = frames
                    out_ch = outdata.shape[1] if outdata.ndim == 2 else 1
                    # Use pending leftovers first
                    if isinstance(self._pending_out, np.ndarray) and self._pending_out.size > 0:
                        use = min(remain, self._pending_out.shape[0])
                        blk = self._pending_out[:use]
                        if out_ch == 2:
                            outdata[:use, 0] = blk
                            outdata[:use, 1] = blk
                        else:
                            outdata[:use, 0] = blk
                        self._last_out = blk.astype(np.float32, copy=False)
                        remain -= use
                        if use < self._pending_out.shape[0]:
                            self._pending_out = self._pending_out[use:]
                        else:
                            self._pending_out = None
                    # Drain queue while we have space
                    while remain > 0:
                        try:
                            processed = self.out_q.get_nowait()
                        except queue.Empty:
                            break
                        if processed.ndim != 1:
                            processed = processed.reshape(-1)
                        use = min(remain, processed.shape[0])
                        if out_ch == 2:
                            outdata[frames-remain:frames-remain+use, 0] = processed[:use]
                            outdata[frames-remain:frames-remain+use, 1] = processed[:use]
                        else:
                            outdata[frames-remain:frames-remain+use, 0] = processed[:use]
                        self._last_out = processed[:use].astype(np.float32, copy=False)
                        remain -= use
                        # keep leftover for next callback
                        if use < processed.shape[0]:
                            self._pending_out = processed[use:]
                            break
                    # If still short and we have last_out, hold it in low-latency mode
                    if remain > 0:
                        if isinstance(self._last_out, np.ndarray) and self._last_out.size > 0:
                            # choose hold vs fade depending on low-latency mode
                            if bool(self.low_latency_var.get()):
                                # Repeat last block to avoid gaps
                                repeat = min(remain, self._last_out.shape[0])
                                if out_ch == 2:
                                    outdata[frames-remain:frames-remain+repeat, 0] = self._last_out[:repeat]
                                    outdata[frames-remain:frames-remain+repeat, 1] = self._last_out[:repeat]
                                else:
                                    outdata[frames-remain:frames-remain+repeat, 0] = self._last_out[:repeat]
                            else:
                                # Quick fade to zero
                                length = min(remain, self._last_out.shape[0])
                                fade = np.linspace(1.0, 0.0, num=length, dtype=np.float32)
                                if out_ch == 2:
                                    outdata[frames-remain:frames-remain+length, 0] = self._last_out[:length] * fade
                                    outdata[frames-remain:frames-remain+length, 1] = self._last_out[:length] * fade
                                else:
                                    outdata[frames-remain:frames-remain+length, 0] = self._last_out[:length] * fade
                        # else leave zeros
                # (echo reference disabled)
            else:
                if outdata is not None:
                    outdata.fill(0)

        if self.mode.get() == "live":
            out_idx = self._parse_device_index(self.out_sel)
            if out_idx is None:
                messagebox.showerror("No output", "Choose an output device or switch to Record.")
                return
            device = (in_idx, out_idx)

            # Decide output channels: prefer stereo if supported
            try:
                outinfo = sd.query_devices(out_idx)
                out_ch = 2 if int(outinfo.get('max_output_channels', 1)) >= 2 else 1
            except Exception:
                out_ch = 1

            # Host API specific extra settings
            extra = None
            try:
                out_host = ''
                try:
                    out_host = sd.query_hostapis()[sd.query_devices(out_idx)['hostapi']]['name']
                except Exception:
                    out_host = ''
                if self._norm_api_name(self.driver_var.get()) == 'wasapi' or self._norm_api_name(out_host) == 'wasapi':
                    extra = sd.WasapiSettings(exclusive=bool(self.low_latency_var.get()), low_latency=bool(self.low_latency_var.get()))
            except Exception:
                extra = None

            # Robust combinations: prefer Windows defaults, allow user override, AI at 48k if forced
            errors = []
            opened = False
            # If specific driver selected, set default hostapi to nudge PortAudio
            try:
                drv_norm = self._norm_api_name(self.driver_var.get())
                if drv_norm and drv_norm != 'auto':
                    for i, ha in enumerate(sd.query_hostapis()):
                        if self._norm_api_name(ha.get('name','')) == drv_norm:
                            sd.default.hostapi = i
                            break
            except Exception:
                pass
            # Determine Windows default sample rates for both devices
            try:
                indevinfo = sd.query_devices(in_idx)
                def_sr_in = int(indevinfo.get('default_samplerate') or 0)
            except Exception:
                def_sr_in = 0
            try:
                outdevinfo = sd.query_devices(out_idx)
                def_sr_out = int(outdevinfo.get('default_samplerate') or 0)
            except Exception:
                def_sr_out = 0
            ai = self.proc_mode.get() == "AI"
            sr_candidates: list[int] = []
            # Forced 48k for AI
            if ai and self.force_48k_var.get():
                sr_candidates.append(48000)
            # If AI and defaults differ, try 48k early to avoid double resampling
            if ai and not self.force_48k_var.get() and def_sr_in > 0 and def_sr_out > 0 and def_sr_in != def_sr_out:
                if 48000 not in sr_candidates:
                    sr_candidates.append(48000)
            # If both defaults match, prefer that
            if def_sr_in > 0 and def_sr_in == def_sr_out and def_sr_in not in sr_candidates:
                sr_candidates.append(def_sr_in)
            # User-specified rate first if provided and not forced otherwise
            if user_sr and int(user_sr) not in sr_candidates:
                sr_candidates.append(int(user_sr))
            # Fill remaining common practical rates
            for cand in (def_sr_in, def_sr_out, 48000, 44100):
                if isinstance(cand, int) and cand > 0 and int(cand) not in sr_candidates:
                    sr_candidates.append(int(cand))
            # Rebuild SR candidates with output-first preference to avoid invalid combos
            sr_candidates = []
            def _add_sr(x):
                xi = int(x)
                if xi > 0 and xi not in sr_candidates:
                    sr_candidates.append(xi)
            # Prefer output default, then input default
            if def_sr_out > 0:
                _add_sr(def_sr_out)
            if def_sr_in > 0:
                _add_sr(def_sr_in)
            # AI needs 48k sooner to avoid double resampling
            if ai:
                _add_sr(48000)
            # User
            if user_sr:
                _add_sr(int(user_sr))
            # Common fallback
            _add_sr(44100)
            # Build blocksize candidates (prefer ~10-20 ms, DF-aligned) and persist for retune
            block_candidates = self._build_block_candidates(int(sr_candidates[0]), int(user_block) if isinstance(user_block, int) else 0)
            # Only attempt the exact user-selected input/output pair; avoid (None, out)
            dev_candidates = [device]
            for rate in sr_candidates:
                for bs in block_candidates:
                    for dev in dev_candidates:
                        # Attempt with fallbacks for WASAPI exclusive
                        attempts = []
                        base = dict(samplerate=int(rate), dtype="float32", channels=(1, out_ch), callback=audio_callback, device=dev)
                        # primary attempt
                        a1 = dict(base)
                        a1["blocksize"] = int(bs)
                        if extra is not None:
                            a1["extra_settings"] = extra
                        if bool(self.low_latency_var.get()) and self._norm_api_name(self.driver_var.get()) == 'wasapi':
                            a1["latency"] = 'low'
                        attempts.append(a1)
                        # WASAPI exclusive fallback to driver-chosen blocksize
                        is_wasapi_excl = False
                        try:
                            is_wasapi_excl = isinstance(extra, sd.WasapiSettings) and bool(getattr(extra, 'exclusive', False))
                        except Exception:
                            is_wasapi_excl = False
                        if is_wasapi_excl:
                            a2 = dict(base)
                            a2["blocksize"] = 0
                            a2["extra_settings"] = extra
                            a2["latency"] = 'low'
                            attempts.append(a2)
                            # fallback to shared
                            try:
                                shared = sd.WasapiSettings(exclusive=False, low_latency=True)
                                a3 = dict(base)
                                a3["blocksize"] = int(bs)
                                a3["extra_settings"] = shared
                                attempts.append(a3)
                            except Exception:
                                pass
                        success = False
                        for kw in attempts:
                            try:
                                self.stream = sd.Stream(**kw)
                                sr = int(rate)
                                block = int(kw.get("blocksize", 0))
                                opened = True
                                success = True
                                self._log_attempt('live', sr, block, dev, 'ok')
                                break
                            except Exception as e:
                                msg = f"{e}"
                                errors.append(f"sr={rate}, block={kw.get('blocksize')}, dev={dev}: {msg}")
                                self._log_attempt('live', rate, int(kw.get('blocksize') or 0), dev, 'err', msg)
                        if success:
                            break
                    if opened:
                        break
                if opened:
                    break
            if not opened:
                messagebox.showerror("Audio error", "Cannot open audio stream:\n" + "\n".join(map(str, errors[-5:])))
                self.stream = None
                return
        else:
            # Record-only: use InputStream (input device only)
            indev = in_idx
            extra = None
            try:
                in_host = ''
                try:
                    in_host = sd.query_hostapis()[sd.query_devices(indev)['hostapi']]['name']
                except Exception:
                    in_host = ''
                if self._norm_api_name(self.driver_var.get()) == 'wasapi' or self._norm_api_name(in_host) == 'wasapi':
                    extra = sd.WasapiSettings(exclusive=bool(self.low_latency_var.get()), low_latency=bool(self.low_latency_var.get()))
            except Exception:
                extra = None
            errors = []
            opened = False
            # Prefer Windows default(s), honor user input, AI 48k if forced
            try:
                indevinfo = sd.query_devices(indev)
                def_sr = int(indevinfo.get('default_samplerate') or 0)
            except Exception:
                def_sr = 0
            sr_candidates: list[int] = []
            ai = self.proc_mode.get() == "AI"
            if ai and self.force_48k_var.get():
                sr_candidates.append(48000)
            elif ai:
                for cand in (user_sr or 0, 48000, def_sr):
                    if isinstance(cand, int) and cand > 0 and int(cand) not in sr_candidates:
                        sr_candidates.append(int(cand))
            else:
                for cand in (user_sr or 0, def_sr, 48000, 44100):
                    if isinstance(cand, int) and cand > 0 and int(cand) not in sr_candidates:
                        sr_candidates.append(int(cand))
            block_candidates = self._build_block_candidates(int(sr_candidates[0]), int(user_block) if isinstance(user_block, int) else 0)
            for rate in sr_candidates:
                for bs in block_candidates:
                    for dev in (indev, None):
                        attempts = []
                        base = dict(samplerate=int(rate), dtype="float32", channels=1, device=dev,
                                    callback=lambda indata, frames, time_info, status: audio_callback(indata, None, frames, time_info, status))
                        a1 = dict(base)
                        a1["blocksize"] = int(bs)
                        if extra is not None:
                            a1["extra_settings"] = extra
                        if bool(self.low_latency_var.get()) and self._norm_api_name(self.driver_var.get()) == 'wasapi':
                            a1["latency"] = 'low'
                        attempts.append(a1)
                        # WASAPI exclusive: try driver default blocksize 0, then shared fallback
                        is_wasapi_excl = False
                        try:
                            is_wasapi_excl = isinstance(extra, sd.WasapiSettings) and bool(getattr(extra, 'exclusive', False))
                        except Exception:
                            is_wasapi_excl = False
                        if is_wasapi_excl:
                            a2 = dict(base)
                            a2["blocksize"] = 0
                            a2["extra_settings"] = extra
                            a2["latency"] = 'low'
                            attempts.append(a2)
                            try:
                                shared = sd.WasapiSettings(exclusive=False, low_latency=True)
                                a3 = dict(base)
                                a3["blocksize"] = int(bs)
                                a3["extra_settings"] = shared
                                attempts.append(a3)
                            except Exception:
                                pass
                        success = False
                        for kw in attempts:
                            try:
                                self.stream = sd.InputStream(**kw)
                                sr = int(rate)
                                block = int(kw.get("blocksize", 0))
                                opened = True
                                success = True
                                self._log_attempt('record', sr, block, dev, 'ok')
                                break
                            except Exception as e:
                                msg = f"{e}"
                                errors.append(f"sr={rate}, block={kw.get('blocksize')}, dev={dev}: {msg}")
                                self._log_attempt('record', rate, int(kw.get('blocksize') or 0), dev, 'err', msg)
                        if success:
                            break
                    if opened:
                        break
                if opened:
                    break
            if not opened:
                messagebox.showerror("Audio error", "Cannot open input stream:\n" + "\n".join(map(str, errors[-5:])))
                return

        # If recording, open the WAV writer now using the actual stream sample rate
        if self.mode.get() == "record":
            p = Path(self.file_path.get().strip() or "enhanced.wav")
            if p.suffix.lower() != ".wav":
                p = p.with_suffix(".wav")
            try:
                if str(p.parent) and not p.parent.exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            try:
                self.writer = sf.SoundFile(str(p), mode="w", samplerate=int(sr), channels=CHANNELS_OUT, subtype="PCM_16")
            except Exception as e:
                messagebox.showerror("File error", f"Cannot open WAV for writing:\n{e}")
                # Close opened stream before exiting
                try:
                    if self.stream is not None:
                        self.stream.stop(); self.stream.close()
                except Exception:
                    pass
                self.stream = None
                return

        # No VST board
        vst_board = None
        vst_after_ai = True

        self.worker = AudioWorker(self.engine, self.in_q, self.out_q, self.writer,
                                      level_callback=self._on_level,
                                      error_callback=self._on_worker_error,
                                      stop_event=self.stop_event,
                                      writer_lock=self.writer_lock,
                                      writer_closed=self.writer_closed,
                                      echo_enabled=(self.aec_mode.get() in ("Simple", "WebRTC")),
                                      aec_mode=self.aec_mode.get(),
                                      ref_q=None,
                                      sample_rate=int(sr),
                                      gain_db=float(self.gain_var.get()) if hasattr(self, 'gain_var') else 6.0,
                                      vst_board=vst_board,
                                      vst_after_ai=vst_after_ai,
                                      low_latency_mode=bool(self.low_latency_var.get()),
                                      vst_bypass=True,
                                      vst_wet=1.0,
                                      spec_callback=(lambda mono: self._on_eq(mono)) if (self.eq_enabled_var.get()) else None)
        self.worker.start()

        try:
            self.stream.start()
        except Exception as e:
            messagebox.showerror("Audio error", f"Failed to start stream:\n{e}")
            self.stop()
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.current_sr = int(sr)
        # Show resolved auto parameters on right panel
        try:
            mode_tag = "AI" if self.proc_mode.get()=="AI" else "Normal"
            f48 = " • Forced 48k" if (self.proc_mode.get()=="AI" and self.force_48k_var.get()) else ""
            self.auto_info_var.set(f"Auto • {mode_tag}{f48} • SR: {int(sr)} Hz  |  Block: {block or 'auto'}")
        except Exception:
            pass
        # Precompute EQ band indices for current SR
        try:
            n = self._eq_n
            sr_i = max(1, int(self.current_sr))
            freqs = np.fft.rfftfreq(n, d=1.0/sr_i)
            self._eq_idx_low = np.where((freqs >= 0) & (freqs < 250))[0]
            self._eq_idx_mid = np.where((freqs >= 250) & (freqs < 4000))[0]
            self._eq_idx_high = np.where(freqs >= 4000)[0]
        except Exception:
            self._eq_idx_low = self._eq_idx_mid = self._eq_idx_high = None
        self.status_var.set(f"Running @ {sr/1000:.1f} kHz | block {block or 'auto'} | mode: {self.mode.get()} | proc: {self.proc_mode.get()}")
        self._set_running(True)
        try:
            self.retune_btn.configure(state="normal")
        except Exception:
            pass
        self._xrun_count = 0
        self._retune_scheduled = False
        self._start_time = time.perf_counter()

    def stop(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_var.set("Stopping…")
        try:
            # Cancel UI timers safely
            try:
                if getattr(self, '_eq_after_id', None) is not None:
                    self.root.after_cancel(self._eq_after_id)  # type: ignore
            except Exception:
                pass
            try:
                if getattr(self, '_hist_after_id', None) is not None:
                    self.root.after_cancel(self._hist_after_id)  # type: ignore
            except Exception:
                pass
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.stream = None
        try:
            # Signal worker to stop and prevent further writes
            self.stop_event.set()
            self.writer_closed.set()
            if self.worker is not None:
                # Wait for worker to finish (try a bit longer)
                self.worker.join(timeout=2.0)
                if self.worker.is_alive():
                    self.worker.join(timeout=3.0)
        except Exception:
            pass
        self.worker = None
        try:
            if self.writer is not None:
                with self.writer_lock:
                    try:
                        self.writer.flush()
                    except Exception:
                        pass
                    try:
                        self.writer.close()
                    except Exception:
                        pass
        except Exception:
            pass
        self.writer = None
        # Join background engine loader if present
        try:
            t = getattr(self, '_engine_thread', None)
            if t is not None and isinstance(t, threading.Thread) and t.is_alive():
                t.join(timeout=1.0)
        except Exception:
            pass
        self.status_var.set("Idle")
        try:
            self._set_running(False)
        except Exception:
            pass
        # Reset retune state
        self._xrun_count = 0
        self._retune_scheduled = False
        try:
            self.retune_btn.configure(state="disabled")
        except Exception:
            pass
        self._set_running(False)

    def _on_level(self, rms_val: float):
        def _upd():
            # Update instantaneous meter
            v = max(0.0, min(0.25, float(rms_val)))
            self.level_var.set(v)
            # Append to 3s history and schedule a lightweight redraw
            try:
                now = time.perf_counter()
                self._hist_vals.append((now, v))
                # Drop old samples beyond window
                cutoff = now - float(self._hist_window_sec)
                while self._hist_vals and self._hist_vals[0][0] < cutoff:
                    self._hist_vals.popleft()
                if self._hist_after_id is None:
                    self._hist_after_id = self.root.after(60, self._redraw_hist)
            except Exception:
                pass
        try:
            self.root.after(0, _upd)
        except Exception:
            pass

    def _on_worker_error(self, msg: str):
        def _upd():
            try:
                self.status_var.set(msg)
                messagebox.showerror("Processing error", msg)
            except Exception:
                pass
        try:
            self.root.after(0, _upd)
        except Exception:
            pass

    # ---- 3s Level History (lightweight) ----
    def _on_hist_resize(self, event=None):
        try:
            # Trigger a redraw on resize
            if self._hist_after_id is None:
                self._hist_after_id = self.root.after(0, self._redraw_hist)
        except Exception:
            pass

    def _redraw_hist(self):
        try:
            self._hist_after_id = None
            c = getattr(self, 'hist_canvas', None)
            if c is None:
                return
            c.delete('all')
            try:
                w = max(20, int(c.winfo_width()))
                h = max(20, int(c.winfo_height()))
            except Exception:
                w, h = 360, int(self.hist_height)
            # Draw baseline
            c.create_line(1, h-1, w-1, h-1, fill=THEME['stroke'])
            if not self._hist_vals:
                return
            # Map last 3s of values to canvas coordinates
            now = time.perf_counter()
            t0 = now - float(self._hist_window_sec)
            vmax = 0.25  # matches meter maximum
            # Build polyline points
            pts = []
            for (t, v) in list(self._hist_vals):
                if t < t0:
                    continue
                x = 1 + (t - t0) / float(self._hist_window_sec) * (w - 2)
                vv = max(0.0, min(vmax, float(v))) / (vmax + 1e-9)
                y = (h - 1) - vv * (h - 2)
                pts.extend([x, y])
            if len(pts) >= 4:
                c.create_line(*pts, fill=THEME['accent'], width=2)
        except Exception:
            pass

    # ---- Diagnostics ----
    def _log_attempt(self, kind: str, rate: int, block: int, dev: tuple | int | None, result: str, err: str | None = None):
        try:
            entry = {
                'time': time.strftime('%H:%M:%S'),
                'kind': kind,
                'driver': self.driver_var.get(),
                'sr': int(rate),
                'block': int(block),
                'latency': 'low' if self.low_latency_var.get() else 'normal',
                'dev': str(dev),
                'result': result,
            }
            if err:
                entry['error'] = err
            self._diag.append(entry)
            if len(self._diag) > 200:
                self._diag = self._diag[-200:]
        except Exception:
            pass

    def _open_diagnostics(self):
        top = tk.Toplevel(self.root)
        top.title('Diagnostics')
        top.geometry('680x420')
        frm = ttk.Frame(top, style='Panel.TFrame'); frm.pack(fill='both', expand=True)
        txt = tk.Text(frm, bg=THEME['sunken'], fg=THEME['text'], insertbackground=THEME['text'])
        txt.pack(fill='both', expand=True, padx=8, pady=8)
        btns = ttk.Frame(frm, style='Panel.TFrame'); btns.pack(fill='x')
        def _refresh():
            try:
                txt.delete('1.0', tk.END)
                for e in self._diag:
                    line = f"[{e.get('time')}] {e.get('kind'):6} drv={e.get('driver'):<8} sr={e.get('sr'):<6} block={e.get('block'):<6} lat={e.get('latency'):<6} dev={e.get('dev')} -> {e.get('result')}\n"
                    if e.get('error'):
                        line += f"    error: {e.get('error')}\n"
                txt.insert(tk.END, line)
            except Exception:
                pass
        ttk.Button(btns, text='Refresh', style='Ghost.TButton', command=_refresh).pack(side='left')
        def _copy():
            try:
                data = txt.get('1.0', tk.END)
                self.root.clipboard_clear()
                self.root.clipboard_append(data)
            except Exception:
                pass
        ttk.Button(btns, text='Copy', style='Ghost.TButton', command=_copy).pack(side='left', padx=(6,0))
        _refresh()

    def _open_glue(self):
        pass

    def _apply_comp_changes(self):
        try:
            if self.worker is not None:
                self.worker._comp_enabled = bool(self.comp_enabled_var.get())
                self.worker._comp_target_db = float(self.comp_target_db_var.get())
                self.worker._comp_max_gain_db = float(self.comp_max_gain_db_var.get())
                self.worker._comp_floor_db = float(self.comp_floor_db_var.get())
                self.worker._comp_max_gain_floor_db = float(self.comp_floor_gain_db_var.get())
                self.worker._comp_knee_db = float(self.comp_knee_db_var.get())
                self.worker._comp_step_up_db = float(self.comp_step_up_db_var.get())
                self.worker._comp_step_down_db = float(self.comp_step_down_db_var.get())
        except Exception:
            pass

    def _apply_glue_preset(self):
        pass

    # Info button removed per request

    # ---- Batch Processing ----
    def _batch_add(self):
        paths = filedialog.askopenfilenames(title="Select audio files", filetypes=[("Audio", "*.wav;*.flac;*.mp3;*.ogg;*.m4a;*.aac"), ("All", "*.*")])
        for p in paths:
            try:
                self.batch_list.insert(tk.END, p)
            except Exception:
                pass

    def _batch_clear(self):
        try:
            self.batch_list.delete(0, tk.END)
        except Exception:
            pass

    def _batch_process(self):
        try:
            files = list(self.batch_list.get(0, tk.END))
        except Exception:
            files = []
        if not files:
            messagebox.showinfo("Batch", "Add some files first.")
            return
        t = threading.Thread(target=self._batch_worker, args=(files,), daemon=True)
        t.start()

    # ---- VST UI handlers ----
    def _vst_add(self):
        pass

    def _vst_remove(self):
        pass

    def _vst_reload(self):
        pass

    def _vst_get_chain(self) -> dict:
        return {}

    def _vst_apply_params(self, plugins: list, chain: dict):
        pass

    def _vst_save_chain(self):
        return
        import json
        try:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], initialfile="vst_chain.json", title="Save VST chain")
        except Exception:
            path = ""
        if not path:
            return
        chain = self._vst_get_chain()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(chain, f, indent=2)
            self.status_var.set(f"Saved VST chain: {Path(path).name}")
        except Exception as e:
            try:
                messagebox.showerror("Save error", str(e))
            except Exception:
                pass

    def _vst_load_chain(self):
        return
        import json
        try:
            path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")], title="Load VST chain")
        except Exception:
            path = ""
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                chain = json.load(f)
        except Exception as e:
            try:
                messagebox.showerror("Load error", str(e))
            except Exception:
                pass
            return
        # Apply list and toggles
        try:
            self.vst_after_ai_var.set(bool(chain.get('after_ai', True)))
            self.vst_bypass_var.set(bool(chain.get('bypass', False)))
            self.vst_wet_var.set(float(chain.get('wet', 1.0)))
        except Exception:
            pass
        try:
            self.vst_list.delete(0, tk.END)
            for item in chain.get('plugins', []):
                p = item.get('path')
                if p:
                    self.vst_list.insert(tk.END, p)
        except Exception:
            pass
        # Save params to apply when starting
        self._vst_chain_params = chain
        try:
            self.status_var.set(f"Loaded VST chain: {Path(path).name}")
        except Exception:
            pass

    def _on_vst_bypass_change(self):
        pass

    def _on_vst_wet_change(self):
        pass

    def _vst_controls(self):
        pass
        """
        # Prefer live plugins from the running worker
        plugs = None
        try:
            if self.worker is not None and getattr(self.worker, 'vst_board', None) is not None:
                try:
                    plugs = list(self.worker.vst_board)
                except Exception:
                    plugs = getattr(self, '_vst_plugins', None)
            else:
                plugs = getattr(self, '_vst_plugins', None)
        except Exception:
            plugs = getattr(self, '_vst_plugins', None)
        if not plugs:
            # Attempt to load plugins from the current list for UI-only editing
            try:
                paths = list(self.vst_list.get(0, tk.END))
            except Exception:
                paths = []
            tmp = []
            for p in paths:
                try:
                    tmp.append(_pb_load(p))
                except Exception:
                    continue
            if tmp:
                plugs = tmp
                self._vst_plugins = tmp
                self._vst_plugin_paths = paths
        top = tk.Toplevel(self.root)
        top.title("VST Controls")
        top.geometry("420x520")
        frm = ttk.Frame(top, style="Panel.TFrame"); frm.pack(fill="both", expand=True)
        canv = tk.Canvas(frm, bg=THEME["panel"], highlightthickness=0)
        scr = ttk.Scrollbar(frm, orient="vertical", command=canv.yview)
        wrap = ttk.Frame(canv, style="Panel.TFrame")
        canv.configure(yscrollcommand=scr.set)
        scr.pack(side="right", fill="y")
        canv.pack(side="left", fill="both", expand=True)
        canv.create_window((0,0), window=wrap, anchor="nw")
        def _cfg(_=None):
            try:
                canv.configure(scrollregion=canv.bbox("all"))
            except Exception:
                pass
        wrap.bind("<Configure>", _cfg)

        def add_param_row(parent, plugin, pname, pobj):
            row = ttk.Frame(parent, style="Panel.TFrame"); row.pack(fill="x", padx=8, pady=4)
            ttk.Label(row, text=pname, style="TLabel").pack(anchor="w")
            # Value & range
            v0 = 0.0
            vmin = 0.0
            vmax = 1.0
            try:
                v0 = float(getattr(pobj, 'value', 0.0))
            except Exception:
                pass
            for attr in ('min_value','minimum','min'):
                try:
                    vmin = float(getattr(pobj, attr))
                    break
                except Exception:
                    continue
            for attr in ('max_value','maximum','max'):
                try:
                    vmax = float(getattr(pobj, attr))
                    break
                except Exception:
                    continue
            if vmax <= vmin:
                vmax = vmin + 1.0
            var = tk.DoubleVar(value=v0)
            def _on_change(_=None):
                val = float(var.get())
                try:
                    if hasattr(pobj, 'value'):
                        setattr(pobj, 'value', val)
                except Exception:
                    pass
            s = ttk.Scale(row, from_=vmin, to=vmax, orient="horizontal", length=300, variable=var, command=lambda _=None: _on_change())
            s.pack(fill="x")
            return row

        # Build controls
        params_count = 0
        if plugs:
        for idx, plug in enumerate(plugs or [], 1):
            sect = ttk.Frame(wrap, style="Panel.TFrame"); sect.pack(fill="x", pady=(6,8))
            title = getattr(plug, 'name', None) or plug.__class__.__name__
                ttk.Label(sect, text=f"[{idx}] {title}", style="TLabel").pack(anchor="w", padx=8)
                params = getattr(plug, 'parameters', None)
                if isinstance(params, dict) and params:
                    for pname, pobj in params.items():
                        add_param_row(sect, plug, str(pname), pobj)
                        params_count += 1
                else:
                    ttk.Label(sect, text="(No parameters exposed)", style="Sub.TLabel").pack(anchor="w", padx=12)
        else:
            ttk.Label(wrap, text="Add VST3 plugins first (right panel → Add .vst3...).",
                      style="Sub.TLabel").pack(anchor="w", padx=12, pady=8)
        if plugs and params_count == 0:
            ttk.Label(wrap, text="Plugins loaded but no automatable parameters were found.",
                      style="Sub.TLabel").pack(anchor="w", padx=12, pady=6)
        """

    def _batch_worker(self, files: list[str]):
        use_ai = self.proc_mode.get() == "AI" and DF_AVAILABLE
        engine = None
        if use_ai:
            self.status_var.set("Batch: loading model...")
            try:
                # Resolve model selection similar to live Start()
                model_dir = None
                choice = self.model_choice.get()
                if choice == "DF2":
                    model_dir = "DeepFilterNet2"
                elif choice == "DF1":
                    model_dir = "DeepFilterNet"
                elif choice == "Custom dir...":
                    model_dir = (self.model_dir.get() or "").strip() or None
                engine = DFEngine(model_dir=model_dir)
                # Apply current attenuation from the slider
                try:
                    engine.set_atten(float(self.atten_var.get()))
                except Exception:
                    pass
            except Exception as e:
                messagebox.showerror("DeepFilterNet error", f"Batch falling back to Normal mode.\n{e}")
                use_ai = False
        for i, path in enumerate(files, 1):
            try:
                mode_lab = "AI" if use_ai else "Normal"
                self.status_var.set(f"Processing {i}/{len(files)} [{mode_lab}]: {Path(path).name}")
                self._batch_enhance_file(engine, path)
            except Exception as e:
                messagebox.showwarning("Batch file error", f"{path}: {e}")
        self.status_var.set("Batch done")

    @staticmethod
    def _resample_best(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        # Delegate to shared module for consistent quality and reuse
        try:
            return _resample_best_mod(x, int(sr_in), int(sr_out))
        except Exception:
            # Fallback: simple linear interpolation
            x = x.astype(np.float32, copy=False)
            if int(sr_in) == int(sr_out) or x.size == 0:
                return x
            n_out = int(round(x.size * (sr_out / sr_in)))
            if n_out <= 0:
                return np.zeros(0, dtype=np.float32)
            xp = np.linspace(0, 1, num=x.size, endpoint=False)
            xq = np.linspace(0, 1, num=n_out, endpoint=False)
            return np.interp(xq, xp, x)

    def _batch_enhance_file(self, engine: DFEngine | None, path: str):
        data, sr_in = sf.read(path, dtype='float32', always_2d=False)
        # Mono mix if needed
        if isinstance(data, np.ndarray) and data.ndim == 2:
            data = data.mean(axis=1)
        if not isinstance(data, np.ndarray):
            data = np.array([], dtype=np.float32)
        use_ai = engine is not None
        # Decide working sample rate
        out_sr = 48000 if use_ai else int(sr_in)
        x = self._resample_best(data, int(sr_in), out_sr)
        # (echo suppression removed in batch)
        # (no pre-processing)

        if use_ai:
            # One-shot enhance to avoid block boundary artifacts
            y = engine.process(x)
            out = y[: x.size].astype(np.float32, copy=False)
            out = np.clip(out, -1.0, 1.0)
            out_path = str(Path(path).with_suffix('')) + "_enhanced.wav"
        else:
            out = np.clip(x.astype(np.float32, copy=False), -1.0, 1.0)
            out_path = str(Path(path).with_suffix('')) + "_processed.wav"

        # (No post-AI VST in batch)
        with sf.SoundFile(out_path, mode='w', samplerate=out_sr, channels=1, subtype='PCM_16') as wf:
            wf.write(out)

    @staticmethod
    def _try_load_ref(path: str, target_sr: int) -> np.ndarray | None:
        stem = Path(path)
        cand = stem.with_name(stem.stem + "_ref.wav")
        if not cand.exists():
            return None
        try:
            ref, sr = sf.read(str(cand), dtype='float32', always_2d=False)
            if isinstance(ref, np.ndarray) and ref.ndim == 2:
                ref = ref.mean(axis=1)
            if not isinstance(ref, np.ndarray):
                return None
            if int(sr) != int(target_sr):
                ref = App._resample_best(ref, int(sr), int(target_sr))  # type: ignore[attr-defined]
            return ref.astype(np.float32, copy=False)
        except Exception:
            return None

    # ---- Tri‑bar EQ drawing ----
    def _on_eq(self, mono: np.ndarray):
        if mono is None or mono.size == 0:
            return
        # Throttle EQ updates to ~80ms
        now = time.perf_counter()
        if now - getattr(self, '_eq_last_t', 0.0) < 0.08:
            return
        self._eq_last_t = now
        n = self._eq_n
        if mono.size < n:
            x = np.empty(n, dtype=np.float32)
            x[:mono.size] = mono
            x[mono.size:] = 0.0
        else:
            x = mono[:n]
        # FFT magnitude (256-point FFT)
        spec = np.abs(np.fft.rfft(x * self._eq_win))
        # Use precomputed band indices if available
        if hasattr(self, '_eq_idx_low') and isinstance(self._eq_idx_low, np.ndarray):
            low = spec[self._eq_idx_low]
            mid = spec[self._eq_idx_mid]
            high = spec[self._eq_idx_high]
        else:
            sr = getattr(self, 'current_sr', 48000)
            freqs = np.fft.rfftfreq(n, d=1.0/max(1, int(sr)))
            low = spec[(freqs >= 0) & (freqs < 250)]
            mid = spec[(freqs >= 250) & (freqs < 4000)]
            high = spec[(freqs >= 4000)]
        lv = float(np.log1p(low.mean() if low.size else 0.0))
        mv = float(np.log1p(mid.mean() if mid.size else 0.0))
        hv = float(np.log1p(high.mean() if high.size else 0.0))
        self._eq_max = max(self._eq_max * 0.98, max(lv, mv, hv, 1e-6))
        self._eq_vals["low"] = lv
        self._eq_vals["mid"] = mv
        self._eq_vals["high"] = hv

    def _on_eq_resize(self, event):
        try:
            self.eq_width = max(100, int(getattr(event, 'width', self.eq_width)))
            self.eq_height = max(80, int(getattr(event, 'height', self.eq_height)))
        except Exception:
            pass

    def _redraw_eq(self):
        try:
            c = self.eq_canvas
            c.delete("all")
            # Use actual canvas size if realized
            try:
                w = int(c.winfo_width()) or int(self.eq_width)
                h = int(c.winfo_height()) or int(self.eq_height)
            except Exception:
                w, h = int(self.eq_width), int(self.eq_height)
            # Normalize values
            m = self._eq_max + 1e-9
            vals = [self._eq_vals.get("low", 0.0)/m, self._eq_vals.get("mid", 0.0)/m, self._eq_vals.get("high", 0.0)/m]
            labels = ["LOW", "MID", "HIGH"]
            # Responsive layout
            margin = max(8, int(w * 0.04))
            gap = max(8, int(w * 0.04))
            bars = 3
            bar_w = max(8, (w - 2*margin - (bars-1)*gap) // bars)
            x0 = margin
            for i, v in enumerate(vals):
                bh = int(max(2, v) * max(2, (h - 24)))
                bx0 = x0 + i * (bar_w + gap)
                bx1 = min(w - margin, bx0 + bar_w)
                color = self._level_color(v, i)
                c.create_rectangle(bx0, h - bh - 8, bx1, h - 8, outline="", fill=color)
                c.create_text((bx0+bx1)//2, h - 4, text=labels[i], fill=THEME["text"], font=("Segoe UI", 9))
            self._eq_after_id = self.root.after(100, self._redraw_eq)
        except Exception:
            try:
                self._eq_after_id = self.root.after(100, self._redraw_eq)
            except Exception:
                pass

    def _level_color(self, v: float, band: int = 0) -> str:
        """Map a 0..1 level to a dynamic per-band gradient.
        LOW:  green→yellow→white, MID: orange→red→white, HIGH: blue→cyan→white."""
        try:
            vv = max(0.0, min(1.0, float(v)))

            def grad(vf: float, stops: list[tuple[float, tuple[int,int,int]]]) -> tuple[int,int,int]:
                if vf <= stops[0][0]:
                    return stops[0][1]
                if vf >= stops[-1][0]:
                    return stops[-1][1]
                for i in range(1, len(stops)):
                    p0, c0 = stops[i-1]
                    p1, c1 = stops[i]
                    if vf <= p1:
                        t = (vf - p0) / max(1e-6, (p1 - p0))
                        r = int(c0[0] + (c1[0] - c0[0]) * t)
                        g = int(c0[1] + (c1[1] - c0[1]) * t)
                        b = int(c0[2] + (c1[2] - c0[2]) * t)
                        return r, g, b
                return stops[-1][1]

            bidx = int(band)
            if bidx == 0:
                # LOW: green -> yellow -> white
                stops = [(0.0, (0,128,0)), (0.5, (255,200,0)), (1.0, (255,255,255))]
            elif bidx == 1:
                # MID: orange -> red -> white
                stops = [(0.0, (255,165,0)), (0.5, (255,64,0)), (1.0, (255,255,255))]
            else:
                # HIGH: blue -> cyan -> white
                stops = [(0.0, (0,64,192)), (0.5, (0,200,255)), (1.0, (255,255,255))]
            r, g, b = grad(vv, stops)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            # Fallback static colors if anything goes wrong
            fallback = ("#33cc33", "#ffd319", "#66ccff")
            return fallback[int(band) % 3]


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()


