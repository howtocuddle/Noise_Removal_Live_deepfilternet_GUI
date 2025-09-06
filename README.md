https://youtu.be/uvYLFvmYZLI?si=z_hWj_wEuWqz0pIu

Denoiser X — Live Mic Noise Filtering (DeepFilterNet)
====================================================

Overview
- Denoiser X is a Windows desktop app for real‑time mic enhancement and batch audio cleanup.
- It can run in two modes:
  - AI (DeepFilterNet): strong noise suppression at high quality.
  - Normal: pass‑through with level normalization (no AI).

System Requirements
- Windows 10/11, 64‑bit
- Audio devices using WASAPI/DirectSound/MME
- Microsoft Visual C++ Redistributable 2015–2022 (x64) — required by Torch/libdf

Getting Started (Standalone)
1) Download the packaged folder (dist\DenoiserX\) and unzip.
2) Double‑click DenoiserX.exe.
3) No installs required — the DeepFilterNet model and dependencies are bundled.

Main Concepts
- Processing: choose AI (DeepFilterNet) or Normal (pass‑through).
- Mode: Live Output (monitor to speakers/headphones) or Record to WAV (save enhanced mic).
- Auto Info (right panel): shows actual Sample Rate and Block in use.
- Tri‑bar EQ: shows LOW/MID/HIGH energy with dynamic colors.

Live Usage
1) Pick your microphone from “Microphone”.
2) Choose Mode:
   - Live Output: also choose an output device (speakers/headphones).
   - Record to WAV: choose/save the file path.
3) Processing:
   - AI (DeepFilterNet) — default. For best results, keep “Force 48 kHz (AI)” ON.
   - Normal — no AI; pass‑through with level normalization.
4) Advanced:
   - Sample rate / Block: leave “auto” unless you need a specific value.
   - Attenuation (dB): 0–100, higher = stronger suppression (AI only). Adjust live.
   - Echo: Off / Simple (beta). Simple uses far‑end playback as a reference.
   - Force 48 kHz (AI): opens stream at 48k to avoid resampling and ensure stability.
5) Click Start. Use Retune Now if you hear dropouts — it tries a new block size.
6) Click Stop to end.

Recording
- In “Record to WAV”, the app writes the enhanced/normalized mic to your chosen .wav path.
- Sample rate and block selection follow the same rules as Live.

Batch Processing
1) Add files (WAV/FLAC/MP3/OGG/M4A/AAC).
2) Processing follows the main Processing/Model/Attenuation settings.
3) Echo in batch (optional): put a sibling reference file named <stem>_ref.wav. If present, the app reduces echo before AI/post.
4) Output naming:
   - AI mode: <stem>_enhanced.wav (48 kHz)
   - Normal:  <stem>_processed.wav (original sample rate)

Model Selection (AI)
- Default: DF3 (DeepFilterNet3) — bundled in the exe for offline use.
- Switch to DF2/DF1 via the UI.
- Custom dir: point to a folder containing config.ini and checkpoints/ (for advanced users).

Troubleshooting
- No audio / device busy:
  - Switch output API/device; ensure no other app has exclusive control.
  - Keep Echo off if your device/driver is unstable.
- Distortion after a few seconds (AI):
  - Keep “Force 48 kHz (AI)” enabled.
  - Use Retune Now; the app will try a better block size.
- Pops/dropouts:
  - Retune Now; or set Block to a fixed value (e.g., 480/512/960/1024).
- Attenuation too strong/weak:
  - Adjust the Attenuation slider (AI only). 0 = minimal suppression, 100 = max.
- Batch echo removal not effective:
  - Ensure a synchronized <stem>_ref.wav is present; for better alignment, capture the actual far‑end.
- WebRTC AEC:
  - Not required. If installed, it may be surfaced in future versions; current “Simple” is lightweight.

Keyboard/UX Tips
- Retune Now: quickly tries the next block size without stopping.
- Auto Info (right): check the actual SR/Block — it’s the source of truth.

Build From Source (Optional)
Prereqs: Python 3.10+, Git, Visual C++ Build Tools.
1) (Optional) Create/activate a venv; install dependencies (torch, deepfilternet, sounddevice, soundfile, pyinstaller, etc.).
2) Prepare model:  python prepare_models.py
3) Build exe:       build_exe.bat
4) Output in:       dist\DenoiserX\

Security & Privacy
- No user audio is uploaded. All processing is local.
- Do not share proprietary audio unless you trust the machine you run on.

License / Credits
- DeepFilterNet by the original authors; bundled model used under its terms.
- PortAudio/sounddevice & libsndfile libraries included via wheels.
- This app is provided as-is.

