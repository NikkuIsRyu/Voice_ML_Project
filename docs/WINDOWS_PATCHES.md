# Windows Native Environment Patches for GPT-SoVITS

This document details all modifications and workarounds applied to `GPT-SoVITS` and related libraries to ensure stable execution on native Windows 11 with NVIDIA RTX 40-series (PyTorch 2.14+cu126 / CUDA 12.x / 13.x).

---

## 1. Multiprocessing & DDP Segfault (0xC0000005) Fixes

### A. `s2_train.py` (SoVITS Acoustic Model Training)
- **Problem:** On Windows, `torch.multiprocessing.spawn` using the PyTorch DDP `gloo` backend with `num_workers > 0` causes an immediate Windows Access Violation segfault (`0xC0000005`).
- **Fix:** When `n_gpus == 1`, bypass `mp.spawn` and DDP initialization. Execute the training loop directly in the primary process with direct `.cuda()` device placement and set DataLoader `num_workers = 0`.

### B. `s1_train.py` & `bucket_sampler.py` (GPT Semantic Model Training)
- **Problem:** `bucket_sampler.py` and `DistributedBucketSampler` called `torch.distributed.get_world_size()` without verifying whether distributed mode was actually initialized (`dist.is_initialized()`).
- **Fix:** In `bucket_sampler.py`, added a fallback: if `dist.is_initialized()` is False, default `num_replicas = 1` and `rank = 0`.
- **In `data_module.py`:** If `num_workers == 0`, PyTorch throws a ValueError if `prefetch_factor` is set. Patched to ensure `prefetch_factor = None` and `persistent_workers = False` when `num_workers == 0`.

---

## 2. Torchaudio & TorchCodec Windows DLL Crash

- **Problem:** `torchaudio` 2.11+ on Windows attempts to load audio files via `torchcodec` or C++ FFmpeg shared DLL bindings by default. Missing or mismatched runtime DLLs trigger fatal process termination.
- **Fix:** In `torchaudio/__init__.py` (or caller modules), configured seamless fallback to `soundfile` as the primary I/O backend for `torchaudio.load()` and `torchaudio.save()`.

---

## 3. Static FFmpeg Deployment

- **Problem:** Conda-forge's standard Windows `ffmpeg.exe` can fail with error code `0xC0000135` (missing DLL dependency) when invoked outside specific shell paths.
- **Fix:** Deployed static Gyan FFmpeg Essentials 9.0.1 binaries (`ffmpeg.exe` and `ffprobe.exe`) directly into `GPT-SoVITS\`, `scripts\`, and the conda environment `Library\bin\`.

---

## 4. FastText Language Identification Model

- **Problem:** `fast_langdetect` failed on initial startup due to missing model file `lid.176.bin`.
- **Fix:** Downloaded the pre-trained `lid.176.bin` (125 MB) and positioned it inside `GPT_SoVITS/pretrained_models/fast_langdetect/`.

---

## 5. API Server & Config Profiles

- **File:** `GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml`
- **Customization:** Configured a dedicated `custom` profile linking fine-tuned weights (`t2s_weights_path: GPT_weights_v2/...`, `vits_weights_path: SoVITS_weights_v2/...`) with `is_half: true` and `device: cuda`.
