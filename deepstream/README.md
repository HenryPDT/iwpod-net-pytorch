# deepstream/ — IWPOD-v2 deployment package

Everything needed to swap the WPOD stage for IWPOD-v2 in `wpod_lpr_pipeline`,
without touching vehicle detection, OCR, tracking, or telemetry.

| File | What it is |
|---|---|
| `DOWNSTREAM_GUIDE.md` | **Start here.** Full rollout: export → 4-line config diff → plugin swap → per-platform build matrix → test checklist → rollback |
| `config_infer_secondary_iwpod.txt` | Drop-in replacement content for `config_infer_secondary_wpod.txt` (NCHW, 416, FP32 bring-up; FP16 path documented) |
| `iwpod_reconstruct_v2.{h,cpp}` | New grid decoder (replaces `reconstructSingle`; NCHW+NMS, fixes 5 old-decoder bugs) |
| `nvdspreprocess_iwpod_impl.cpp` | Full TU replacing `nvdspreprocess_impl.cpp` (same class/entry points; vectorized tensor write, identical downstream contract) |
| `PREPROCESS_CONTRACT.md` | Phase-2 design note (drop passthrough; only after Phase 1 is green) |

Key constraints enforced by these files (see guide §2 for why):
- No `symmetric-padding` (LPR.cpp assumes asymmetric letterbox math).
- FP32 first, FP16 only after the histogram check.
- Decode constants are compile-time in the plugin (not `[user-configs]`).
- Phase-1 ONNX is `iwpodv2_416_w_passthrough.onnx`; the plugin converts NCHW passthrough to HWC.
- One LPD model at a time; old `.so` + `.onnx` + `.engine` stay on disk as rollback.
