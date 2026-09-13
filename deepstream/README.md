# deepstream/ — IWPOD-v2 deployment package

Reference copies of the IWPOD LPD stage that ships in Conducive
`deepstream-streamer`. Vehicle detection, OCR, tracking, and classic WPOD
rollback stay in the product repo.

| File | What it is |
|---|---|
| `DOWNSTREAM_GUIDE.md` | **Start here.** Export, config, plugin contract, per-platform build, checklist, rollback |
| `config_infer_secondary_iwpod.txt` | NCHW 416, single `lpd_pred`, FP16 (`network-mode=2`) |
| `iwpod_reconstruct.{h,cpp}` | Grid decoder (NCHW + NMS) |
| `nvdspreprocess_iwpod_impl.cpp` | Letterbox from `NvBufSurface`, warp to 256×96, JPEG packed after 16 floats of `lpd_pred` |
| `PREPROCESS_CONTRACT.md` | Preprocess / telemetry contract as shipped |

Key constraints:
- No `symmetric-padding` (LPR.cpp assumes asymmetric letterbox math).
- Product ONNX is `iwpod_416.onnx` — **no** `--with-passthrough`.
- Decode constants are compile-time in the plugin (not `[user-configs]`).
- One LPD model at a time. Classic WPOD is `lpr.lpd_model: "wpod"` in Conducive.
