# Downstream Guide: IWPOD in the Conducive DeepStream LPR chain

Target pipeline: `PGIE vehicle id=1 → SGIE LPD id=2 → nvdspreprocess id=3 →
SGIE OCR id=4`. Product destination:
`Conducive-AI-Deployment` (`deepstream-streamer`).

**IWPOD is the default LPD.** Export a single `lpd_pred` tensor. The preprocess
plugin letterboxes the vehicle crop from `NvBufSurface` (not a TRT passthrough)
and packs JPEG telemetry after the first 16 floats of `lpd_pred`.

`--with-passthrough` is **not** a DeepStream product path. Classic WPOD (400
NHWC + passthrough crop) stays in Conducive as `lpr.lpd_model: "wpod"` rollback.

All paths below assume vehicle-crop input and default `256×96` warp (unchanged).

## Do we run old WPOD and new IWPOD side by side? No.

Only **one** LPD model ever runs, in the same slot (`gie-unique-id=2`,
`operate-on-gie-id=1`). Switch with Analytics JSON
`cameras[i].config.lpr.lpd_model` (`"iwpod"` | `"wpod"`). First camera that sets
it wins; changing it rebuilds the pipeline.

| Slot | IWPOD (default) | WPOD rollback |
|---|---|---|
| `onnx-file` | `iwpod_416.onnx` | `lp_exp_7_06_12_23_w_passthrough.onnx` |
| plugin `.so` | `libcustom_iwpod_ocr_preprocess.so` | `libcustom_wpod_ocr_preprocess.so` |
| crop source | `NvBufSurface` letterbox to 416 | TRT `output[1]` passthrough 400 HWC |
| JPEG | packed after 16 floats of `lpd_pred` | `out_buf[1]` passthrough buffer |

---

## 0. File map (this repo → Conducive)

| This repo (`iwpod-net-pytorch/deepstream/`) | Pipeline destination |
|---|---|
| `iwpod_reconstruct.{h,cpp}` | `ocr_preprocessor/` (IWPOD `.so`) |
| `nvdspreprocess_iwpod_impl.cpp` | `ocr_preprocessor/` (IWPOD `.so`) |
| `config_infer_secondary_iwpod.txt` | `streamer/config/config_infer_secondary_iwpod.txt` |
| `*.onnx` from `iwpod export` (no passthrough) | `/app/deepstream_src/models/iwpod_416.onnx` |
| `../iwpod/decode.py` | Python mirror of the C++ decode (parity tests) |

Conducive already builds **both** plugins from `ocr_preprocessor/Makefile`.
Do not replace the WPOD `.so` with the IWPOD TU.

---

## 1. Export the model (training box)

```bash
source .venv/bin/activate
iwpod export -w out/train/iwpod_lpr/iwpod_lpr_best.pth -s 416 \
  --dynamic --dynamic-shape --simplify --check-parity -o iwpod_416.onnx
python -c "import onnxruntime as ort
s=ort.InferenceSession('iwpod_416.onnx',providers=['CPUExecutionProvider'])
print([(o.name,o.shape) for o in s.get_outputs()])"
# expect: lpd_pred [B,7,Gh,Gw]  — only that output
cp iwpod_416.onnx /app/deepstream_src/models/
rm -f /app/deepstream_src/models/*.engine   # force TRT rebuild on next run
```

Do **not** pass `--with-passthrough`. That Identity crop copy is leftover
export machinery; Conducive IWPOD never reads `output[1]`.

---

## 2. Config (`config_infer_secondary_iwpod.txt`)

```ini
onnx-file=/app/deepstream_src/models/iwpod_416.onnx
infer-dims=3;416;416
network-input-order=0
network-mode=2
maintain-aspect-ratio=1
# do NOT set symmetric-padding=1
```

Notes:
- `network-mode=2` (FP16) is the shipped default. Fall back to `0` (FP32) only
  if a platform's `wpod_confidence` histogram shifts.
- Do **not** add `symmetric-padding=1`: LPR maps quads with
  `scale=max(w,h ratios)` + top-left offset (asymmetric letterbox). Symmetric
  padding would shift every quad by half the pad, uncompensated.
- Everything else stays: `network-type=100`, `output-tensor-meta=1`,
  `cluster-mode=4`, `gie-unique-id=2`, `batch-size=8`. Xavier NX, Orin and x86
  all run `3;416;416`.

### GIE chain (unchanged)

| Stage | Element | gie-unique-id | Operates on |
|---|---|---|---|
| PGIE vehicle detector | `nvinfer` | 1 | full frame |
| SGIE IWPOD | `nvinfer` | 2 | vehicle objects (`operate-on-gie-id=1`) |
| Warp plugin | `nvdspreprocess` | 3 | LPD tensor meta (`wpod-unique-id=2`), emits OCR tensor |
| SGIE OCR | `nvinfer` | 4 | preprocess output (`operate-on-gie-id=3`, `input-tensor-from-meta=1`) |

Only the row with `gie-unique-id=2` and the preprocess `.so` change vs classic
WPOD. If a branch remaps IDs (`generate_wpod_config`), keep relative offsets.

---

## 3. Plugin (already in Conducive)

| File (this repo) | Destination | Purpose |
|---|---|---|
| `iwpod_reconstruct.{h,cpp}` | `ocr_preprocessor/` | grid decoder (NCHW + NMS) |
| `nvdspreprocess_iwpod_impl.cpp` | `ocr_preprocessor/` | letterbox from `NvBufSurface`, warp, packed JPEG |

Makefile (Conducive, already wired):

```make
WPOD_SRCS:= nvdspreprocess_impl.cpp
IWPOD_SRCS:= nvdspreprocess_iwpod_impl.cpp iwpod_reconstruct.cpp
# → libcustom_wpod_ocr_preprocess.so
# → libcustom_iwpod_ocr_preprocess.so
```

OCR still reads the NCHW 256×96 plate tensor. `LPR.cpp` reads quads from
`lpd_pred[0..10]` and JPEG after 16 floats when there is no passthrough layer.

**Decode constants** (compile-time in the plugin, not `[user-configs]`):

| Symbol | Value |
|---|---|
| stride | 16 |
| side | 7.75 = `((208+40)/2)/16` |
| threshold / topk / nms | 0.3 / 1 / 0.25 |
| warp | 256×96 (`config_ocr_preprocess.txt`) |
| OCR in | `3;96;256`, `input-tensor-from-meta=1` |

Python `iwpod/decode.py` uses the same AABB size gate and
`getPerspectiveTransform` as C++. Offline infer letterbox is top-left black
pad, matching `maintain-aspect-ratio=1` without `symmetric-padding`.

### Old-decoder bugs the IWPOD decoder does not carry over

1. Cell mismatch: confidence used transposed NHWC indexing; affine fetched from `(c,r)`.
2. `Affines[x][y]` filled but `Affines[y][x]` consumed.
3. `abs()` on doubles in the 30×10 size gate (int truncation) → `fabs` + scale-aware gate.
4. Stack VLA `double Affines[W][H][D-2]` → `std::vector`.

---

## 4. DS 6.2 (JP5, TRT8.5) vs DS 9.1 (JP7.1, TRT10)

- Same ONNX (opset 17) + same plugin source for both. Rebuild the plugin and
  the TRT engine **on each platform** — engines are not portable.
- Plugin build matrix (`ocr_preprocessor/Makefile`; `CUDA_VER` selects the
  `DS_VERSION_MAJOR/MINOR` macros automatically):

| Target | Command |
|---|---|
| Xavier NX/AGX, DS 6.2 / JP5 | `make CUDA_VER=11.4` |
| Orin Nano/NX + x86, DS 9.1 / JP7.1 | `make CUDA_VER=13.0` (use the exact `nvcc --version` on the box) |

- First run takes minutes (engine build); set `workspace-size` as before.

## 5. Test checklist (per platform)

- [ ] Engine builds, no `COND_LOG_ERROR` in streamer logs.
- [ ] `ort` output name is only `lpd_pred`; shape `[B,7,Gh,Gw]`.
- [ ] Warp/JPEG look like the vehicle crop (letterbox top-left, not a 3×416 sliver).
- [ ] Offline `iwpod infer` quads on a non-square crop match DeepStream OSD.
- [ ] `wpod_confidence > 0` on vehicles with visible plates; OSD polygon correct.
- [ ] Rectified plates (`images.plate: true`) sharp/horizontal; OCR strings in AMQP.
- [ ] Rollback: `"lpr": { "lpd_model": "wpod" }` restores classic 400 passthrough.

## 6. Rollback

In Analytics JSON (not pipeline-root):

```json
"lpr": { "lpd_model": "wpod" }
```

That rebuilds onto `config_infer_secondary_wpod.txt` +
`libcustom_wpod_ocr_preprocess.so`. Keep the old WPOD ONNX/engine on disk.
