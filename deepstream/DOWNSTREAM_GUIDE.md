# Downstream Guide: running the new IWPOD-v2 in your DeepStream pipeline

Target pipeline: `wpod_lpr_pipeline` (`PGIE vehicle id=1 → SGIE WPOD id=2 →
nvdspreprocess id=3 → SGIE OCR id=4`). This doc gives you two rollout options.
**Phase 1 is recommended**: new accuracy, ~15 lines changed, everything else
(incl. `LPR.cpp`, JPEG telemetry, buffer-hijack contract) untouched, old engine
kept as instant rollback.

All paths below assume vehicle-crop input and default `256×96` warp (unchanged).

## Do we run old WPOD and new IWPOD side by side? No.

Only **one** LPD model ever runs, in the same slot (`gie-unique-id=2`,
`operate-on-gie-id=1`). "Fallback" means the old artifacts stay on disk and a
config flip restores them — not dual runtime (two SGIEs would double GPU cost
for zero benefit):

| Slot | Normal (new) | Fallback (old) |
|---|---|---|
| `onnx-file` | `iwpodv2_416_w_passthrough.onnx` | `lp_exp_7_06_12_23_w_passthrough.onnx` |
| plugin `.so` | rebuilt with `iwpod_reconstruct_v2.cpp` | previous `.so` (keep a copy) |
| engine | `iwpodv2_*.engine` | previous `.engine` (keep a copy) |

Phase 1 vs Phase 2 is **sequencing, not parallelism**: Phase 1 = new weights in
the existing pipeline shape (passthrough kept, §1–§3). Phase 2 = later cleanup
that removes the passthrough copy (design in §5), done only once Phase 1 is
green on all platforms. At no point do two LPD models run concurrently.

---

## 0. File map (this repo → your pipeline repo)

| This repo (`iwpod-net-pytorch/deepstream/`) | Pipeline destination | Phase |
|---|---|---|
| `iwpod_reconstruct_v2.cpp` | `ocr_preprocessor/` (add to `Makefile` SOURCES, or `#include` it) | 1 |
| `config_infer_secondary_iwpod.txt` | `streamer/config/config_infer_secondary_wpod.txt` (drop-in; see §2) | 1 |
| `*.onnx` from `iwpod export` | `/app/deepstream_src/models/` | 1 |
| `PREPROCESS_CONTRACT.md` | design note for Phase 2 | 2 |
| `../iwpod/decode.py` | Python mirror of the C++ decode (parity tests) | test |

---

## 1. Export the model (do this first, on the training box)

```bash
source .venv/bin/activate
# Phase 1 (keeps pipeline's passthrough design — REQUIRED for Phase 1):
iwpod export -w out/train/exp1/exp1_best.pth -s 416 --dynamic --dynamic-shape \
  --simplify --with-passthrough -o iwpodv2_416_w_passthrough.onnx
# verify:
python -c "import onnxruntime as ort
s=ort.InferenceSession('iwpodv2_416_w_passthrough.onnx',providers=['CPUExecutionProvider'])
print([(o.name,o.shape) for o in s.get_outputs()])"
# expect: lpd_pred [B,7,Gh,Gw] + pass_through_output [B,3,H,W]
cp iwpodv2_416_w_passthrough.onnx /app/deepstream_src/models/
rm -f /app/deepstream_src/models/*.engine   # force TRT rebuild on next run
```

Why `--with-passthrough`: your `prepare_tensor()` reads the vehicle crop from
`out_buf_ptrs_host[1]`. Keeping it means **zero changes** to image flow, JPEG
telemetry, and `LPR.cpp`. (Removing it = Phase 2, §5.)

---

## 2. Config change (`config_infer_secondary_wpod.txt`)

4-line diff against your current file (old TF/Keras WPOD, NHWC → IWPOD-v2, PyTorch NCHW):

```ini
# BEFORE:
onnx-file=/app/deepstream_src/models/lp_exp_7_06_12_23_w_passthrough.onnx
infer-dims=3;400;400
network-input-order=1

# AFTER (full replacement file: deepstream/config_infer_secondary_iwpod.txt):
onnx-file=/app/deepstream_src/models/iwpodv2_416_w_passthrough.onnx
infer-dims=3;416;416
network-input-order=0
network-mode=0
```

Notes:
- `network-mode=0` (FP32) is deliberate for bring-up — your current file runs
  FP32 too. Flip to `2` (FP16) only after the §6 histogram check.
- Do **not** add `symmetric-padding=1`: `LPR.cpp:178,197` maps quads with
  `scale=max(w,h ratios)` + top-left offset (asymmetric letterbox math).
  Symmetric padding would shift every quad by half the pad, uncompensated.
  Our config file omits the key for exactly this reason.
- Everything else stays: `network-type=100`, `output-tensor-meta=1`,
  `cluster-mode=4`, `maintain-aspect-ratio=1`, `gie-unique-id=2`, `batch-size=8`.
  No per-device tuning: Xavier NX, Orin and x86 all run `3;416;416`.

### GIE chain (unchanged, for reference)

| Stage | Element | gie-unique-id | Operates on |
|---|---|---|---|
| PGIE vehicle detector | `nvinfer` | 1 | full frame |
| SGIE IWPOD (this swap) | `nvinfer` | 2 | vehicle objects (`operate-on-gie-id=1`) |
| Warp plugin | `nvdspreprocess` | 3 | WPOD tensor meta (`wpod-unique-id=2`), emits OCR tensor |
| SGIE OCR | `nvinfer` | 4 | preprocess output (`operate-on-gie-id=3`, `input-tensor-from-meta=1`) |

Only the row with `gie-unique-id=2` changes. If your branch remaps IDs
(`generate_wpod_config`), keep the relative offsets identical.

---

## 3. Code patch: reuse the pipeline, rewrite the hot path

**Reuse vs reimplement (the rule applied):** reuse everything that is pipeline
plumbing (GIE chain, configs, `LPR.cpp`/`Vehicle.cpp`, telemetry/JPEG blocks,
`nvdspreprocess_lib.cpp`, Makefile structure); reimplement only what runs per
vehicle per frame (grid decode + tensor write). Two files do that:

| File (this repo) | Destination | Purpose |
|---|---|---|
| `iwpod_reconstruct_v2.{h,cpp}` | `ocr_preprocessor/` | new decoder (replaces `reconstructSingle`) |
| `nvdspreprocess_iwpod_impl.cpp` | `ocr_preprocessor/` | full TU replacing `nvdspreprocess_impl.cpp` (same class, same header, same lib entry points) |

**Swap-in (2 lines in `ocr_preprocessor/Makefile`):**

```make
# BEFORE:
SRCS:= nvdspreprocess_lib.cpp nvdspreprocess_impl.cpp \
       nvdspreprocess_conversion.cu
# AFTER:
SRCS:= nvdspreprocess_lib.cpp nvdspreprocess_iwpod_impl.cpp \
       iwpod_reconstruct_v2.cpp nvdspreprocess_conversion.cu
```

No changes to `nvdspreprocess_lib.cpp`, configs (beyond §2), `LPR.cpp`, or OCR.
`nvdspreprocess_impl.cpp` stays on disk as the rollback copy.

**Why the rewrite is faster (same outputs, less work per object):**

| Hot spot | Old | New |
|---|---|---|
| Grid scan | full `W×H×(D-2)` double copy into a stack VLA, then decode | single pass, decode winners only, no copy, no VLA |
| Candidate handling | max-only, transposed indexing | top-k + quad-NMS (`topk=1` = identical behavior), correct NCHW/NHWC indexing |
| CHW interleave (256×96×3 px) | `channels×rows×cols` loop with `.at<Vec3f>` per pixel | row-pointer planar write, one fused scale multiply |
| Normalization | `(v*255)*m_Scale` per pixel | `v*k`, `k=255*m_Scale` hoisted (same values) |
| JPEG/telemetry/hijack/`LPR.cpp` contract | — | byte-identical (kept, incl. quirks, to avoid drift) |

**Minimal alternative** (if you prefer a smaller diff): keep
`nvdspreprocess_impl.cpp`, drop in only `iwpod_reconstruct_v2.cpp`, and replace
just the decode call (~line 540) with:

```cpp
// OLD:
float min_confidence = 0.3;
float confidence = reconstructSingle(&image_output, pts, image, wpod,
        in_size, wpod_dims, out_size, min_confidence);

// NEW (auto-detects v2 NCHW [7,Gh,Gw] vs legacy NHWC [Gh,Gw,8]):
float min_confidence = 0.3;
int vC, vGh, vGw; bool vNCHW, vLogits;
if (wpod_dims[2] == 8 && wpod_dims[0] != 8) {      // legacy TF model, NHWC
    vNCHW = false; vC = 8; vGh = wpod_dims[0]; vGw = wpod_dims[1]; vLogits = false;
} else {                                            // IWPOD-v2, NCHW [7,Gh,Gw]
    vNCHW = true;  vC = wpod_dims[0]; vGh = wpod_dims[1]; vGw = wpod_dims[2]; vLogits = true;
}
float confidence = iwpod_v2::reconstructIwpod(
        &image_output, pts, image, wpod, vC, vGh, vGw, vNCHW,
        /*in_w=*/(int)in_size[1], /*in_h=*/(int)in_size[0],
        /*out_w=*/(int)out_size[0], /*out_h=*/(int)out_size[1],
        /*stride=*/16.0, /*side=*/7.75, min_confidence,
        /*from_logits=*/vLogits, /*topk=*/1, /*nms_iou=*/0.25);
```

Notes:
- `in_size`/`out_size` semantics: the plugin converts Phase-1 passthrough
  `pass_through_output [B,3,H,W]` (NCHW planar) to an HWC `cv::Mat` before
  warp/JPEG. Legacy HWC `[H,W,3]` is still accepted. Rank/channel mismatches
  fail closed (object skipped, no crash). Warp is still
  `processing-width/height` = 256×96 from `config_ocr_preprocess.txt`.
- `topk=1` reproduces the old max-only behavior exactly; raise later if you
  ever need 2 plates per vehicle crop.
- Downstream (`wpod[0..10]` hijack, `LPR.cpp`, `Vehicle.cpp` gate) is untouched
  — `tensor_output[0..7]` + return value keep the old contract byte-for-byte.
- Python `iwpod/decode.py` uses the same AABB size gate and
  `getPerspectiveTransform` as C++. Offline eval/infer letterbox is top-left
  black pad, matching `maintain-aspect-ratio=1` without `symmetric-padding`.

**3c.** Rebuild the plugin (`make` in `ocr_preprocessor/`, redeploy
`libcustom_wpod_ocr_preprocess.so`), delete the old TRT engine, restart.

### Old-decoder bugs fixed by the swap (for the record)

1. Cell mismatch: confidence uses transposed NHWC indexing, then x/y are
   swapped back for `mn` but NOT for the affine fetch — quad for cell (r,c)
   uses the affine of cell (c,r). Benign only for near-symmetric plates.
2. `Affines[x][y]` filled but `Affines[y][x]` consumed.
3. `abs()` on doubles in the 30×10 size gate (int truncation) → `fabs` + scale-aware gate.
4. Stack VLA `double Affines[W][H][D-2]` → `std::vector`.
5. Hardcoded `stride/side/threshold` as `reconstructIwpod` parameters
   (compile-time in the full TU; **not** DeepStream `[user-configs]`).

---

## 4. Constants (must match everywhere)

| Symbol | Value | Lives in |
|---|---|---|
| stride | 16 | `iwpod/constants.py` + C++ `kNetStride` (compile-time) |
| side | 7.75 = ((208+40)/2)/16 | same |
| threshold / topk / nms | 0.3 / 1 / 0.25 | C++ `kMinConfidence` / `kTopK` / `kNmsIou` (compile-time). Tune offline, then rebuild the plugin. |
| warp | 256×96 | `config_ocr_preprocess.txt` only (never in ONNX) |
| OCR in | `3;96;256`, `input-tensor-from-meta=1` | `config_infer_secondary_ocr_yoloDarknet.txt` (unchanged) |

---

## 5. Phase 2 (optional clean-up, later): drop the passthrough

Export **without** `--with-passthrough` (single `lpd_pred` output, ~½ host
traffic), and change `prepare_tensor()` to crop the vehicle image from the
`NvBufSurface` frame (`obj_meta->rect_params` + letterbox to `infer-dims`)
instead of `out_buf_ptrs_host[1]`. Move JPEG bytes out of `out_buf[1]` into an
`NvDsUserMeta` (requires the matching 10-line `LPR.cpp` read-side change).
Full sketch: `PREPROCESS_CONTRACT.md`. Do this only after Phase 1 is green —
it touches the `LPR.cpp` telemetry contract.

---

## 6. DS 6.2 (JP5, TRT8.5) vs DS 9.1 (JP7.1, TRT10)

- Same ONNX (opset 17) + same plugin source for both. Rebuild the plugin and
  the TRT engine **on each platform** — engines are not portable.
- Plugin build matrix (`ocr_preprocessor/Makefile`; `CUDA_VER` selects the
  `DS_VERSION_MAJOR/MINOR` macros automatically):

| Target | Command |
|---|---|
| Xavier NX/AGX, DS 6.2 / JP5 | `make CUDA_VER=11.4` |
| Orin Nano/NX + x86, DS 9.1 / JP7.1 | `make CUDA_VER=13.0` (13.x branch; use the exact `nvcc --version` on the box if it differs) |

- First run takes minutes (engine build); set `workspace-size` as before.
- Precision path: bring up on FP32 (`network-mode=0`, matches your current
  file), then flip to FP16 (`network-mode=2`) on Xavier + Orin after
  validating `wpod_confidence` histograms (<1pp shift).

## 7. Test checklist (per platform)

- [ ] Engine builds, no `COND_LOG_ERROR` in streamer logs.
- [ ] `ort` output names `lpd_pred + pass_through_output`; shapes `[B,7,Gh,Gw]` + `[B,3,H,W]`.
- [ ] Plugin logs passthrough as NCHW `[3,416,416]` (or accepted HWC) and warp/JPEG look like the vehicle crop, not a 3×416 sliver.
- [ ] Offline `iwpod infer` quads on a non-square crop match DeepStream OSD (top-left letterbox; no half-pad shift).
- [ ] `wpod_confidence > 0` on vehicles with visible plates; OSD polygon correct.
- [ ] Rectified plates (`images.plate: true`) sharp/horizontal; OCR strings in AMQP.
- [ ] `wpod_tensor` JSON non-zero; regression clips ≥ old strings.
- [ ] FP32 bring-up first; FP16 only if confidence histogram, corner RMSE, recall, and latency gates pass (`EVALUATION.md`).
- [ ] Rollback: old `.so` + old `onnx-file` + old engine restore the previous behavior.

## 8. Rollback

Keep the previous `libcustom_wpod_ocr_preprocess.so`, previous `onnx-file`, and
its `.engine` file. Rollback = restore 2 files + restart. No DB/config migration.
