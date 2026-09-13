# IWPOD preprocess contract (as shipped)

Classic WPOD: SGIE emitted `[25x25x8 + 400x400x3 passthrough]`; the plugin read
both from tensor meta, overwrote `out_buf[0]` with float[11] + JPEG in
`out_buf[1]`. That path remains Conducive rollback (`lpr.lpd_model: "wpod"`).

IWPOD: SGIE emits a single `lpd_pred [B,7,Gh,Gw]`. The plugin letterboxes the
vehicle crop from the `NvBufSurface` frame (`obj_meta->rect_params` →
`infer-dims`, top-left pad). JPEG/base64 is packed after the first 16 floats of
`lpd_pred` so `LPR.cpp` does not need a TRT `output[1]`.

## nvdspreprocess config (OCR warp target)

```ini
processing-width=256      # plate W (default; change without re-export)
processing-height=96      # plate H
network-input-shape=8;3;96;256
pixel-normalization-factor=0.0039215697906911373
target-unique-ids=<OCR gie id, e.g. 4>
custom-lib-path=.../libcustom_iwpod_ocr_preprocess.so
```

## `prepare_tensor()` (IWPOD)

```text
for each vehicle object:
  tensor = find_meta(unique_id == wpod-unique-id)   # lpd_pred host ptr + dims
  crop   = NvBufSurface letterbox(vehicle bbox → infer-dims)
  quads  = reconstructIwpod(...)                     # iwpod_reconstruct.cpp
  if empty: mark no-plate, continue
  H      = getPerspectiveTransform(quads[0], dst 256×96)
  plate  = warpPerspective(crop, H, (256,96), INTER_LINEAR, BORDER_CONSTANT 0)
  write plate*scale to OCR devBuf (FP32 [0,1])
  write tensor_output[0..7]=quad, [8]=conf, [10]=is-plate
  pack JPEG bytes at tensor_output[16..] ; [9]=jpeg length
```

Quad pixels are in **letterbox / SGIE input** space, matching
`maintain-aspect-ratio=1` without `symmetric-padding`.
