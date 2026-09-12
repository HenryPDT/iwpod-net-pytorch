# IWPOD-v2 preprocess contract (Phase 2: replaces passthrough + buffer-hijack)

Old WPOD: SGIE emitted `[25x25x8 + 400x400x3 passthrough]`; the plugin read both
from tensor meta, overwrote `out_buf[0]` with float[11] + JPEG in `out_buf[1]`.

v2: SGIE emits single `lpd_pred [B,7,Gh,Gw]`. The plugin reads vehicle pixels
from the `NvBufSurface` frame (operate-on-gie-id vehicle objects), NOT from a
TRT passthrough output. Typed user meta `IWOD_QUAD` carries quads; JPEG/base64
moves to async telemetry (off hot path).

## nvdspreprocess config (keep your existing file, change only these)

```ini
processing-width=256      # plate W (default; change without re-export)
processing-height=96      # plate H
network-input-shape=8;3;96;256
pixel-normalization-factor=0.0039215697906911373
target-unique-ids=<OCR gie id, e.g. 4>
```

## `prepare_tensor()` pseudocode

```text
for each vehicle object:
  tensor = find_meta(unique_id == wpod-unique-id)   # lpd_pred host ptr + dims
  quads  = reconstructIwpod(...)                     # see iwpod_reconstruct_v2.cpp
  if empty: mark no-plate, continue
  crop   = NvBufSurface crop(vehicle bbox, letterboxed to infer-dims coords)
  H      = getPerspectiveTransform(quads[0], dst(0,0,Wplate,0,Wplate,Hplate,0,Hplate))
  plate  = warpPerspective(crop, H, (Wplate,Hplate), INTER_LINEAR, BORDER_CONSTANT 0)
  write plate*scale to devBuf (FP32 [0,1]); attach IWOD_QUAD {quad[8] crop-px, conf}
```
