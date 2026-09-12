# ONNX Export & Deployment Prep

Export converts a `.pth` checkpoint to the ONNX the DeepStream stage consumes:
NCHW `lpd_pred [B,7,Gh,Gw]` (ch0 = raw objectness **logits**, ch1..6 = affine).
Phase 1 keeps a passthrough copy of the vehicle crop (`--with-passthrough`) for
the legacy pipeline shape.

**Train at 384 (multi-scale), deploy at 416.** Export at the deploy size.

## Recipes

```bash
# Standard: dynamic batch + dynamic shape, simplified (recommended default)
iwpod export -w out/train/exp1/exp1_best.pth -s 416 --dynamic --dynamic-shape --simplify

# Phase 1 DeepStream (legacy pipeline shape keeps its passthrough design)
iwpod export -w out/train/exp1/exp1_best.pth -s 416 --dynamic --dynamic-shape \
  --simplify --with-passthrough --check-parity -o iwpodv2_416_w_passthrough.onnx

# Max-compat fallback for old TRT 8.5 stacks (validate with trtexec on target)
iwpod export -w out/train/exp1/exp1_best.pth -s 384 --opset 13 --dynamic --simplify
```

## Flag reference

| Flag | Default | Meaning |
|---|---|---|
| `-w/--weights` | (required) | Input checkpoint (state dict or `{model_state_dict}`) |
| `-s/--size` | `[416]` | `[H,W]` or single square; **must be multiples of `--align`**. 416 is the Xavier NX Phase-1 infer-dims |
| `--opset` | 17 | ONNX opset. 17 works on TRT 8.5 (DS 6.2) and TRT 10 (DS 9.1) for the 5 stable ops used here (Conv/Relu/MaxPool/Add/Concat). 13 is a fallback, not the default |
| `--simplify` | off | Run `onnxslim` (recommended: smaller graph, same numerics) |
| `--dynamic` | off | Dynamic batch axis (`batch`) |
| `--dynamic-shape` | off | Dynamic spatial axes (`height/width` in, `h_out=h/16, w_out=w/16` out). Combine with `--dynamic` for full flexibility |
| `--batch` | 1 | Static batch used for the dummy trace (ignored with `--dynamic`) |
| `--align` | 32 | H/W alignment enforced at export (covers stride-16 + TRT + future heads) |
| `--with-sigmoid` | off | Bake sigmoid into the graph. Default (raw logits) is recommended: the DS parser applies sigmoid so the threshold stays runtime-tunable and INT8 calibration sees well-behaved activations |
| `--with-passthrough` | off | LEGACY Phase-1: append `Identity(input→pass_through_output)` 2nd output for the old pipeline's crop-forwarding hack. Required by the deployed Phase-1 pipeline (`DOWNSTREAM_GUIDE.md`); omit for the clean Phase 2 design |
| `--fuse` | off | Fold Conv-BN before export (opt-in; does not change default weights). Re-run `--check-parity` |
| `--check-parity` | off | Compare torch vs ORT on a representative (non-zero) crop; with `--dynamic` also asserts batch-2 |
| `-o/--output` | `<weights>.onnx` | Output path |

`--dynamic` + `--batch > 1` together is rejected (contradictory).

## What "dynamic" buys you

One engine serves many configs without re-export: DeepStream `infer-dims` is
`3;416;416` on all platforms (Xavier NX, Orin, x86) and batch varies per
load — the grid follows as `H/16 × W/16` (416→26×26).
Verified in ORT: batch-2 runs from one file.

## Validation (do this on every export)

```bash
python -c "import onnxruntime as ort
s = ort.InferenceSession('model.onnx', providers=['CPUExecutionProvider'])
print([(o.name, o.shape) for o in s.get_outputs()])"
# expect lpd_pred [B,7,Gh,Gw] (+ pass_through_output [B,3,H,W] iff --with-passthrough)
```

Then compare against torch (`--check-parity`): representative `[0,1]` crop, not
zeros. Shape inference + `onnx.checker` always run.

## TensorRT via `iwpod export-trt` (on target)

Engines are **not portable** — build on each platform (run on the target,
not here):

```bash
iwpod export-trt --onnx out/train/exp1/model.onnx --size 416 --fp16
iwpod export-trt --onnx model.onnx --size 416 --batch 8 --int8 --calib calib.cache
```

Profiles derive from `--size` (min 256 / opt `<size>` / max 512, batch 1..N)
and are emitted **only if the ONNX is dynamic**. Workspace flag is
`--workspace=` on TRT 8.5 and `--memPoolSize=workspace:` on TRT 10.
`--fp16` is default-on, `--int8` requires `--calib`, and the builder log is
captured next to the engine.

- FP16 is the recommended default on Xavier + Orin **after** FP32 bring-up;
  check `wpod_confidence` histograms before/after (expect <1pp shift).
- INT8-PTQ only with a calibration set of vehicle crops, gated on <0.5pp
  recall drop (see EVALUATION.md).
- First DeepStream run builds the engine and takes minutes; keep the
  `.engine` file next to the `.onnx`.

## Known caveat

With `--opset 13` the torch exporter logs `No Adapter From Version $16 for
Identity` but still writes a checker-valid file that runs in ORT. Treat 13 as
"works, but must be confirmed with `trtexec` on the DS 6.2 box" — default to 17.
