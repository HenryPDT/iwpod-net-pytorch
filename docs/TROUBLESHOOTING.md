# Troubleshooting

## GPU / environment

| Symptom | Cause → Fix |
|---|---|
| `Invalid handle. Cannot load symbol cudnnGetVersion`, `cudnn.is_available()==False` | NoMachine `LD_PRELOAD` hook (see README). Fixed permanently via `~/.bashrc`; ad-hoc: `env -u LD_PRELOAD <cmd>`; services: `UnsetEnvironment=LD_PRELOAD` |
| `torch.cuda.is_available()==False` on a CUDA box | CPU torch installed. Reinstall: `uv pip install torch --index-url https://download.pytorch.org/whl/cu126` (match your CUDA) |
| ORT CUDA EP: `libcudnn.so missing` | `pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12` (cu12 for ORT 1.29) |
| CUDA OOM in training | Halve `batch_size`; narrow `multi_scale`; `num_workers: 4`. Batch size matters less than epoch count here |

## Training

| Symptom | Cause → Fix |
|---|---|
| `IndexError` in `project_all` / `random_crop` on old code | Fixed: BG composite + crop are size-robust. Ensure `bgimages/` short side ≥ train `dim` anyway (DATASET.md) |
| Train loss falls, val stalls >20 epochs | Overfitting: more crops, wider augmentation, or stop and ship `_best.pth` |
| All-zero positives in `y` (`pos 0`) | Annotation not parsed: check `.txt` format (DATASET.md) and that the quad lies inside the image |
| `size must be multiple of 32` (train or export) | `dim`/`multi_scale`/`-s` must be %32 — pick 256/320/384/448/512 |

## Export / TensorRT

| Symptom | Cause → Fix |
|---|---|
| `No Adapter From Version $16 for Identity` with `--opset 13` | Known torch-exporter notice; file still validates + runs in ORT. Confirm with `trtexec` on the DS 6.2 box; prefer opset 17 |
| TRT engine fails to parse | Check opset (17 default), confirm single `lpd_pred` output, rebuild **on the target** — engines aren't portable |
| First DeepStream run hangs for minutes | Normal: engine building. Keep the `.engine` file next to the `.onnx` |
| `wpod_confidence` distribution shifts after FP16/INT8 | Re-run `iwpod eval` thresholds on the quantized engine; gate per EVALUATION.md |

## Detection quality

| Symptom | Cause → Fix |
|---|---|
| High recall, low IoU | Localization weak: raise `loss.w_loc`, check aspect-bucket labels, verify quads are clockwise from top-left |
| Low recall, decent IoU on hits | Threshold too high, or plates outside trained scale (feed vehicle crops, not full frames) |
| Plates found but strings wrong | Warp OK, OCR issue — or plate size changed without OCR retrain (keep 256×96) |
| Double detections per plate | Lower NMS IoU (compile-time `kNmsIou`) or raise threshold; check `topk=1` in production |
