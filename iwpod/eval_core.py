"""Shared quad-geometric evaluation (single source of truth).

Used by both `iwpod eval` (standalone) and the training loop (per-epoch val).
Letterboxes inputs to square like DeepStream (maintain-aspect-ratio=1,
top-left pad) and compares quads in original normalized coords.

Entry format: (jpg_path, gt_quad_or_None) where gt is a (2,4) normalized array
or None for known-negative crops (only false-positive counted for those).

Decode runs once at a low base threshold keeping the top-1 confidence; the
operating point and the full threshold/F1 sweep derive analytically from the
per-image (conf, IoU) pairs — no extra forwards.
"""
import os
import time

import cv2
import numpy as np
import torch
from loguru import logger

from iwpod.constants import IMAGE_EXTS, is_real_plate
from iwpod.decode import decode_single
from iwpod.preprocess import letterbox, unletterbox_quad

#: IoU thresholds for the accuracy curve (COCO-style 0.50-0.95 sweep).
IOU_THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.5, 1.0, 0.05))

#: Thresholds for the operating-point (F1) sweep.
SWEEP_THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.10, 0.95, 0.05))

#: Decode floor for sweep bookkeeping. Top-1 by confidence is identical at any
#: threshold that keeps it, so decoding once here reproduces the operating
#: point exactly while also recording sub-threshold confidences for the sweep.
BASE_THRESHOLD = 0.05


class EvalResult(dict):
    """Val metrics with attribute access + summary formatting."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k) from None


def read_quad(txt):
    with open(txt) as f:
        line = f.readline().strip()
    parts = [p for p in line.split(",") if p != ""]
    n = int(parts[0])
    vals = list(map(float, parts[1:1 + 2 * n]))
    return np.array(vals, dtype=float).reshape(2, n)


def quad_iou(a, b, size=256):
    def mask(pts):
        m = np.zeros((size, size), np.uint8)
        p = (pts * size).astype(np.int32).T.reshape(-1, 1, 2)
        cv2.fillPoly(m, [p], 1)
        return m
    ma, mb = mask(a), mask(b)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / max(1, union))


def poly_area(qn):
    """Normalized polygon area (shoelace) for a (2,4) quad."""
    x, y = qn[0], qn[1]
    return float(0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i]
                               for i in range(4))))


def entries_from_annotated_dir(data_dir):
    """(jpg, gt) pairs for a dir of image + sibling-.txt pairs (eval format).

    Missing or empty `.txt` becomes a known negative (gt=None) so the FP rate
    is measurable; matches dataset.image_label_loader background policy.
    Malformed non-empty labels are skipped (not counted as negatives).
    """
    files = []
    if os.path.isdir(data_dir):
        for name in os.listdir(data_dir):
            if name.lower().endswith(IMAGE_EXTS):
                files.append(os.path.join(data_dir, name))
    entries = []
    for jpg in sorted(files):
        txt = os.path.splitext(jpg)[0] + ".txt"
        if not os.path.isfile(txt):
            entries.append((jpg, None))
            continue
        try:
            with open(txt) as f:
                content = f.read()
        except OSError as e:
            logger.warning(f"Unreadable label '{txt}': {e} — skipped (invalid)")
            continue
        if not content.strip():
            entries.append((jpg, None))
            continue
        try:
            quad = read_quad(txt)
        except (ValueError, IndexError) as e:
            logger.warning(f"Unreadable label '{txt}': {e} — skipped (invalid)")
            continue
        if not is_real_plate(quad):
            entries.append((jpg, None))
        else:
            entries.append((jpg, quad))
    return entries


def entries_from_loader_entries(loader_entries):
    """Convert image_label_loader entries to (jpg, gt_or_None) eval entries.

    Background entries (fake degenerate plate) become known negatives.
    """
    out = []
    for jpg, shapes in loader_entries:
        gt = None
        for s in shapes:
            pts = np.asarray(s.pts, dtype=float)
            if pts.shape == (2, 4) and is_real_plate(pts):
                gt = pts
                break
        out.append((jpg, gt))
    return out


@torch.no_grad()
def evaluate_quads(model, entries, size_px=384, threshold=0.3, device=None,
                   topk=1, progress=False, extra_thresholds=None):
    """Run decode + quad-IoU over entries. Returns an EvalResult.

    Gating contract: `decode_single` size/NMS gates apply BEFORE sweep/AP
    bookkeeping. A high-logit but degenerate/negative-area quad yields
    `found=[]` -> conf 0.0 / IoU 0.0, so it contributes no FP and ranks last
    in AP. Sweep and mAP therefore measure detector behavior, not raw logits.

    One forward pass per image. Extra operating thresholds are derived from
    stored (conf, IoU) pairs — exact values, not snapped to 0.05.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    was_training = model.training
    model.eval()
    base_thr = min(threshold, BASE_THRESHOLD)

    ious, rmses_det = [], []
    confs, raw_ious, areas, is_pos, rmses_all = [], [], [], [], []
    dets = dets_pos = tp50 = fp = 0
    infer_ms_total = 0.0
    n = 0
    it = entries
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(entries, desc="eval", unit="img")
        except ImportError:
            pass
    for jpg, gt in it:
        img = cv2.imread(jpg)
        if img is None:
            logger.warning(f"Cannot read image '{jpg}' — skipped")
            continue
        h0, w0 = img.shape[:2]
        canvas, s, dx, dy = letterbox(img, size_px)
        inp = canvas.astype(np.float32) / 255.0
        t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
        t0 = time.perf_counter()
        pred = model(t).squeeze(0).cpu().numpy()
        found = decode_single(pred, size_px, size_px, threshold=base_thr,
                              from_logits=True, topk=topk)
        infer_ms_total += (time.perf_counter() - t0) * 1000.0
        n += 1
        conf = found[0][1] if found else 0.0
        if gt is None:  # known negative: only false positives count
            if conf >= threshold:
                fp += 1
                dets += 1
            confs.append(conf)
            raw_ious.append(0.0)
            areas.append(0.0)
            is_pos.append(False)
            rmses_all.append(float("nan"))
            continue
        if found:
            quad, _c = found[0]
            q0 = unletterbox_quad(quad, size_px, s, dx, dy, w0, h0)
            qn = q0 / np.array([[w0], [h0]])
            iou = quad_iou(qn, gt)
            rmse = float(np.sqrt(((qn - gt) ** 2).mean()))
            area = poly_area(gt)
        else:
            iou, rmse, area = 0.0, 0.0, poly_area(gt)
        confs.append(conf)
        raw_ious.append(iou)
        areas.append(area)
        is_pos.append(True)
        rmses_all.append(rmse)
        if conf >= threshold:
            dets += 1
            dets_pos += 1
            ious.append(iou)
            if found:
                rmses_det.append(rmse)
            if iou > 0.5:
                tp50 += 1
        else:
            ious.append(0.0)

    if was_training:
        model.train()
    ious = np.array(ious, dtype=float)
    confs = np.array(confs, dtype=float)
    raw_ious = np.array(raw_ious, dtype=float)
    areas = np.array(areas, dtype=float)
    is_pos = np.array(is_pos, dtype=bool)
    n_pos = int(is_pos.sum())
    curve = {t: float(np.mean(ious > t)) if n_pos else 0.0 for t in IOU_THRESHOLDS}
    maps = map_scores(confs, raw_ious, is_pos)
    rmse_detected = float(np.mean(rmses_det)) if rmses_det else 0.0
    extra = list(extra_thresholds or [])
    res = EvalResult(
        n=n, dets=dets, fp=fp,
        det_rate=float(dets_pos / max(1, n_pos)),
        recall=float(tp50 / max(1, n_pos)),  # recall@IoU>0.5 at operating thr
        recall50=float(tp50 / max(1, n_pos)),
        mean_iou=float(ious.mean()) if n_pos else 0.0,
        iou50=curve.get(0.5, 0.0), iou70=curve.get(0.7, 0.0),
        iou_curve=curve,
        map50=maps["map50"], map=maps["map"],
        ap_curve=maps["curve"],
        # Localization quality given a hit. Misses are captured by recall.
        rmse_detected=rmse_detected,
        rmse_n=int(len(rmses_det)),
        rmse=rmse_detected,  # alias
        infer_ms=float(infer_ms_total / max(1, n)),
        area=area_split(confs, ious, areas, is_pos, threshold),
        sweep=threshold_sweep(confs, raw_ious, is_pos, threshold, extra),
        confs=confs, raw_ious=raw_ious, is_pos=is_pos,
    )
    return res


def average_precision(confs, matched, n_pos):
    """Area under the monotone precision envelope (COCO all-points style).

    confs/matched: per-candidate arrays (matched = TP flag per candidate).
    Single class here, so this *is* the AP that mAP averages (trivially).
    """
    if n_pos <= 0 or len(confs) == 0:
        return 0.0
    order = np.argsort(-np.asarray(confs, dtype=float), kind="stable")
    tp = np.asarray(matched, dtype=bool)[order].astype(float)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall = tp_cum / max(1, n_pos)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def map_scores(confs, raw_ious, is_pos):
    """AP per IoU threshold + mAP@50 / mAP@50-95.

    Match rule mirrors the sweep: positives with IoU > t are TP; every
    candidate ranks by confidence regardless of the operating threshold,
    so the scores are threshold-free (unlike recall@IoU at a fixed thr).
    With top-k=1 decode each crop contributes at most one candidate, which
    matches the single-plate-per-crop task.
    """
    confs = np.asarray(confs, dtype=float)
    raw_ious = np.asarray(raw_ious, dtype=float)
    is_pos = np.asarray(is_pos, dtype=bool)
    pos_idx = np.where(is_pos)[0]
    n_pos = len(pos_idx)
    curve = {}
    for t in IOU_THRESHOLDS:
        matched = np.zeros(len(confs), dtype=bool)
        if n_pos:
            hit = pos_idx[raw_ious[pos_idx] > t]
            matched[hit] = True
        curve[t] = average_precision(confs, matched, n_pos)
    aps = [curve[t] for t in IOU_THRESHOLDS]
    return {"curve": curve, "map50": curve.get(0.5, 0.0),
            "map": float(sum(aps) / len(aps)) if aps else 0.0}


def area_split(confs, ious, areas, is_pos, threshold):
    """Recall/mIoU by GT plate-area tercile (YOLOX area-row analog).

    Distance-to-camera is the main variance axis for plates; terciles keep the
    buckets balanced without hand-tuned cutoffs.
    """
    out = {}
    pos_areas = areas[is_pos]
    if len(pos_areas) == 0:
        return {k: dict(n=0, recall=0.0, miou=0.0) for k in ("small", "medium", "large")}
    q1, q2 = np.quantile(pos_areas, [1 / 3, 2 / 3])
    idx = np.where(is_pos)[0]
    # ious[] is ordered over positives only, aligned with idx
    for name in ("small", "medium", "large"):
        if name == "small":
            members = [i for i in idx if areas[i] <= q1]
        elif name == "medium":
            members = [i for i in idx if q1 < areas[i] <= q2]
        else:
            members = [i for i in idx if areas[i] > q2]
        if not members:
            out[name] = dict(n=0, recall=0.0, miou=0.0)
            continue
        pos_index = {v: k for k, v in enumerate(idx.tolist())}
        det = sum(1 for i in members if confs[i] >= threshold)
        out[name] = dict(n=len(members), recall=det / len(members),
                         miou=float(np.mean([ious[pos_index[i]] for i in members])))
    return out


def threshold_sweep(confs, raw_ious, is_pos, operating, extra_thresholds=None):
    """Precision/recall/F1 over thresholds (YOLOX confidence-analysis analog).

    Match = IoU > 0.5 on positives; negatives with conf >= t count as FPs.
    `extra_thresholds` are evaluated exactly (not snapped to 0.05).
    """
    pos_idx = np.where(is_pos)[0]
    neg_idx = np.where(~is_pos)[0]
    thrs = list(SWEEP_THRESHOLDS)
    if extra_thresholds:
        for t in extra_thresholds:
            thrs.append(round(float(t), 4))
    thrs.append(round(float(operating), 4))
    thrs = sorted(set(thrs))
    table = []
    for t in thrs:
        det_pos = pos_idx[confs[pos_idx] >= t] if len(pos_idx) else np.array([], dtype=int)
        matched = det_pos[raw_ious[det_pos] > 0.5] if len(det_pos) else det_pos
        fp = int((confs[neg_idx] >= t).sum()) if len(neg_idx) else 0
        prec = len(matched) / max(1, len(det_pos) + fp)
        rec = len(matched) / max(1, len(pos_idx))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        table.append(dict(thr=float(t), prec=float(prec), rec=float(rec), f1=float(f1)))
    best = max(table, key=lambda r: (r["f1"], r["rec"]))
    op = min(table, key=lambda r: abs(r["thr"] - operating))
    return dict(table=table, best_thr=best["thr"], best_f1=best["f1"],
                best_prec=best["prec"], best_rec=best["rec"],
                op_thr=op["thr"], op_f1=op["f1"])


def format_summary(r, prefix="val"):
    curve = " ".join(f"@{t:.2f}={v:.3f}" for t, v in r["iou_curve"].items()
                      if t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95))
    return (f"{prefix}: n={r['n']} dets={r['dets']} fp_neg={r['fp']} "
            f"recall={r['recall']:.3f} det_rate={r.get('det_rate', r['recall']):.3f} "
            f"mIoU={r['mean_iou']:.3f} {curve} "
            f"mAP@50={r['map50']:.3f} mAP@50-95={r['map']:.3f} "
            f"RMSE_det={r.get('rmse_detected', r['rmse']):.4f} "
            f"(n={r.get('rmse_n', 0)}) infer={r['infer_ms']:.1f}ms/img")


def format_block(r, epoch=None, epochs=None, size_px=None, threshold=None,
                 with_sweep=False):
    """Multi-line eval block (YOLOX summary analog)."""
    bar = "=" * 70
    if epoch is not None and epochs is not None:
        head = f"EVAL epoch {epoch}/{epochs}"
    else:
        head = "EVAL"
    if size_px is not None and threshold is not None:
        head += f" (size={size_px} thr={threshold:.2f})"
    curve = "  ".join(f"@{t:.2f}={v:.3f}" for t, v in r["iou_curve"].items()
                      if t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95))
    a = r["area"]
    area = " | ".join(f"{k}(n={a[k]['n']}): recall={a[k]['recall']:.3f} "
                      f"miou={a[k]['miou']:.3f}" for k in ("small", "medium", "large"))
    lines = [bar, head, "-" * 70,
             f"recall={r['recall']:.3f}  det_rate={r.get('det_rate', r['recall']):.3f}  "
             f"mIoU={r['mean_iou']:.3f}  RMSE_det={r.get('rmse_detected', r['rmse']):.4f} "
             f"(n={r.get('rmse_n', 0)})  dets={r['dets']}  fp_neg={r['fp']}  "
             f"infer={r['infer_ms']:.1f}ms/img",
             f"mAP@50={r['map50']:.3f}  mAP@50-95={r['map']:.3f}",
             f"IoU: {curve}",
             f"area: {area}"]
    if with_sweep:
        s = r["sweep"]
        lines.append(f"thr sweep (F1): best={s['best_thr']:.2f} "
                     f"(F1={s['best_f1']:.4f} P={s['best_prec']:.4f} R={s['best_rec']:.4f})  "
                     f"@op={s['op_thr']:.2f} F1={s['op_f1']:.4f}")
    lines.append(bar)
    return "\n".join(lines)
