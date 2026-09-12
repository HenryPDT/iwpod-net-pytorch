"""Dataset preparation: raw scene folders -> train/val dataset layout.
Each directory containing image pairs is one scene; every scene is split
independently (seeded shuffle, --ratio to train) so small scenes appear in
both splits proportionally instead of being swallowed whole into one side.

Annotation contract: quad `.txt` files (`4,x0..x3,y0..y3,,` per line,
multi-line = multi-plate). Coordinates may be normalized [0,1] or pixels
(auto-detected per file and normalized). Empty/missing `.txt` = background
sample (kept). Unparseable files are skipped with a counted reason.
"""
import csv
import hashlib
import os
import random
import shutil

IMG_EXTS = (".jpg", ".jpeg", ".png")


def is_image(path):
    return path.lower().endswith(IMG_EXTS)


def _scene_id(scene, input_root):
    """Collision-safe id: relative path with separators replaced by `__`."""
    rel = os.path.relpath(scene, input_root)
    rel = rel.replace("\\", "/").strip("/.")
    if not rel or rel == ".":
        return os.path.basename(os.path.normpath(input_root)) or "scene"
    return rel.replace("/", "__")


def find_scenes(root):
    """Walk root; every dir holding >=1 image is a scene. Sorted for determinism."""
    scenes = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(is_image(f) for f in filenames):
            scenes.append(dirpath)
    return sorted(scenes)


def parse_quad_file(txt_path, img_w, img_h):
    """Returns (quads, error). quads = list of ((2,4) float array, text) in [0,1].

    Line parsing delegates to label.parse_shape_line (single parser for the
    repo); this layer adds quad-specific policy: only n == 4 kept, pixel
    coords auto-normalized, range-checked. Vehicle labels are preserved so
    training keeps the right aspect bucket (bike vs car).
    """
    from iwpod.constants import normalize_quad_order
    from iwpod.label import ShapeParseError, parse_shape_line

    try:
        with open(txt_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        return None, f"unreadable: {e}"
    if not lines:
        return [], None  # empty file = background sample
    quads = []
    for lineno, line in enumerate(lines, 1):
        try:
            pts, n, text = parse_shape_line(line, lineno)
        except ShapeParseError as e:
            return None, str(e)
        if n != 4:
            return None, f"line {lineno}: expected 4 corners, got n={n}"
        q = [pts[0].tolist(), pts[1].tolist()]
        flat = q[0] + q[1]
        if any(v != v or v in (float("inf"), float("-inf")) for v in flat):
            return None, f"line {lineno}: NaN/inf coordinate"
        if any(v > 1.5 for v in flat):
            if img_w <= 0 or img_h <= 0:
                return None, f"line {lineno}: pixel coords but bad image size"
            q = [[v / img_w for v in q[0]], [v / img_h for v in q[1]]]
            flat = q[0] + q[1]
        if any(v < 0.0 or v > 1.0 for v in flat):
            return None, f"line {lineno}: coordinate outside [0,1]"
        ordered = normalize_quad_order(q)
        quads.append((ordered.tolist(), text))
    return quads, None


def collect_pairs(scene_dir):
    """Image pairs in a scene dir (non-recursive). Returns [(jpg, txt|None)]."""
    pairs = []
    for name in sorted(os.listdir(scene_dir)):
        full = os.path.join(scene_dir, name)
        if os.path.isfile(full) and is_image(name):
            txt = os.path.splitext(full)[0] + ".txt"
            pairs.append((full, txt if os.path.isfile(txt) else None))
    return pairs


def _read_image_size(jpg_path):
    import cv2
    img = cv2.imread(jpg_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def prepare_dataset(input_root, output_root, ratio=0.8, seed=42,
                    overwrite=False, image_reader=None):
    """Convert scene folders to <output>/{train,val}/. Returns stats dict.

    ratio: train fraction per scene (default 0.8; remaining 1 - ratio to val).
    image_reader(jpg_path) -> (w, h) | None; injectable for tests (defaults
    to cv2). Files are always copied; the source tree is never modified.
    Output filenames are prefixed `<scene>__` to prevent cross-scene collisions.
    """
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")

    read_size = image_reader or _read_image_size
    if os.path.exists(output_root):
        if not overwrite:
            raise RuntimeError(
                f"Output '{output_root}' exists: pass --overwrite or choose another --output.")
    train_dir = os.path.join(output_root, "train")
    val_dir = os.path.join(output_root, "val")

    stats = {"scenes": [], "train": 0, "val": 0, "invalid": 0, "background": 0}
    manifest = []
    for scene in find_scenes(input_root):
        scene_name = _scene_id(scene, input_root)
        pairs = collect_pairs(scene)
        if not pairs:
            continue
        # Phase 1: validate everything up front so the split applies to
        # usable data (not diluted by files dropped later).
        valid, per_scene_invalid = [], 0
        for jpg, txt in pairs:
            size = read_size(jpg)
            if size is None:
                per_scene_invalid += 1
                continue
            img_w, img_h = size
            quads, error = (parse_quad_file(txt, img_w, img_h) if txt
                            else ([], None))
            if error is not None:
                per_scene_invalid += 1
                continue
            if not quads:
                stats["background"] += 1
            valid.append((jpg, txt, quads))
        # Phase 2: per-scene split of the valid pairs.
        order = list(valid)
        rng = random.Random(
            seed + int(hashlib.md5(scene_name.encode()).hexdigest(), 16) % (2 ** 16))
        rng.shuffle(order)
        n_val = int(len(order) * (1.0 - ratio) + 1e-8)
        if len(order) >= 2 and ratio < 1.0:
            # Tiny scenes (N<5 at 0.8) would otherwise get 0 val; keep >=1
            # val sample whenever a split is requested so every scene validates.
            n_val = max(1, n_val)
        n_val = min(n_val, len(order))
        train_items, val_items = order[n_val:], order[:n_val]
        per_scene = {"scene": scene_name, "train": 0, "val": 0,
                     "invalid": per_scene_invalid}
        # Phase 3: materialize (quads already validated; rewritten canonical).
        for split, subset in (("train", train_items), ("val", val_items)):
            dest = train_dir if split == "train" else val_dir
            for jpg, _txt, quads in subset:
                stem = os.path.splitext(os.path.basename(jpg))[0]
                base = f"{scene_name}__{stem}"
                os.makedirs(dest, exist_ok=True)
                shutil.copy2(jpg, os.path.join(dest, base + os.path.splitext(jpg)[1]))
                lines = []
                for q, text in quads:
                    flat = [f"{v:.6f}" for v in q[0] + q[1]]
                    lines.append("4," + ",".join(flat) + f",{text},")
                with open(os.path.join(dest, base + ".txt"), "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                per_scene[split] += 1
                manifest.append((base, scene_name, split))
        stats["scenes"].append(per_scene)
        stats["train"] += per_scene["train"]
        stats["val"] += per_scene["val"]
        stats["invalid"] += per_scene["invalid"]

    os.makedirs(output_root, exist_ok=True)
    with open(os.path.join(output_root, "split_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "scene", "split"])
        w.writerows(manifest)
    return stats
