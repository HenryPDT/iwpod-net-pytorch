import os
import random
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from .constants import HSV_H_DELTA, HSV_S_DELTA, HSV_V_DELTA, IMAGE_EXTS, SIDE, is_real_plate, normalize_quad_order
from .label import Label
from .projection_utils import find_T_matrix, getRectPts, perspective_transform
from .utils import getWH, hsv_transform, im2single

#
#  Use UseBG if you want to pad distorted images with bakcground data
#
UseBG = True
bgimages = []
dim0 = 208
#: Per-axis 3D rotation limits (degrees, half-range) applied in
#: perspective_transform order: Rx pitch (mount height), Ry yaw (side angle),
#: Rz roll (in-plane). Calibrated per axis, not copied from the paper's
#: +/-80 deg (CCPD tilt extremes our fixed entry/exit cameras never see):
#: yaw gets headroom for gate turns (plates stay legible to ~60 deg, slivers
#: beyond); pitch stays near the original mount-height range (past ~45 deg
#: you're looking at roofs); roll is cheap (no foreshortening, plate stays
#: legible when rotated) so it gets mount-error margin.
MAX_ROT_DEG = np.array([45., 60., 30.])
#: Cap on the sum of |angles| (degrees); over-budget draws are rescaled
#: direction-preserving by limit_angles. Sized to the joint corner
#: (45+60+30) so the per-axis ranges can actually realize together.
MAX_ROT_SUM = 135.
#: Horizontal-flip probability in augment_sample. Mirror preserves the
#: top/bottom corner convention (only TL<->TR, BL<->BR swap); vertical flip
#: would break it, so only horizontal is used.
HFLIP_PROB = 0.5


def _bg_dir():
    return Path(__file__).resolve().parent.parent / 'bgimages'


def _ensure_bgimages():
    """Lazy-load BG images (cross-platform, case-insensitive, import-safe)."""
    global bgimages
    if bgimages or not UseBG:
        return bgimages
    bgdir = _bg_dir()
    imglist = []
    for name in os.listdir(bgdir) if bgdir.is_dir() else []:
        if name.lower().endswith(IMAGE_EXTS):
            imglist.append(str(bgdir / name))
    for im in sorted(set(imglist)):
        img = cv2.imread(im)
        if img is None:
            continue
        factor = max(1, dim0 / min(img.shape[0:2]))
        img = cv2.resize(img, (0, 0), fx=factor, fy=factor).astype('float32') / 255
        bgimages.append(img)
    return bgimages


# Eager load best-effort (keeps old behaviour) but never crash import.
try:
    _ensure_bgimages()
except Exception:
    bgimages = []


def random_crop(img, width, height):
    #
    #  generates random crop of img with desired size
    #
    h, w = img.shape[:2]
    if h < height or w < width:
        img = cv2.resize(img, (max(w, width), max(h, height)), interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
    top = 0 if h <= height else int(np.random.randint(0, h - height + 1))
    left = 0 if w <= width else int(np.random.randint(0, w - width + 1))
    return img[top:top + height, left:left + width, :]


def GetCentroid(pts):
    #
    #  Gets centroids of a quadrilateral
    #
    return np.mean(pts, 1)


def ShrinkQuadrilateral(pts, alpha=0.75):
    #
    #  Shribks quadtrilateral by a factor alpha
    #
    centroid = GetCentroid(pts)
    temp = centroid + alpha * (pts.T - centroid)
    return temp.T


def LinePolygonEdges(pts):
    #
    # Finds the line equations of the polygon edges (given the verices in clockwise order)
    #
    lines = []
    for i in range(4):
        x1 = np.hstack((pts[:, i], 1))
        x2 = np.hstack((pts[:, (i + 1) % 4], 1))
        lines.append(np.cross(x1, x2))
    return lines


def insidePolygon(pt, lines):
    #
    #  Checks if a point pt is inside quadrilateral given by pts.
    #  Winding-agnostic: accepts all-non-negative OR all-non-positive signs.
    #
    pth = np.hstack((pt, 1))
    pos = neg = False
    for i in range(len(lines)):
        sig = np.dot(pth, lines[i])
        if sig > 1e-9:
            pos = True
        elif sig < -1e-9:
            neg = True
        if pos and neg:
            return False
    return True


def labels2output_map(labelist, lpptslist, dim, stride, alpha=0.75):
    #
    #  Generates outpmut map with binary (classification) labels and quadrilateral corners (regression)
    #  label is the bounding box of the quadrilateral, and its locations are given in a list of plates
    #   lpptslist
    #

    side = SIDE
    outsize = int(dim / stride)

    #
    # Prepares GT map with 9 channels
    #
    Y = np.zeros((outsize, outsize, 2 * 4 + 1), dtype='float32')
    MN = np.array([outsize, outsize])
    WH = np.array([dim, dim], dtype=float)

    #
    #  Scans all annotated LPs in the image
    #
    for i in range(0, len(labelist)):
        # Belt-and-braces TL-first: annotation/fit order is arbitrary, but the
        # affine targets, decode and warp all assume TL,TR,BR,BL. Export
        # normalizes too (X-AnyLabeling custom_to_wpod); this covers
        # hand-written or third-party files. No-op on compliant quads.
        # Clip to [0,1]: edge plates have true corners outside the frame;
        # training targets the visible extent (mirrors exporter clamping).
        lppts = np.clip(normalize_quad_order(lpptslist[i]), 0.0, 1.0)
        label = labelist[i]
        tlx, tly = np.floor(np.maximum(label.tl(), 0.) * MN).astype(int).tolist()
        brx, bry = np.ceil(np.minimum(label.br(), 1.) * MN).astype(int).tolist()
        p_WH = lppts * WH.reshape((2, 1))
        p_MN = p_WH / stride
        pts2 = (ShrinkQuadrilateral(lppts, alpha).T * MN).T
        lines = LinePolygonEdges(pts2)
        for x in range(tlx, brx):
            for y in range(tly, bry):
                mn = np.array([float(x) + .5, float(y) + .5])
                if insidePolygon(mn, lines):
                    p_MN_center_mn = p_MN - mn.reshape((2, 1))
                    p_side = p_MN_center_mn / side
                    Y[y, x, 0] = 1.
                    Y[y, x, 1:] = p_side.T.flatten()
        # Always set a true label at centroid if not fake LP (per plate).
        if is_real_plate(lppts):
            cc = np.array(np.round(GetCentroid(p_MN) - 0.5), np.int32)
            x = max(0, min(cc[0], outsize - 1))
            y = max(0, min(cc[1], outsize - 1))
            mn = np.array([float(x) + .5, float(y) + .5])
            p_MN_center_mn = p_MN - mn.reshape((2, 1))
            p_side = p_MN_center_mn / side
            Y[y, x, 0] = 1.
            Y[y, x, 1:] = p_side.T.flatten()
    return Y



def pts2ptsh(pts):
    #
    #  Gets homogeneous coordinates (ndarray, no np.matrix)
    #
    return np.concatenate((np.asarray(pts, dtype=float), np.ones((1, pts.shape[1]))), 0)


def project(I, T, pts, dim):
    #
    #  Projects image I and points pts according to matrix T
    #
    ptsh = np.concatenate((np.asarray(pts, dtype=float), np.ones((1, 4))), 0)
    ptsh = np.matmul(T, ptsh)
    ptsh = ptsh / ptsh[2]
    ptsret = ptsh[:2]
    ptsret = ptsret / dim
    Iroi = cv2.warpPerspective(I, T, (dim, dim), borderValue=.0, flags=cv2.INTER_CUBIC)
    return Iroi, ptsret


def project_all(I, T, ptslist, dim, bgimages=None):
    #
    #  Warps image I to desired dimensions using matrix T. if bgimage is not empty,
    #  completes with background
    #  Also projects LP coordinates given in ptslist to keep coherence
    #
    if bgimages is None:
        bgimages = _ensure_bgimages()
    #
    outptslist = []
    #
    #  Scans annotated LPs and warps them
    #
    for pts in ptslist:
        ptsh = np.concatenate((np.asarray(pts, dtype=float), np.ones((1, 4))), 0)
        ptsh = np.matmul(T, ptsh)
        ptsh = ptsh / ptsh[2]
        ptsret = ptsh[:2]
        ptsret = ptsret / dim
        outptslist.append(np.array(ptsret))
    #
    #  Warps input image (possibly padding with BG images)
    #
    Iroi = cv2.warpPerspective(I, T, (dim, dim), borderValue=(.5,.5,.5), flags=cv2.INTER_CUBIC)
    if len(bgimages) > 0:
        bgimage = bgimages[int(np.random.rand()*len(bgimages))]
        if bgimage.shape[0] < dim or bgimage.shape[1] < dim:
            bgimage = cv2.resize(bgimage, (dim, dim), interpolation=cv2.INTER_CUBIC)
        else:
            bgimage = random_crop(bgimage, dim, dim)
        bw = np.ones(I.shape, dtype=np.float32)
        bw = cv2.warpPerspective(bw, T, (dim, dim), borderValue=(0, 0, 0), flags=cv2.INTER_LINEAR)
        # bw is (dim,dim,3); threshold soft warped edges
        mask = (bw.mean(axis=2, keepdims=True) < 0.5)
        mask3 = np.repeat(mask, 3, axis=2)
        Iroi[mask3] = bgimage[mask3]
    return Iroi, outptslist

def limit_angles(angles, maxsum):
    """Cap total 3D rotation magnitude, preserving direction.

    The legacy rescale divided by the signed sum, which collapses the vector
    when axes cancel (e.g. [65,-65,10]) and explodes near a zero sum.
    Proportional rescale keeps the sampled direction and caps |.|_1 at maxsum.
    """
    total = float(np.abs(np.asarray(angles, dtype=float)).sum())
    if total <= maxsum:
        return angles
    return np.asarray(angles, dtype=float) * (maxsum / total)


def jitter_crop_window(I, ptslist, max_shift=0.15):
    """Simulate upstream vehicle-detector box error: translate the crop window.

    Shifts the image and every plate quad coherently in pixel space (relative
    pts are scaled to pixels, shifted, scaled back). Falls back to the
    original frame if the first plate's centroid leaves it.
    """
    h, w = I.shape[:2]
    tx = float(np.random.uniform(-max_shift, max_shift)) * w
    ty = float(np.random.uniform(-max_shift, max_shift)) * h
    M = np.array([[1., 0., -tx], [0., 1., -ty]], dtype=np.float32)
    J = cv2.warpAffine(I, M, (w, h), borderValue=(.5, .5, .5),
                       flags=cv2.INTER_LINEAR)
    shifted = []
    for pts in ptslist:
        q = np.asarray(pts, dtype=float).copy()
        if q.size:
            q = (q * np.array([[w], [h]]) - np.array([[tx], [ty]])) / np.array([[w], [h]])
        shifted.append(q)
    for c in shifted:
        if c.size and ((c.mean(axis=1) < 0).any() or (c.mean(axis=1) > 1).any()):
            return I, ptslist  # a plate left the frame: keep the original
    return J, shifted


def guarded_cutout(Iroi, ptslist, dim, p=0.3, max_holes=4, hole=30,
                   max_plate_cover=0.5, fill=0.5):
    """Few small gray holes for occlusion robustness, never swallowing the plate.

    Hole/plate overlap is measured against the first plate's bbox (pixels);
    holes covering more than `max_plate_cover` of it are skipped.
    """
    if np.random.rand() > p:
        return Iroi
    out = Iroi.copy()
    plate = np.asarray(ptslist[0], dtype=float) * dim if len(ptslist) else None
    if plate is not None and plate.size:
        px0, py0, px1, py1 = plate[0].min(), plate[1].min(), plate[0].max(), plate[1].max()
        plate_area = max(1.0, (px1 - px0) * (py1 - py0))
    else:
        px0 = py0 = px1 = py1 = plate_area = None
    for _ in range(int(np.random.randint(1, max_holes + 1))):
        hw = int(np.random.uniform(10, hole))
        hh = int(np.random.uniform(10, hole))
        x0 = int(np.random.uniform(0, max(1, dim - hw)))
        y0 = int(np.random.uniform(0, max(1, dim - hh)))
        if plate_area is not None:
            ix0, iy0 = max(x0, px0), max(y0, py0)
            ix1, iy1 = min(x0 + hw, px1), min(y0 + hh, py1)
            overlap = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            if overlap / plate_area > max_plate_cover:
                continue
        out[y0:y0 + hh, x0:x0 + hw] = fill
    return out


_pixel_stage = None


def _pixel_stage_fn():
    """Albumentations pixel-only stage (no keypoints: the label map is built
    downstream from the warped pts, so pixel ops need no label plumbing)."""
    global _pixel_stage
    if _pixel_stage is None:
        import albumentations as A
        _pixel_stage = A.Compose([
            # IR/night cameras switch to grayscale: drop chroma first so the
            # later photometric ops still vary the gray images. Any vehicle or
            # plate color must remain detectable without hue cues.
            A.ToGray(p=0.15),
            A.OneOf([A.MotionBlur(blur_limit=(3, 11), p=1.0),
                     A.GaussianBlur(blur_limit=(3, 5), p=1.0)], p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.OneOf([A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                     A.ImageCompression(quality_range=(40, 85), p=1.0)], p=0.3),
        ])
    return _pixel_stage


def flip_image_and_ptslist(I, ptslist):    #
    #  Applies random flip to image and labels
    #
    I = cv2.flip(I, 1)
    out = []
    for pts in ptslist:
        pts = np.asarray(pts, dtype=float).copy()
        pts[0] = 1. - pts[0]
        idx = [1, 0, 3, 2]
        pts = pts[..., idx]
        out.append(pts)
    return I, out


def _sample_angles(maxangle=None, maxsum=None):
    """Sample a 3D rotation within production limits (degrees)."""
    if maxangle is None:
        maxangle = 2 * MAX_ROT_DEG
    if maxsum is None:
        maxsum = MAX_ROT_SUM
    angles = (np.random.rand(3) - 0.5) * np.asarray(maxangle, dtype=float)
    return limit_angles(angles, maxsum)


def augment_sample(I, shapelist, dim, maxangle=None, maxsum=None,
                   detail_boost=0.0):
    #
    #  Main augmentation function. Generates an augmented version
    #  of input image I and the corresponding LP corners given in shapelist
    #

    #
    #  Input is image I, list of shape elements (shape), and input dim
    #

    #
    #  Gets first LP corners and label
    #
    ptslist = [np.asarray(entry.pts, dtype=float).copy() for entry in shapelist]
    pts = ptslist[0]

    vtype = shapelist[0].text
    angles = _sample_angles(maxangle, maxsum)

    #
    # Normalizes intensities to [0,1]
    #
    I = im2single(I)
    #
    #  Possible negative of the image
    #
    if np.random.uniform(0,1) < 0.05:
        I = 1 - I

    #
    #  Upstream-detector box jitter: translate the crop window (image + all
    #  quads coherently). Blur/noise/compression live in the post-warp pixel
    #  stage below, so the legacy pre-warp blur was removed (no stacked blur).
    #
    I, ptslist = jitter_crop_window(I, ptslist)
    pts = ptslist[0]  # keep the warp target coherent with the shifted frame

    #
    #  Gets image dimensions
    #
    iwh = getWH(I.shape)

    #
    #  Checks is annotation is a real or fake plate
    #

    if is_real_plate(pts):  # real plate: diagonal > REAL_PLATE_DIAG

        for i in range(len(ptslist)):
            #
            #  LP region from relative to absolute coordinates
            #
            ptslist[i] = ptslist[i] * iwh.reshape((2, 1))

        #
        #  Target aspect ratio of the LP (bike or car)
        #
        if vtype == 'bike':
            whratio = random.uniform(1.25, 2.5)
        else:
            whratio = random.uniform(2.5, 4.5)

        #
        #  Width of LP in training image
        #
        dim0 = 208 # augments data w.r.t. a fixed resolution

        #
        #  Defines range of LP widths w.r.t to baseline resolution dim0 = 208
        #  detail_boost (DPOD-style): per-sample extra scale jitter so small /
        #  detailed plates are oversampled, e.g. 0.5 = ±50% on wsiz.
        #
        wsiz = random.uniform(dim0*0.2, dim0*1.0)
        if detail_boost and detail_boost > 0:
            wsiz = float(np.clip(
                wsiz * random.uniform(1.0 - detail_boost, 1.0 + detail_boost),
                dim0 * 0.1, float(dim)))

        #
        #  Defines height based on width and aspect ratio
        #
        hsiz = wsiz/whratio

        #
        #  Defines horizontal and vertical offsets
        #
        dx = random.uniform(0., max(0.0, dim - wsiz))
        dy = random.uniform(0., max(0.0, dim - hsiz))

        #
        #  Warps annotated plate to a rectified rectangle - frontal view
        #
        pph = getRectPts(dx, dy, dx+wsiz, dy+hsiz)
        pts = pts*iwh.reshape((2,1))
        T = find_T_matrix(pts2ptsh(pts), pph)

        #
        # Finds 3D rotation matrix based on angles
        #
        H = perspective_transform((dim,dim), angles=angles)

        #
        #  Applies 3D rotation to rectification transform
        #
        H = np.matmul(H,T)

        #
        # projects images and labels according to 3D rotation
        #
        Iroi, ptslist = project_all(I, H, ptslist, dim)
        pts = ptslist[0]
    else:  # if fake plate
        #
        #  Random BG crop if no plate is present, resizes in x and y
        #
        h, w = I.shape[:2]
        rfactorx = max(dim / max(1, w), 0.5 + 0.5 * np.random.rand())
        rfactory = max(dim / max(1, h), 0.5 + 0.5 * np.random.rand())
        Iroi = cv2.resize(I, (0, 0), fx=rfactorx, fy=rfactory)
        Iroi = random_crop(Iroi, dim, dim)

    #
    #   Just a sanity check, test should never hold
    #
    if (Iroi.shape[0] < dim):
        logger.warning(f"Iroi height {Iroi.shape[0]} < dim {dim} after crop — skipping frame")

    #
    #  Random horizontal flip (mirror): valid for plates, keeps the
    #  top/bottom corner convention. Applied post-warp on dim-normalized pts.
    #
    if np.random.rand() < HFLIP_PROB:
        Iroi, ptslist = flip_image_and_ptslist(Iroi, ptslist)

    #
    #  Set of non-geometric transforms
    #

    #
    # Color transformations in HSV space
    #
    hsv_mod = np.array([
        (np.random.rand() - 0.5) * 2.0 * HSV_H_DELTA,
        (np.random.rand() - 0.5) * 2.0 * HSV_S_DELTA,
        (np.random.rand() - 0.5) * 2.0 * HSV_V_DELTA,
    ], dtype=np.float32)
    Iroi = hsv_transform(Iroi, hsv_mod)

    #
    #  Pixel-only sensor stage (blur/exposure/noise/compression): needs no
    #  label plumbing — the output map is built from ptslist below, after this.
    #  uint8 round-trip: ImageCompression needs uint8 input; this is also
    #  albumentations' best-tested dtype path for every op in the stage.
    #
    u8 = (np.clip(Iroi, 0.0, 1.0) * 255.0).astype(np.uint8)
    Iroi = _pixel_stage_fn()(image=u8)["image"].astype(np.float32) / 255.0
    Iroi = np.clip(Iroi, 0.0, 1.0)

    #
    #  Guarded cutout (occlusion robustness, never swallows the plate)
    #
    Iroi = guarded_cutout(Iroi, ptslist, dim)

    #
    #  Finds bounding boxes of all annotated plates
    #
    labelist = []
    for pts in ptslist:
        tl, br = pts.min(1), pts.max(1)
        labelist.append(Label(0, tl, br))

    #
    #  Returns image, and two lists with LP lalbels and quadrilateral points
    #
    return Iroi, labelist, ptslist

