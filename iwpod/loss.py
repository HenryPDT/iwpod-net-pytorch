from typing import NamedTuple

import torch
import torch.nn.functional as F


def logloss(Ptrue, Pred, szs, eps=1e-10):
    b, h, w, ch = szs
    Pred = torch.clamp(Pred, eps, 1.0 - eps)
    Pred = -torch.log(Pred)
    Pred = Pred * Ptrue
    Pred = Pred.view(b, h * w * ch)
    Pred = torch.sum(Pred, 1)
    return Pred


def l1(true, pred, szs):
    b, h, w, ch = szs
    # res = (true - pred).view(b, h * w * ch)
    res = (true - pred).reshape(b, h * w * ch)
    res = torch.abs(res)
    res = torch.sum(res, 1)
    return res


def clas_loss(Ytrue, Ypred):
    wtrue = 0.5
    wfalse = 0.5
    b, h, w = Ytrue.size(0), Ytrue.size(2), Ytrue.size(3)

    obj_probs_true = Ytrue[:, 0, ...]
    obj_probs_pred = Ypred[:, 0, ...]

    non_obj_probs_true = 1 - obj_probs_true
    non_obj_probs_pred = 1 - obj_probs_pred

    res = wtrue * logloss(obj_probs_true, obj_probs_pred, (b, h, w, 1))
    res += wfalse * logloss(non_obj_probs_true, non_obj_probs_pred, (b, h, w, 1))
    return res


def _affine_to_pts(affine_pred, b, h, w, device):
    """Shared affine(6ch)->corner(8ch) mapping. Keeps no-flip clamp on diagonal."""
    affinex = torch.stack([torch.clamp(affine_pred[:, 0, ...], min=0.), affine_pred[:, 1, ...], affine_pred[:, 2, ...]], 1)
    affiney = torch.stack([affine_pred[:, 3, ...], torch.clamp(affine_pred[:, 4, ...], min=0.), affine_pred[:, 5, ...]], 1)

    v = 0.5
    base = torch.tensor([-v, -v, 1., v, -v, 1., v, v, 1., -v, v, 1.], device=device)
    base = base.repeat(b, h, w, 1)
    base = base.permute(0, 3, 1, 2)

    pts = torch.zeros((b, 0, h, w), device=device)

    for i in range(0, 12, 3):
        row = base[:, i:(i + 3), ...]
        ptsx = torch.sum(affinex * row, 1)
        ptsy = torch.sum(affiney * row, 1)

        pts_xy = torch.stack([ptsx, ptsy], 1)
        pts = torch.cat([pts, pts_xy], 1)
    return pts


def loc_loss(Ytrue, Ypred):
    b, h, w = Ytrue.size(0), Ytrue.size(2), Ytrue.size(3)

    # device = 'cpu'

    obj_probs_true = Ytrue[:, 0, ...]
    affine_pred = Ypred[:, 1:, ...]
    pts_true = Ytrue[:, 1:, ...]

    pts = _affine_to_pts(affine_pred, b, h, w, Ypred.device)

    flags = obj_probs_true.view(b, 1, h, w)
    res = 1.0 * l1(pts_true * flags, pts * flags, (b, h, w, 4 * 2))
    return res


def iwpodnet_loss(Ytrue, Ypred):
    wclas = 0.5
    wloc = 0.5
    return wloc * loc_loss(Ytrue, Ypred) + wclas * clas_loss(Ytrue, Ypred)


# ----------------------------- v2 losses -----------------------------

def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0, pos_weight=None):
    """Sigmoid focal loss, mean reduction. Handles 1:150 foreground imbalance."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    # down-weight easy negatives but keep mean stable
    return focal.mean()


def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum()
    denom = probs.sum() + targets.sum() + eps
    return 1.0 - (2 * inter + eps) / denom


def wing_loss(pred, target, w=10.0, eps=2.0, mask=None):
    """Robust corner regression (more weight near small errors than L1)."""
    diff = (pred - target).abs()
    c = w - w * torch.log(torch.tensor(1.0 + w / eps))
    loss = torch.where(diff < w, w * torch.log(1 + diff / eps), diff - c)
    if mask is not None:
        # mask: [B,1,H,W] broadcast over 8 corner channels
        loss = loss * mask.expand_as(loss)
        denom = mask.sum().clamp_min(1.0) * loss.shape[1]
        return loss.sum() / denom
    return loss.mean()


class LossResult(NamedTuple):
    total: torch.Tensor   # per-sample vector [B] (mean it downstream)
    cls:   torch.Tensor   # scalar
    dice:  torch.Tensor   # scalar
    loc:   torch.Tensor   # scalar


def iwpodnet_loss_v2(Ytrue, Ypred_logits, w_cls=1.0, w_dice=0.5, w_loc=1.0,
                     focal_alpha=0.25, focal_gamma=2.0, ohem_neg_ratio=3.0,
                     dice_ohem=False):
    """Ytrue: [B,9,H,W] with ch0={0,1} mask + ch1..8 corner targets (side-normalized).
    Ypred_logits: [B,7,H,W] RAW logits in ch0 + 6 affine params.
    OHEM: keep all positives + hardest negatives at `ohem_neg_ratio` ratio
    (index-based top-k, so score ties can't over-keep).
    dice_ohem: compute Dice on the kept OHEM subset instead of globally
    (ablation for 1:150 imbalance; default False preserves current behavior).
    Returns a LossResult namedtuple with (total, cls, dice, loc) tensors."""
    logits = Ypred_logits[:, 0:1, ...]
    targets = Ytrue[:, 0:1, ...]
    b = Ytrue.size(0)
    with torch.no_grad():
        pos = (targets > 0.5)
        n_pos = pos.sum().clamp_min(1).item()
        # hardest negatives (index-based: exact count, ties can't over-keep)
        bce_nored = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        neg = ~pos
        n_neg_keep = min(int(n_pos * ohem_neg_ratio), int(neg.sum().item()))
        keep = pos.clone()
        if n_neg_keep > 0:
            neg_scores = torch.where(neg, bce_nored,
                                     torch.full_like(bce_nored, float("-inf")))
            topk_idx = torch.topk(neg_scores.view(-1), n_neg_keep, sorted=False).indices
            keep_flat = keep.view(-1)
            keep_flat[topk_idx] = True
            keep = keep_flat.view_as(pos)
    logits_k = logits[keep]
    targets_k = targets[keep]
    cls = focal_bce_with_logits(logits_k, targets_k, alpha=focal_alpha, gamma=focal_gamma)
    if dice_ohem:
        dice = dice_loss_from_logits(logits_k, targets_k)
    else:
        dice = dice_loss_from_logits(logits, targets)

    affine_pred = Ypred_logits[:, 1:, ...]
    pts_true = Ytrue[:, 1:, ...]
    h, w = Ytrue.size(2), Ytrue.size(3)
    pts_pred = _affine_to_pts(affine_pred, b, h, w, Ypred_logits.device)
    mask = (Ytrue[:, 0:1, ...] > 0.5).float()
    loc = wing_loss(pts_pred, pts_true, mask=mask)
    total = w_cls * cls + w_dice * dice + w_loc * loc
    # return per-sample vector to match legacy API (mean downstream)
    return LossResult(total=total.expand(b), cls=cls, dice=dice, loc=loc)

