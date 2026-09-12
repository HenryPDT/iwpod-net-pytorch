"""VRAM probing for automatic batch size (yolox-style).

Trigger: `--batch-size -1`. Repeats the densest real sample to powers of two,
running a full forward+backward+optimizer step each, and keeps the largest
batch whose peak reservation stays under `target_fraction` of VRAM — then
bisects for the exact max. OOM during probing is caught per-trial (returns
False, not a crash); total failure falls back to bs=4 with a warning.
CUDA-only: elsewhere warns and returns the configured batch size.
"""
import torch
from loguru import logger


def auto_batch_size(model_fn, loss_fn, sample, device, target_fraction=0.7,
                    amp_enabled=True, max_bs=256):
    """model_fn() -> fresh train-mode model; loss_fn(out, target) -> scalar loss.

    sample: (inputs_1, target_1) at the largest shape training will use.
    Returns the selected micro-batch size.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        logger.warning("Auto batch size needs CUDA; keeping configured batch size")
        return None
    try:
        selected = _probe(model_fn, loss_fn, sample, device, target_fraction,
                          amp_enabled, max_bs)
        logger.info(f"Auto batch selected micro-batch size: {selected}")
        return selected
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.warning(f"Auto batch probing OOMed, defaulting to batch_size=4: {e}")
            return 4
        raise


def _probe(model_fn, loss_fn, sample, device, target_fraction, amp_enabled, max_bs):
    logger.info("Probing VRAM for the largest fitting batch size...")
    inputs_1, target_1 = sample
    inputs_1 = inputs_1.to(device)
    target_1 = target_1.to(device)

    model = model_fn().to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    total = torch.cuda.get_device_properties(device).total_memory
    target_mem = int(total * target_fraction)

    def _try_batch(bs, model, opt, scaler):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            batch_in = inputs_1.repeat(bs, 1, 1, 1)
            batch_tg = target_1.repeat(bs, 1, 1, 1)
            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    loss = loss_fn(model(batch_in), batch_tg)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = loss_fn(model(batch_in), batch_tg)
                loss.backward()
                opt.step()
            peak = torch.cuda.max_memory_reserved(device)
            model.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            return peak <= target_mem
        except RuntimeError as e:
            model.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if "out of memory" in str(e).lower():
                return False
            raise

    best, fail, bs = 1, None, 2
    while bs <= max_bs:
        if _try_batch(bs, model, opt, scaler):
            best = bs
            bs *= 2
        else:
            fail = bs
            break
    if fail is not None and fail - best > 1:
        lo, hi = best + 1, fail - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if _try_batch(mid, model, opt, scaler):
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
    torch.cuda.empty_cache()
    del model, opt
    logger.info(f"Probed batch size {best} "
                f"(target {target_fraction:.0%} of {total / 1024**3:.1f} GiB VRAM)")
    return best
