"""
engine_fedprox_mb.py
====================
FedProx-aware training engine for the MobileNet backbone.

This file is the MobileNet equivalent of engine_fedprox.py used by the
ResNet Phase 1 experiments. The proximal regularization logic is identical —
this ensures a fair, apple-to-apple comparison between the two backbones.

Key Difference from standard engine.py:
    The standard engine trains freely. This engine computes the proximal
    regularization term: L_prox = (mu/2) * ||w_local - w_global||^2
    and adds it to the detection loss before backprop.
"""

import sys
import copy
import torch
from tqdm.auto import tqdm


def train_one_epoch_fedprox(model,global_weights,optimizer,data_loader,
    device,epoch,mu=0.0,scaler=None,scheduler=None,grad_clip=1.0,
    desc_prefix="[MobileNet FedProx]"):
    
    model.train()
    torch.backends.cudnn.benchmark = True  # Significant speedup on Windows

    # ── OPTIMIZATION: Move global weights to device ONCE per epoch ──────────
    # This prevents thousands of slow CPU->GPU transfers inside the batch loop
    global_weights_gpu = {}
    if mu > 0.0:
        global_weights_gpu = {k: v.to(device).detach() for k, v in global_weights.items()}

    total_loss = 0.0
    total_prox = 0.0
    n_batches = 0

    pbar = tqdm(
        data_loader,
        desc=f"{desc_prefix} Epoch {epoch}",
        file=sys.stdout,
        leave=False
    )

    for images, targets in pbar:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # ── Mixed Precision Training (FAST) ──────────────────────────────────
        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            detection_loss = sum(loss for loss in loss_dict.values())

            # ── Proximal regularisation ─────────────────────────────────────
            prox_loss = torch.tensor(0.0, device=device)
            if mu > 0.0:
                for name, local_param in model.named_parameters():
                    if local_param.requires_grad and name in global_weights_gpu:
                        diff = local_param - global_weights_gpu[name]
                        prox_loss = prox_loss + (mu / 2.0) * diff.norm() ** 2

            total_obj = detection_loss + prox_loss

        # ── Scaled Backward & Optimization ──────────────────────────────────
        if scaler is not None:
            scaler.scale(total_obj).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_obj.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += total_obj.item()
        total_prox += prox_loss.item()
        n_batches  += 1

        pbar.set_postfix({
            'loss': f'{total_obj.item():.4f}',
            'prox': f'{prox_loss.item():.5f}',
        })

    avg_loss = total_loss / max(n_batches, 1)
    avg_prox = total_prox / max(n_batches, 1)
    return avg_loss, avg_prox
