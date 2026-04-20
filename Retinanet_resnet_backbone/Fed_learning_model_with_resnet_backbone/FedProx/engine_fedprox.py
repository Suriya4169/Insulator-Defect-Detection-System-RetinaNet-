"""
engine_fedprox.py
=================
Drop-in replacement for engine.py that adds FedProx proximal-term support.

Key change
----------
train_one_epoch_fedprox() adds the proximal regularisation term

    L_prox = (mu / 2) * || w_local - w_global ||^2

to every batch loss before back-propagation.  When mu == 0 the function is
mathematically identical to the original train_one_epoch(), guaranteeing
backward compatibility with vanilla FedAvg.

All other helpers (evaluate, print_metrics, compute_iou, calculate_ap)
are re-exported unchanged from the original engine.py so the rest of the
code-base can simply swap the import.
"""

import math
import sys
import copy
import torch
import torch.nn as nn
import torchvision
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
import subprocess
import time

# ── re-export unchanged helpers from the sibling engine ──────────────────────
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'Fed_learning_Code'))
from engine import (compute_iou, calculate_ap, evaluate,print_metrics, get_gpu_temperature)         


# ─────────────────────────────────────────────────────────────────────────────
# FedProx-aware local training step
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch_fedprox(model,optimizer,data_loader,device,epoch,global_params,
                            mu: float = 0.0,scaler=None,scheduler=None,grad_clip=None,
                            extra_info=None,):
    """
    Train for one local epoch with optional FedProx proximal regularisation.
    Parameters
    ----------
    model         : local client model (already loaded with global weights)
    optimizer     : torch.optim.Optimizer
    data_loader   : DataLoader for this client's data shard
    device        : torch.device
    epoch         : int — current local epoch index (for progress-bar label)
    global_params : list of torch.Tensor — the GLOBAL model parameters frozen
                    at the start of this communication round.
                    Obtain with:
                        global_params = [p.detach().clone()
                                         for p in global_model.parameters()]
    mu            : float — FedProx proximal coefficient.
                    mu = 0   → vanilla FedAvg (no regularisation).
                    mu > 0   → proximal term pushes local weights towards
                               the global weights on every gradient step.
                    Recommended starting value: 0.01  (tune in {0.001, 0.01, 0.1})
    scaler        : torch.cuda.amp.GradScaler (optional, for mixed precision)
    scheduler     : LR scheduler (step-level, e.g. OneCycleLR)
    grad_clip     : float or None — max gradient norm for clipping
    extra_info    : dict — extra items for the tqdm progress bar

    Returns
    -------
    dict with keys: total_loss, cls_loss, box_loss, prox_loss
    """
    model.train()

    total_loss     = 0.0
    loss_cls_sum   = 0.0
    loss_box_sum   = 0.0
    prox_loss_sum  = 0.0

    torch.backends.cudnn.benchmark = False

    pbar = tqdm(data_loader, desc=f'[FedProx μ={mu}] Epoch {epoch}',file=sys.stdout, leave=False)

    for i, (images, targets) in enumerate(pbar):
        # ── update progress-bar postfix every 10 steps ────────────────────────
        if i % 10 == 0:
            postfix = {'loss': f'{total_loss / max(i, 1):.4f}',
                       'prox': f'{prox_loss_sum / max(i, 1):.5f}'}
            if extra_info:
                postfix.update(extra_info)
            pbar.set_postfix(postfix)

        images  = [img.to(device)   for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # ── forward + detection loss (same as original engine) ────────────────
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss_dict  = model(images, targets)
                task_loss  = sum(loss for loss in loss_dict.values())

                # ── FedProx proximal term ─────────────────────────────────────
                prox_loss = _compute_proximal_loss(model, global_params, mu,device)
                losses = task_loss + prox_loss

            scaler.scale(losses).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict  = model(images, targets)
            task_loss  = sum(loss for loss in loss_dict.values())

            # ── FedProx proximal term ─────────────────────────────────────────
            prox_loss = _compute_proximal_loss(model, global_params, mu,device)
            losses = task_loss + prox_loss
            losses.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # ── accumulate metrics ────────────────────────────────────────────────
        total_loss    += losses.item()
        loss_cls_sum  += loss_dict.get('classification',
                                        torch.tensor(0.0)).item()
        loss_box_sum  += loss_dict.get('bbox_regression',
                                        torch.tensor(0.0)).item()
        prox_loss_sum += prox_loss.item() if isinstance(prox_loss, torch.Tensor) \
                         else float(prox_loss)

    n = max(len(data_loader), 1)
    print(f"    Epoch {epoch} | TotalLoss: {total_loss/n:.4f}  "
          f"ProxLoss: {prox_loss_sum/n:.5f}", flush=True)

    return {
        'total_loss': total_loss    / n,
        'cls_loss':   loss_cls_sum  / n,
        'box_loss':   loss_box_sum  / n,
        'prox_loss':  prox_loss_sum / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Proximal loss helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_proximal_loss(model,global_params,mu: float,device,) -> torch.Tensor:
    """
    Compute  (mu / 2) * SUM_k || w_k - w_k^t ||^2

    where w_k are the CURRENT local parameters and w_k^t are the GLOBAL
    parameters fixed at the start of the round.

    When mu == 0 this returns the scalar tensor 0.0 (no computational
    overhead beyond this branch check).
    """
    if mu == 0.0:
        return torch.tensor(0.0, device=device, requires_grad=False)

    prox = torch.tensor(0.0, device=device)
    for local_p, global_p in zip(model.parameters(), global_params):
        if local_p.requires_grad:
            prox = prox + torch.sum((local_p - global_p.to(device)) ** 2)

    return (mu / 2.0) * prox
