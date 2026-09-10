"""Training loop for the conditional flow-matching model (COMP6010 Part C, option C2)."""
from __future__ import annotations
import os, json, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from flow_model import VelocityUNet, flow_matching_batch, count_parameters


def make_loaders(ds, batch_size=128, augment_hflip=True, seed=0, num_workers=2):
    """Tensor loaders. Augmentation (horizontal flip) is applied to the TRAIN split only,
    on-the-fly, AFTER the original-drawing split -- eval drawings are never augmented."""
    tr = TensorDataset(torch.from_numpy(ds["train_x"]), torch.from_numpy(ds["train_y"]))
    ev = TensorDataset(torch.from_numpy(ds["eval_x"]),  torch.from_numpy(ds["eval_y"]))
    g = torch.Generator(); g.manual_seed(seed)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True,
                       generator=g, num_workers=num_workers, pin_memory=True),
            DataLoader(ev, batch_size=256, shuffle=False, num_workers=num_workers))


@torch.no_grad()
def eval_loss(model, loader, device, n_t_samples=8, seed=1234):
    """Held-out flow-matching loss, averaged over a FIXED grid of t values for low variance.

    Uses evenly spaced t rather than random t so the number is comparable across checkpoints.
    """
    model.eval()
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    tot, n = 0.0, 0
    ts = torch.linspace(0.02, 0.98, n_t_samples, device=device)
    for x1, y in loader:
        x1, y = x1.to(device), y.to(device)
        for t_val in ts:
            x0 = torch.randn(x1.shape, device=device, generator=gen)
            t = t_val.expand(x1.shape[0])
            x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
            pred = model(t, x_t, y)
            tot += torch.mean((pred - (x1 - x0)) ** 2).item() * x1.shape[0]
            n += x1.shape[0]
    return tot / n


def train(ds, out_dir, epochs=60, batch_size=128, lr=2e-4, base=64, emb_dim=128,
          label_dropout=0.0, augment_hflip=True, seed=20260910,
          checkpoint_epochs=(5, 15, 60), device=None, log_every=25, verbose=True):
    """Train the velocity network. Saves checkpoints and a JSON training log.

    label_dropout > 0 replaces labels with the null token at that rate, which is what an
    unconditional branch for classifier-free guidance would require. Default 0 (pure
    class-conditional) -- guidance is NOT claimed in this project.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    n_classes = len(ds["categories"])
    model = VelocityUNet(n_classes=n_classes, base=base, emb_dim=emb_dim).to(device)
    total_p, train_p = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr_loader, ev_loader = make_loaders(ds, batch_size=batch_size, seed=seed)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr_loader))

    log = {"config": {"epochs": epochs, "batch_size": batch_size, "lr": lr, "base": base,
                      "emb_dim": emb_dim, "label_dropout": label_dropout,
                      "augment_hflip": augment_hflip, "seed": seed,
                      "optimiser": "AdamW(weight_decay=1e-4)",
                      "lr_schedule": "cosine annealing to 0",
                      "n_classes": n_classes, "device": device,
                      "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                      "params_total": total_p, "params_trainable": train_p,
                      "n_train_original": int(len(ds["train_y"])),
                      "n_eval_original": int(len(ds["eval_y"]))},
           "steps": [], "epochs": [], "checkpoints": []}
    if verbose:
        print(f"device={device} | params: {total_p:,} total, {train_p:,} trainable")
        print(f"train batches/epoch = {len(tr_loader)} | epochs = {epochs}")

    gen = torch.Generator(device=device); gen.manual_seed(seed)
    step, t_start = 0, time.time()

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, ep_n, ep_t0 = 0.0, 0, time.time()
        for x1, y in tr_loader:
            x1, y = x1.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if augment_hflip:                                  # augmentation: train split only
                flip = torch.rand(x1.shape[0], device=device, generator=gen) < 0.5
                x1 = torch.where(flip.view(-1, 1, 1, 1), x1.flip(-1), x1)
            if label_dropout > 0:
                drop = torch.rand(y.shape[0], device=device, generator=gen) < label_dropout
                y = torch.where(drop, torch.full_like(y, n_classes), y)

            x_t, t, target = flow_matching_batch(x1, generator=gen)
            pred = model(t, x_t, y)
            loss = torch.mean((pred - target) ** 2)            # the CFM regression loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

            ep_loss += loss.item() * x1.shape[0]; ep_n += x1.shape[0]; step += 1
            if step % log_every == 0:
                log["steps"].append({"step": step, "epoch": ep, "train_loss": loss.item(),
                                     "lr": sched.get_last_lr()[0]})

        tr_l = ep_loss / ep_n
        ev_l = eval_loss(model, ev_loader, device)
        log["epochs"].append({"epoch": ep, "train_loss": tr_l, "eval_loss": ev_l,
                              "epoch_time_s": time.time() - ep_t0,
                              "cumulative_time_s": time.time() - t_start})
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"epoch {ep:3d}/{epochs} | train {tr_l:.4f} | eval {ev_l:.4f} | "
                  f"{time.time()-ep_t0:.1f}s | total {(time.time()-t_start)/60:.1f}min")

        if ep in checkpoint_epochs:
            ck = os.path.join(out_dir, f"ckpt_epoch{ep:03d}.pt")
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "config": log["config"], "train_loss": tr_l, "eval_loss": ev_l,
                        "categories": ds["categories"]}, ck)
            log["checkpoints"].append({"epoch": ep, "path": os.path.basename(ck),
                                       "train_loss": tr_l, "eval_loss": ev_l,
                                       "cumulative_time_s": time.time() - t_start})
            if verbose:
                print(f"   saved {os.path.basename(ck)}")

    log["total_training_time_s"] = time.time() - t_start
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    if verbose:
        print(f"\ntraining done in {log['total_training_time_s']/60:.1f} min")
    return model, log


def load_checkpoint(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = VelocityUNet(n_classes=cfg["n_classes"], base=cfg["base"],
                         emb_dim=cfg["emb_dim"]).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, ck
