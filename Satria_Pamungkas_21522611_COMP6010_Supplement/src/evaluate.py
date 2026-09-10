"""Quantitative evaluation for COMP6010 Part C option C2.

Three measures, all applied to the FULL output set of every configuration (never to
hand-picked successes):

1. `recognisability` -- class-conditional accuracy of an independent CNN classifier trained
   only on the HELD-OUT split of real drawings. It answers: "would a third party assign this
   generated image to the class we asked for?"
2. `nn_distance` -- L2 distance from each generated image to its nearest TRAINING drawing, used
   as a memorisation probe (compared against a real-vs-train baseline).
3. `diversity` -- mean pairwise L2 distance within a class, to detect mode collapse.

Limitations are documented in the notebook: the classifier is itself imperfect, an L2 nearest
neighbour is a weak perceptual metric, and none of these is a substitute for human judgement.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# ------------------------------------------------------------------ independent classifier
class DrawingCNN(nn.Module):
    """Small CNN used ONLY as an evaluation instrument, never in the generative pipeline."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.body(x))


def train_classifier(x_train, y_train, x_val, y_val, n_classes=3, epochs=25,
                     batch_size=128, lr=1e-3, seed=777, device=None, verbose=True):
    """Train the evaluation classifier on REAL drawings from the held-out split.

    Using the held-out split (not the generator's training split) keeps the evaluator
    independent of the data the generative model saw.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = DrawingCNN(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(x_train).to(device); yt = torch.from_numpy(y_train).to(device)
    xv = torch.from_numpy(x_val).to(device);   yv = torch.from_numpy(y_val).to(device)
    g = torch.Generator(device=device); g.manual_seed(seed)
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt), device=device, generator=g)
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(xv).argmax(1) == yv).float().mean().item()
        hist.append({"epoch": ep, "val_accuracy": acc})
        if verbose and (ep % 5 == 0 or ep == 1):
            print(f"  classifier epoch {ep:2d}: held-out val accuracy = {acc:.4f}")
    return model, hist


@torch.no_grad()
def recognisability(clf, images: torch.Tensor, labels: torch.Tensor, device="cuda"):
    """Fraction of generated images the independent classifier assigns to the REQUESTED class.

    Returns dict with overall accuracy, per-class accuracy and mean confidence.
    """
    clf.eval()
    images = images.to(device).clamp(-1, 1); labels = labels.to(device)
    logits = clf(images)
    pred = logits.argmax(1)
    prob = F.softmax(logits, dim=1)
    correct = (pred == labels)
    per_class = {}
    for c in labels.unique().tolist():
        m = labels == c
        per_class[int(c)] = {"n": int(m.sum()),
                             "accuracy": float(correct[m].float().mean()),
                             "mean_confidence_in_requested_class":
                                 float(prob[m, c].mean())}
    return {"overall_accuracy": float(correct.float().mean()),
            "mean_confidence_in_requested_class":
                float(prob[torch.arange(len(labels), device=device), labels].mean()),
            "per_class": per_class,
            "predictions": pred.cpu().numpy().tolist()}


def nearest_neighbour_distance(gen: np.ndarray, train: np.ndarray, batch=64):
    """Min L2 distance from each generated image to any training image.

    Returns (min_distances (N,), argmin indices (N,)). Used as a memorisation probe:
    resemblance alone is not proof of copying, so these are compared against the
    real-held-out-vs-train baseline computed the same way.
    """
    g = torch.from_numpy(gen.reshape(len(gen), -1)).float()
    t = torch.from_numpy(train.reshape(len(train), -1)).float()
    if torch.cuda.is_available():
        g, t = g.cuda(), t.cuda()
    dists, idxs = [], []
    for i in range(0, len(g), batch):
        d = torch.cdist(g[i:i + batch], t)          # (b, M)
        m, a = d.min(dim=1)
        dists.append(m.cpu()); idxs.append(a.cpu())
    return torch.cat(dists).numpy(), torch.cat(idxs).numpy()


def diversity(images: np.ndarray, max_n=200, seed=0):
    """Mean pairwise L2 distance within a set of images (mode-collapse probe)."""
    x = images.reshape(len(images), -1)
    if len(x) > max_n:
        x = x[np.random.default_rng(seed).choice(len(x), max_n, replace=False)]
    t = torch.from_numpy(x).float()
    if torch.cuda.is_available():
        t = t.cuda()
    d = torch.cdist(t, t)
    iu = torch.triu_indices(len(t), len(t), offset=1)
    return float(d[iu[0], iu[1]].mean())


def ink_statistics(images: np.ndarray, threshold=0.0):
    """Ink coverage stats, comparable between real and generated sets (a crude realism check)."""
    ink = (images > threshold).reshape(len(images), -1).mean(axis=1)
    return {"mean_ink_fraction": float(ink.mean()), "std_ink_fraction": float(ink.std()),
            "min": float(ink.min()), "max": float(ink.max())}
