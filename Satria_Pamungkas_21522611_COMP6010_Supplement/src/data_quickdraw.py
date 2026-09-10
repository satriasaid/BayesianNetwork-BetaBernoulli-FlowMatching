"""Quick, Draw! data preparation for COMP6010 Part C (option C2, flow matching).

Data source: Google Quick, Draw! `numpy_bitmap` files (28x28 uint8 grayscale, centred).
    https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<category>.npy
Attribution: "Quick, Draw! Dataset" by Google Creative Lab, CC BY 4.0.

The split is performed on ORIGINAL drawings, before any augmentation, and the numbers
reported are counts of original drawings.
"""
from __future__ import annotations
import os, json, hashlib
import numpy as np

CATEGORIES = ["cat", "dog", "fish"]          # 3 categories, class-conditional model
N_PER_CATEGORY = 4000                        # original drawings kept per category
TRAIN_FRACTION = 0.8
SPLIT_SEED = 20260910                        # fixed, documented split seed
IMG_SIZE = 28                                # native Quick, Draw! resolution

# Disclosed filter, calibrated on the observed ink-fraction distribution of these three
# categories (see notebook 03, "data inspection"): the bulk lies between 0.16 and 0.43.
# These bounds remove only clear outliers -- near-blank drawings and dense scribbles --
# and reject well under 1% of candidates. The thresholds are deliberately loose so that the
# subset stays a representative random sample rather than a hand-picked "attractive" one.
INK_MIN_FRACTION = 0.05                      # reject near-blank drawings
INK_MAX_FRACTION = 0.55                      # reject near-solid scribbles


def data_dir(start: str | None = None) -> str:
    """Locate <supplement>/data/quickdraw from notebooks/ or the supplement root."""
    here = os.path.abspath(start or os.getcwd())
    for cand in (os.path.join(here, "..", "data", "quickdraw"),
                 os.path.join(here, "data", "quickdraw")):
        cand = os.path.abspath(cand)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("data/quickdraw not found; run download_quickdraw.sh first")


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def load_category(name: str, root: str | None = None):
    """Memory-map one category file. Returns (array (M,784) uint8, path)."""
    root = root or data_dir()
    path = os.path.join(root, f"{name}.npy")
    return np.load(path, mmap_mode="r"), path


def ink_fraction(flat_uint8: np.ndarray) -> np.ndarray:
    """Fraction of pixels with any ink, per drawing. Input (M, 784) uint8."""
    return (flat_uint8 > 0).mean(axis=1)


def build_dataset(categories=CATEGORIES, n_per_category=N_PER_CATEGORY,
                  train_fraction=TRAIN_FRACTION, seed=SPLIT_SEED, root=None,
                  verbose=True):
    """Select a fixed random subset per category, filter, then split BEFORE augmentation.

    Returns dict with train/eval images in [-1, 1] float32 of shape (N, 1, 28, 28),
    integer labels, and a provenance/manifest record.
    """
    root = root or data_dir()
    rng = np.random.default_rng(seed)

    tr_x, tr_y, ev_x, ev_y = [], [], [], []
    record = {"source": "Google Quick, Draw! numpy_bitmap (CC BY 4.0, Google Creative Lab)",
              "url_pattern": ("https://storage.googleapis.com/quickdraw_dataset/full/"
                              "numpy_bitmap/<category>.npy"),
              "image_size": IMG_SIZE, "split_seed": seed,
              "train_fraction": train_fraction,
              "filter": {"rule": "keep drawings with ink fraction in range",
                         "ink_min_fraction": INK_MIN_FRACTION,
                         "ink_max_fraction": INK_MAX_FRACTION},
              "categories": []}

    for label, cat in enumerate(categories):
        arr, path = load_category(cat, root)
        total_available = int(arr.shape[0])

        # Draw a fixed random subset of INDICES first (not "the nicest examples"), then
        # filter; oversample so the filter still leaves n_per_category drawings.
        oversample = min(total_available, int(n_per_category * 1.6) + 500)
        cand_idx = np.sort(rng.choice(total_available, size=oversample, replace=False))
        cand = np.asarray(arr[cand_idx])                        # (oversample, 784)

        frac = ink_fraction(cand)
        keep = (frac >= INK_MIN_FRACTION) & (frac <= INK_MAX_FRACTION)
        n_rejected = int((~keep).sum())
        kept_idx = cand_idx[keep][:n_per_category]
        kept = cand[keep][:n_per_category]
        if len(kept) < n_per_category:
            raise RuntimeError(f"{cat}: only {len(kept)} drawings survived filtering")

        # ---- split ORIGINAL drawings before any augmentation ----
        perm = rng.permutation(len(kept))
        n_train = int(round(train_fraction * len(kept)))
        tr_sel, ev_sel = perm[:n_train], perm[n_train:]

        imgs = (kept.reshape(-1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32) / 255.0)
        imgs = imgs * 2.0 - 1.0                                 # scale to [-1, 1]

        tr_x.append(imgs[tr_sel]); tr_y.append(np.full(len(tr_sel), label, np.int64))
        ev_x.append(imgs[ev_sel]); ev_y.append(np.full(len(ev_sel), label, np.int64))

        record["categories"].append({
            "name": cat, "label": label, "file": os.path.basename(path),
            "sha256": file_sha256(path),
            "drawings_available_in_file": total_available,
            "candidates_drawn": int(oversample),
            "rejected_by_ink_filter": n_rejected,
            "original_drawings_used": int(len(kept)),
            "n_train_original": int(len(tr_sel)),
            "n_eval_original": int(len(ev_sel)),
            "train_indices_sha256": hashlib.sha256(
                np.sort(kept_idx[tr_sel]).tobytes()).hexdigest(),
            "eval_indices_sha256": hashlib.sha256(
                np.sort(kept_idx[ev_sel]).tobytes()).hexdigest(),
        })
        if verbose:
            print(f"{cat:5s}: {total_available:,} available -> drew {oversample:,} candidates, "
                  f"rejected {n_rejected} by ink filter -> {len(kept):,} used "
                  f"({len(tr_sel):,} train / {len(ev_sel):,} eval original drawings)")

    out = {
        "train_x": np.concatenate(tr_x), "train_y": np.concatenate(tr_y),
        "eval_x":  np.concatenate(ev_x), "eval_y":  np.concatenate(ev_y),
        "categories": list(categories), "manifest": record,
    }
    record["totals"] = {"n_train_original": int(len(out["train_y"])),
                        "n_eval_original": int(len(out["eval_y"])),
                        "n_categories": len(categories)}
    if verbose:
        print(f"\nTOTAL original drawings: {record['totals']['n_train_original']:,} train, "
              f"{record['totals']['n_eval_original']:,} held-out eval "
              f"(these are ORIGINAL counts, not augmented copies)")
    return out


def save_manifest(ds, path):
    with open(path, "w") as f:
        json.dump(ds["manifest"], f, indent=2)
    return path
