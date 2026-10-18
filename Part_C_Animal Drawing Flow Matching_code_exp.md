# Part C: Animal Drawing Flow Matching — Code Explanation (cell by cell)

This walks through `PartC02_Animal Drawing Flow Matching.ipynb` from top to bottom. It is written for someone who knows basic Python (variables, lists, `for` loops, functions) but is new to NumPy, PyTorch and generative models.

How references work:

- Cells are numbered from the top of the notebook, starting at Cell 1. Only Cell 2 has a notebook cell id (`id 5dd5461e`). Every other cell has none, so each heading also quotes the cell's first comment line (e.g. `# === the network`) so you can find it with Ctrl+F.
- Code being discussed is quoted in a code block right before (or inside) the explanation.
- Markdown (text) cells are summarised briefly; code cells get the detailed treatment.
- "Sample output" blocks show what a line or step actually produces. I generated them by running the notebook's own code with Python 3.10.12, PyTorch 2.3.0 (CPU) and NumPy 1.26.4, using the existing checkpoints and data. No training was re-run. Blocks marked "notebook output" are copied from the notebook's saved results. Tiny examples (2–8 images, or a 4-number "image") say which seed and size they use.

## The big picture in one paragraph

The goal is a program that invents new doodles of a cat, a dog or a fish. It learns from 9,600 real 28×28 Quick, Draw! doodles. The method is flow matching. Picture every image as a point, and imagine sliding in a straight line from a random-noise image to a real drawing. A neural network (a "U-Net") is trained to answer one question: "If I am at this blurry in-between image at time `t`, which direction should I move, and how fast?" Once it has learned that, generating is easy. Start from fresh noise and take small steps in the direction the network says, from `t = 0` (noise) to `t = 1` (drawing). The notebook then measures how recognisable the results are. A separate small classifier scores them. It also checks how varied they are, and whether the model simply copied training drawings. It answers two research questions. Q1: does training longer help? Q2: how few steps can you get away with, and is the "midpoint" stepping rule better than plain "Euler" steps?

---

## Cell 1 (markdown): Title, goal, data

States the goal: train a small model from scratch that turns random noise into cat, dog and fish doodles. The data is Google's Quick, Draw! `numpy_bitmap` files (28×28 grayscale, CC BY 4.0). It uses 4,000 drawings per category, split 80% train and 20% held out. The notebook loads saved checkpoints if they exist, so you can rebuild every figure without retraining.

---

## Cell 2 (code, `id 5dd5461e`): Install the packages

```python
%pip install -r requirements_frozen.txt
```

The same as Part A. `%pip` runs pip inside the notebook's environment and installs the exact package versions listed in the file.

---

## Cell 3 (code): `# === setup and folders`

### Part 1: imports

```python
import os, sys, json, time, math, platform, hashlib, urllib.request
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
```

New compared with Part A:

| Name | What it is used for here |
|---|---|
| `torch` (PyTorch) | Neural networks. Its arrays are called tensors (like NumPy arrays, but they can run on a GPU and track gradients) |
| `nn` | Ready-made network building blocks (layers) |
| `F` | Functions without trainable weights, e.g. `F.silu`, `F.softmax`, `F.cross_entropy` |
| `TensorDataset`, `DataLoader` | Serve training data to the model in shuffled batches |
| `math` | Plain maths functions (`math.log`) |
| `hashlib` | Computes a SHA-256 "fingerprint" of a file |
| `urllib.request` | Downloads files from the internet |


### Part 2: choosing folders

```python
def pick_dir(*candidates):
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c):
            return c
    chosen = os.path.abspath(candidates[0])
    os.makedirs(chosen, exist_ok=True)
    return chosen

RES  = pick_dir(os.path.join(HERE, "results", "partC"), os.path.join(HERE, "..", "results", "partC"))
DATA = pick_dir(os.path.join(HERE, "data", "quickdraw"), os.path.join(HERE, "..", "data", "quickdraw"))
CKPT = pick_dir(os.path.join(HERE, "checkpoints"), os.path.join(HERE, "..", "checkpoints"))
```

- `*candidates` lets the function take any number of folder paths. Inside the function they arrive as a tuple.
- It returns the first candidate folder that already exists. If none exists, it creates the first one (the folder beside the notebook).
- `RES` is where results go, `DATA` where the drawings are, `CKPT` where trained models are saved. `GRID` (a subfolder of `RES`) holds the sample-image grids.

Preferring an existing folder matters. It is how the notebook finds the checkpoints already on disk and skips the 8-minute training.

### Part 3: device and reproducibility

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- `DEVICE` is where the maths runs. `"cuda"` means an NVIDIA graphics card (GPU), which is much faster for neural networks; `"cpu"` is the normal processor. Later, `.to(DEVICE)` moves tensors and models there.
- The `cudnn` lines make the GPU produce identical results every run. Without them, two runs with the same seed could differ by about 0.0007 (Cell 22 explains this).

### Part 4: fixed settings

```python
CATEGORIES   = ["cat", "dog", "fish"]
N_PER_CAT    = 4000
TRAIN_FRAC   = 0.8
IMG          = 28
GLOBAL_SEED  = 20260910
INK_MIN, INK_MAX = 0.05, 0.55
SEED_LIST = [101, 202, 303, ..., 3232]       # 32 seeds
```

- The class labels are positions in `CATEGORIES`: cat = 0, dog = 1, fish = 2.
- `SEED_LIST` holds 32 fixed seeds. Each seed makes one drawing per class, giving 32 × 3 = 96 drawings per configuration. The same seeds are used in every comparison, so differences come from the setting being changed, not from luck.
- `INK_MIN, INK_MAX = 0.05, 0.55` assigns two variables at once (tuple unpacking).

Sample output (notebook output, this run was on CPU):

```text
device   : cpu
torch    : 2.3.0+cu121 | numpy 1.26.4 | python 3.10.12
...
32 generation seeds -> 96 outputs per configuration; margin of error at that size is +/-0.051 (worst case)
```

Where 0.051 comes from: `np.sqrt(0.25 / 96)` = 0.051. If a score is a percentage of n yes/no outcomes, its standard error is `sqrt(p(1−p)/n)`. That is largest when p = 0.5, giving `sqrt(0.25/n)`. Strictly, this is one standard error. A 95% "margin of error" would be 1.96 times that, about ±0.10. See the issues list at the end.


---

## Cell 4 (markdown): `## 1. Method and research questions`

This explains the method with formulas, a training/generating diagram, a table linking Parts A, B and C, and the two research questions with predictions written before running. The key formulas are:

- `x_t = (1−t)·x0 + t·x1`: the point at time `t` on the straight line from noise `x0` to drawing `x1`.
- `u_t = x1 − x0`: the speed (velocity) along that line. It is the same at every `t`.
- Loss = average of `(network's guess − (x1 − x0))²`.
- Generating: start at noise and repeatedly do `x ← x + Δt · v(t, x, y)`.

### Flow matching, simply

Think of a sat-nav. To train it, you show it thousands of trips. Each trip runs in a straight line from a random starting point (noise) to a real destination (a drawing). At random moments along each trip you ask, "From here, which way and how fast?" The answer is always "straight towards the destination, at a constant speed". To use it, you drop it at a brand-new random start, and it keeps telling you which way to go until you arrive at something that looks like a real drawing.

A one-pixel example. Pretend an image is a single number. Noise gives `x0 = −0.5` and the real drawing has `x1 = 0.8`.

```text
t      x_t = (1−t)·(−0.5) + t·0.8
0.00   −0.500     <- pure noise
0.25   −0.175
0.50    0.150
0.75    0.475
1.00    0.800     <- the drawing
speed  x1 − x0 = 0.8 − (−0.5) = 1.3   (the same at every t)
```

If the network learns "speed = 1.3" perfectly, four Euler steps of size 0.25 get you exactly to the drawing:

```text
start  −0.5
step 1 −0.5   + 0.25 × 1.3 = −0.175
step 2 −0.175 + 0.25 × 1.3 =  0.15
step 3  0.15  + 0.25 × 1.3 =  0.475
step 4  0.475 + 0.25 × 1.3 =  0.8
```

(These numbers come from running exactly that arithmetic.) A real image is the same thing done for all 784 pixels at once. In a real image, the network does not know which drawing you are heading to. It learns the average direction from that point, and the markdown explains that this is enough.

---

## Cell 5 (markdown): `## 2. Data: download, inspect, filter and split`

Lists the preprocessing:

1. Pick a fixed random subset.
2. Apply a loose "ink" filter that drops blank or solid scribbles.
3. Rescale pixels to [−1, 1].
4. Split 80/20 before any augmentation.

The only augmentation is a random left-right flip, applied to the training split only.


---

## Cell 6 (code): `# === download the dataset`

```python
def download_category(name, folder=DATA):
    path = os.path.join(folder, f"{name}.npy")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    url = f"{BASE_URL}/{name}.npy"
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path
```

- If the file is already there and bigger than 1 MB, it is used as-is. The size check guards against a half-finished download.
- Otherwise it downloads to `cat.npy.part` first, then renames it to `cat.npy`. If the download breaks halfway, you never end up with a broken `cat.npy` that looks finished.

```python
def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()
```

- This computes a fingerprint of the file. If even one byte differs, the fingerprint is completely different, so anyone can check they have the same data.
- `1 << 20` is "1 shifted left 20 bits" = 2²⁰ = 1,048,576 bytes (1 MB). The file is read 1 MB at a time so a 100 MB file never has to be in memory all at once.
- `"rb"` means "read, binary".
- `while (b := f.read(chunk)):` uses the "walrus" operator `:=`, which assigns and tests in one go. It reads a chunk into `b` and keeps looping while `b` is not empty.

```python
PATHS = {c: download_category(c) for c in CATEGORIES}
```

A dictionary comprehension: `{"cat": ".../cat.npy", "dog": ..., "fish": ...}`.

Sample output (notebook output):

```text
  cat.npy already present (97 MB)
  dog.npy already present (119 MB)
  fish.npy already present (105 MB)

SHA-256 of each source file (recorded in the manifest for verification):
  cat   21a281839d3f2eef601d57d2338a4eafdf24649f8d0a0e42d3ec3e595911463e
  ...
```

### What is inside a `.npy` file

```text
>>> arr = np.load(PATHS["cat"], mmap_mode="r")
>>> type(arr), arr.shape, arr.dtype
(<class 'numpy.memmap'>, (123202, 784), dtype('uint8'))
>>> row = np.asarray(arr[0]); row.min(), row.max()
(0, 255)
```

- There are 123,202 cat drawings. Each is a row of 784 numbers (28 × 28 pixels laid out flat), from 0 (white) to 255 (full ink). `uint8` means whole numbers from 0 to 255.
- `mmap_mode="r"` (memory-map, read-only) does not load the 97 MB file into memory. It reads only the rows you actually index. That is why picking a few thousand rows later is fast.


---

## Cell 7 (code): `# === inspect ink distribution`

```python
for c, p in PATHS.items():
    arr = np.load(p, mmap_mode="r")
    idx = np.sort(np.random.default_rng(1).choice(arr.shape[0], 20_000, replace=False))
    frac = (np.asarray(arr[idx]) > 0).mean(axis=1)
    rows.append({"category": c, "ink p1": np.percentile(frac, 1), "ink median": np.median(frac), ...})
```

The ink filter in Cell 8 needs cut-offs, so this cell first measures what is normal.

- `rng.choice(N, 20_000, replace=False)` picks 20,000 different row numbers at random. `np.sort` puts them in order, which makes reading from the memory-mapped file faster.
- `arr[idx] > 0` gives True for every pixel with any ink. `.mean(axis=1)` averages along each row (each drawing), so `frac` holds one number per drawing: the fraction of its 784 pixels that have ink.
- `np.percentile(frac, 1)` is the value that 1% of drawings fall below. The median is the 50% point.

Sample output (notebook output):

```text
category  drawings in file  ink p1  ink median  ink p99  below 0.05  above 0.55  sampled
     cat            123202  0.1798      0.3342   0.4745           0           5    20000
     dog            152159  0.1684      0.2934   0.4796           0          16    20000
    fish            134150  0.1620      0.2449   0.3712           3           3    20000
```

How to read it: a typical cat drawing has ink on about a third of its pixels. Only 5 of 20,000 cats exceed 0.55, so the filter `[0.05, 0.55]` removes only extreme outliers.

---

## Cell 8 (code): `# === build the dataset`

### The key steps inside `build_dataset`

```python
rng = np.random.default_rng(seed)
for label, cat in enumerate(categories):
    arr = np.load(PATHS[cat], mmap_mode="r")
    oversample = min(total, int(n_per_cat * 1.6) + 500)
    cand_idx = np.sort(rng.choice(total, size=oversample, replace=False))
    cand = np.asarray(arr[cand_idx])
```

- `enumerate(categories)` gives `(0, "cat"), (1, "dog"), (2, "fish")`, so each category gets its number label.
- It draws more candidates than needed, `int(4000 × 1.6) + 500` = 6,900, so that 4,000 remain after filtering.

```python
    frac = (cand > 0).mean(axis=1)
    keep = (frac >= INK_MIN) & (frac <= INK_MAX)
    kept_idx, kept = cand_idx[keep][:n_per_cat], cand[keep][:n_per_cat]
```

This is the same boolean-mask idea as Part A, Cell 7. `keep` is True for drawings that pass the ink filter. `[keep]` keeps those rows, and `[:n_per_cat]` takes the first 4,000.

```python
    perm = rng.permutation(len(kept))
    n_train = int(round(train_frac * len(kept)))
    tr_sel, ev_sel = perm[:n_train], perm[n_train:]
```

`rng.permutation(n)` shuffles the numbers 0 to n−1. The first 80% of the shuffled list are the training drawings and the rest are held out.

Sample output (tiny illustration, `default_rng(0)`, 6 items, 5 for training):

```text
>>> perm = np.random.default_rng(0).permutation(6)
>>> perm
[3 2 5 4 0 1]
>>> perm[:5], perm[5:]
([3 2 5 4 0], [1])
```


### Reshaping and rescaling

```python
    imgs = kept.reshape(-1, 1, IMG, IMG).astype(np.float32) / 255.0
    imgs = imgs * 2.0 - 1.0                                  # rescale to [-1, 1]
```

- `reshape(-1, 1, 28, 28)` turns each flat 784-number row into a 1 × 28 × 28 image. The `-1` means "work out this size yourself" (here 4,000). The `1` is the number of colour channels: grayscale has 1, colour photos have 3. PyTorch image layers expect the shape `(batch, channels, height, width)`.
- `/ 255.0` maps 0…255 to 0…1. Then `* 2 − 1` maps that to −1…1, the same range as the noise, which is roughly centred on 0.

```text
>>> np.arange(784).reshape(-1, 1, 28, 28).shape
(1, 1, 28, 28)
>>> np.arange(2 * 784).reshape(-1, 1, 28, 28).shape
(2, 1, 28, 28)
>>> 0 / 255 * 2 - 1,  128 / 255 * 2 - 1,  255 / 255 * 2 - 1
(-1.0, 0.0039, 1.0)        # white paper -> -1, mid grey -> about 0, full ink -> +1
```

Sample output, row 14 (the middle row) of the first training cat after rescaling. The −1s are blank paper; the values near +1 are pen strokes crossing that row:

```text
[-1.   -1.   -1.   -1.   -1.   -1.   -0.57  0.87  0.89 -0.08 -0.86 -1.
 -1.   -1.   -1.   -1.   -1.   -1.   -0.    0.98  1.    1.    1.    0.48
 -0.85 -1.   -1.   -1.  ]
```

### Labels, joining, and the result

```python
    tr_y.append(np.full(len(tr_sel), label, np.int64))
...
ds = {"train_x": np.concatenate(tr_x), "train_y": np.concatenate(tr_y), ...}
```

- `np.full(3200, 0, np.int64)` is 3,200 copies of the label (0 for cat). `int64` is a 64-bit whole number, the type PyTorch wants for class labels.
- `np.concatenate` glues the three per-category arrays end to end, cats first, then dogs, then fish.

Sample output (rerun of the cell):

```text
train_x (9600, 1, 28, 28) float32
train_y (9600,) int64
eval_x (2400, 1, 28, 28) float32
eval_y (2400,) int64
>>> DS["train_y"][3195:3205]      # around the cat/dog boundary
[0 0 0 0 0 1 1 1 1 1]
>>> np.bincount(DS["train_y"])     # how many of each label
[3200 3200 3200]
```

`train_y` is sorted by class: rows 0–3199 are cats, 3200–6399 dogs, 6400–9599 fish. This matters in Cell 13.

The function also builds a `record` (a "manifest"): file fingerprints, counts, how many drawings the filter rejected, and fingerprints of the chosen indices. It is saved as `C_data_manifest.json`. The `hashlib.sha256(np.sort(...).tobytes()).hexdigest()` lines fingerprint the list of chosen row numbers, so someone else can verify they used exactly the same split.

Notebook output:

```text
cat  : 123,202 available -> drew 6,900 candidates, rejected 3 by ink filter -> 4,000 used (3,200 train / 800 eval original drawings)
...
TOTAL original drawings: 9,600 train, 2,400 held out (ORIGINAL counts, not augmented copies)
```


---

## Cell 9 (code): `# === look at the real data first`

```python
rng_show = np.random.default_rng(7)
fig, axes = plt.subplots(3, 12, figsize=(12, 3.2))
for r, c in enumerate(CATS):
    pool = DS["train_x"][DS["train_y"] == r]
    for k, j in enumerate(rng_show.choice(len(pool), 12, replace=False)):
        axes[r, k].imshow(pool[j, 0], cmap="gray_r", vmin=-1, vmax=1); axes[r, k].axis("off")
```

- This draws a grid of 3 rows (cat, dog, fish) × 12 random real drawings, so you see what the model is learning from.
- `DS["train_x"][DS["train_y"] == r]` is a boolean mask that keeps only the drawings of class `r`.
- `pool[j, 0]` is drawing `j`, channel 0, which is a 28 × 28 array.
- `imshow(..., cmap="gray_r", vmin=-1, vmax=1)` displays it. `gray_r` is "reversed grey": −1 shows as white and +1 as black, like pen on paper.

```python
for i, c in enumerate(CATS):
    ink = (DS["train_x"][DS["train_y"] == i] > 0).reshape(-1, IMG*IMG).mean(axis=1)
    print(f"  {c:5s} mean {ink.mean():.4f}  sd {ink.std():.4f}")
```

This computes the ink fraction per class in the training split, used later as a baseline for generated drawings. `{c:5s}` pads the name to 5 characters so the columns line up.

Sample output (notebook output, reproduced exactly by my rerun):

```text
  cat   mean 0.1984  sd 0.0389
  dog   mean 0.1808  sd 0.0420
  fish  mean 0.1402  sd 0.0258
```

Why is this 0.198 for cats when Cell 7 said the median was 0.334? The two cells measure ink differently:

- Cell 7 counts a pixel as ink if its raw value is above 0 (any trace of grey).
- Cell 9 counts a pixel as ink if its rescaled value is above 0, i.e. raw above 127.5 (darker than mid-grey).

I checked this on the 20,000-cat sample: the "raw > 0" median is 0.334, the "raw > 127.5" mean is 0.199. Both are fine, but the word "ink fraction" means two different things in the two cells (see the issues list).

---

## Cell 10 (markdown): `## 3. The model, the training loop, and the three key code sections`

Describes the network:

- It is a small U-Net. It shrinks the image 28 → 14 → 7 and grows it back, with "skip connections" that carry detail across.
- The time `t` and the class label are turned into vectors and added together, then fed into every block.
- The output layer starts at all zeros, so an untrained model predicts "speed 0".
- There are 4 class slots, not 3. The extra "null" slot is never trained.


---

## Cell 11 (code): `# === the network`

This cell only defines classes. Nothing runs and nothing prints. There are three classes: `SinusoidalTimeEmbedding`, `ResBlock` and `VelocityUNet`.

### How a PyTorch model is written (first time you see `nn.Module`)

```python
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, c_emb):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        ...
    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        ...
        return h + self.skip(x)
```

Every network in PyTorch follows this pattern:

- `class ResBlock(nn.Module)` says "this is a network piece". Inheriting from `nn.Module` gives it saving, loading, `.to(DEVICE)` and parameter tracking for free.
- `__init__` creates the layers (the parts with learnable numbers, called weights) and stores them as `self.something`. `super().__init__()` must come first; it sets up the `nn.Module` machinery.
- `forward` says what to do with the input. You never call `forward` directly. Writing `block(x, emb)` calls it for you.

### The time embedding

```python
class SinusoidalTimeEmbedding(nn.Module):
    def forward(self, t):
        t = t.view(-1).float()
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device) / (half - 1))
        ang = t[:, None] * freqs[None, :] * 1000.0
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
```

The network needs to know what time `t` it is: near the start, the image is mostly noise; near the end, it is almost a drawing. A single number like 0.3 is hard for a network to use, so it is turned into a list of sines and cosines at many different speeds.

#### The time embedding, simply

Think of a clock. Given just "the hour hand", you know roughly the time. Add the minute hand and the second hand, and you know it exactly. Each sine/cosine pair here is a hand that spins at a different speed. Together they give a detailed "fingerprint" of `t`: two close times get different fingerprints, and the network can pick whichever "hand" is most useful.

Line by line, with a tiny `dim = 8` (the notebook uses 128):

- `half = 4`.
- `freqs = exp(−ln(10000) × [0,1,2,3] / 3)` gives 4 speeds, from fast to slow.
- `t[:, None] * freqs[None, :]` multiplies every time by every speed. `t[:, None]` turns shape `(3,)` into a column `(3, 1)`, and `freqs[None, :]` turns `(4,)` into a row `(1, 4)`. Multiplying a column by a row gives a `(3, 4)` table. This "stretching to fit" is called broadcasting.
- `torch.cat([sin, cos], dim=-1)` puts the 4 sines and 4 cosines side by side, giving 8 numbers per time.

Sample output (`dim = 8`, `t = [0.0, 0.5, 1.0]`):

```text
>>> freqs
tensor([1.0000e+00, 4.6416e-02, 2.1544e-03, 1.0000e-04])
>>> t[:, None].shape, freqs[None, :].shape
(torch.Size([3, 1]), torch.Size([1, 4]))
>>> SinusoidalTimeEmbedding(8)(torch.tensor([0.0, 0.5, 1.0]))      # shape (3, 8)
tensor([[ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000,  1.0000,  1.0000,  1.0000],
        [-0.4680, -0.9380,  0.8810,  0.0500, -0.8840, -0.3470,  0.4740,  0.9990],
        [ 0.8270,  0.6500,  0.8340,  0.1000,  0.5620, -0.7600, -0.5510,  0.9950]])
```

At `t = 0` every angle is 0, so the sines are 0 and the cosines are 1. With `dim = 128`, `t = 0.30` and `t = 0.31` end up a distance of 6.46 apart, so even a 0.01 change in time is clearly visible to the network.


### The residual block

```python
self.norm1 = nn.GroupNorm(min(8, c_in), c_in)
self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
self.emb   = nn.Linear(c_emb, c_out)
...
def forward(self, x, emb):
    h = self.conv1(F.silu(self.norm1(x)))
    h = h + self.emb(F.silu(emb))[:, :, None, None]     # broadcast conditioning over pixels
    h = self.conv2(F.silu(self.norm2(h)))
    return h + self.skip(x)
```

New building blocks:

| Layer | What it does |
|---|---|
| `nn.Conv2d(c_in, c_out, 3, padding=1)` | A convolution: slides a small 3×3 window over the image and mixes the pixels it sees. It finds local patterns like edges. `padding=1` keeps the size at 28×28. It turns `c_in` channels into `c_out` channels |
| `nn.GroupNorm` | Rescales numbers to a steady range so training stays stable |
| `F.silu` | An "activation": a simple bend, `x × sigmoid(x)`. Without bends, many layers would collapse into one straight-line formula |
| `nn.Linear(a, b)` | A fully connected layer: turns `a` numbers into `b` numbers using learnable weights |
| `nn.Identity()` | Does nothing (returns its input). Used when no reshaping is needed |

Two important lines:

- `self.emb(F.silu(emb))[:, :, None, None]` turns the time+class vector (128 numbers per image) into one number per channel, shape `(B, 64)`. `[:, :, None, None]` then adds two size-1 axes, giving `(B, 64, 1, 1)`. Broadcasting stretches that over all 28×28 pixels, so every pixel of a channel gets the same "it is time t, and draw a cat" nudge.
- `return h + self.skip(x)` is the residual part. The block only learns a correction on top of its input. This makes deep networks much easier to train. `skip` is a 1×1 convolution when the channel count changes, otherwise it does nothing.

### The U-Net

```python
def forward(self, t, x, y):
    emb = self.time_embed(t) + self.class_embed(y)
    h0 = self.stem(x)
    h1 = self.down1(h0, emb)                     # (B, c1, 28, 28)
    h2 = self.down2(self.pool1(h1), emb)         # (B, c2, 14, 14)
    m  = self.mid2(self.mid1(self.pool2(h2), emb), emb)          # (B, c2, 7, 7)
    d2 = self.dec2(torch.cat([self.up2(m), h2], dim=1), emb)     # (B, c1, 14, 14)
    d1 = self.dec1(torch.cat([self.up1(d2), h1], dim=1), emb)    # (B, c1, 28, 28)
    return self.out_conv(F.silu(self.out_norm(d1)))
```

- `self.class_embed = nn.Embedding(4, 128)` is a lookup table with 4 rows of 128 learnable numbers. `class_embed(y)` picks row `y` for each image. This is class conditioning: the class tells the network which animal to steer towards. Its vector is added to the time vector, and the sum `emb` goes into every block.
- `pool1`/`pool2` are convolutions with `stride=2`, which move the window 2 pixels at a time and so halve the size. `up1`/`up2` (`ConvTranspose2d`) double it back.
- `torch.cat([..., h2], dim=1)` is a skip connection. It stacks the early, detailed features `h2` next to the upsampled coarse features, along the channel axis (`dim=1`). The coarse path knows "it's a cat shape"; the skip brings back "exactly where the lines are".


Sample output: the shape after each step for a batch of 2 real drawings, with `y = [0, 2]` (cat, fish), `t = [0.3, 0.8]` and a fresh model (`torch.manual_seed(0)`). Shapes are `(batch, channels, height, width)`:

```text
x              (2, 1, 28, 28)     input: 2 grayscale images
stem h0        (2, 64, 28, 28)    1 channel -> 64 "feature maps"
down1 h1       (2, 64, 28, 28)
pool1          (2, 64, 14, 14)    halve the size
down2 h2       (2, 128, 14, 14)   more channels
pool2          (2, 128, 7, 7)     halve again
mid            (2, 128, 7, 7)     the bottom of the "U"
up2            (2, 128, 14, 14)   double the size
cat[up2,h2]    (2, 256, 14, 14)   skip connection: 128 + 128 channels stacked
dec2           (2, 64, 14, 14)
up1            (2, 64, 28, 28)
cat[up1,h1]    (2, 128, 28, 28)   skip connection: 64 + 64
dec1           (2, 64, 28, 28)
out            (2, 1, 28, 28)     same shape as the input: one speed per pixel
>>> emb.shape, m.class_embed.weight.shape
(torch.Size([2, 128]), torch.Size([4, 128]))      # 4 rows: cat, dog, fish, unused "null"
>>> bool((out == 0).all())
True                                              # zero-initialised output layer
```

The shape goes down and back up like the letter U, which is where the name comes from. The output has the same shape as the input because it is a speed for every pixel.

```python
nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)
```

This sets the last layer's weights to 0, so a brand-new model predicts speed 0 everywhere: a calm starting point. The trailing `_` in `zeros_` is PyTorch's convention for "changes the tensor in place".

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
```

`model.parameters()` lists every learnable weight tensor, and `p.numel()` is how many numbers each one holds. The result is 1,838,081 numbers, all learned from scratch.


---

## Cell 12 (code): `# === the three key pieces`

### (a) `flow_matching_batch`: building training examples

```python
def flow_matching_batch(x1, generator=None):
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t  = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype, generator=generator)
    t_b = t.view(-1, 1, 1, 1)                     # per-image scalar, NOT per-pixel
    return (1.0 - t_b) * x0 + t_b * x1, t, (x1 - x0)
```

- `torch.randn(shape)` is random noise from a bell curve (mean 0, spread 1), with the same shape as the drawings. This is `x0`.
- `torch.rand(B)` is one random time between 0 and 1 per image. Note `randn` (bell curve) versus `rand` (uniform).
- `t.view(-1, 1, 1, 1)` reshapes the times from `(B,)` to `(B, 1, 1, 1)`, so that broadcasting applies the same `t` to every pixel of that image. `.view` is PyTorch's `reshape`.
- It returns three things: the in-between image `x_t`, the times `t`, and the target speed `x1 − x0`.

Sample output, a 2×2 "image" (4 pixels) with `generator` seeded at 0:

```text
x1      tensor([ 1., -1., -1.,  1.])                  the "drawing"
x0      tensor([ 1.5410, -0.2934, -2.1788,  0.5684])  random noise
t       tensor([0.4556])                              one time for this image
x_t     tensor([ 1.2945, -0.6154, -1.6417,  0.7651])  (1-0.4556)*x0 + 0.4556*x1
target  tensor([-0.5410, -0.7066,  1.1788,  0.4316])  x1 - x0
```

Check pixel 1 by hand: `(1 − 0.4556) × 1.5410 + 0.4556 × 1 = 0.8389 + 0.4556 = 1.2945`. Target: `1 − 1.5410 = −0.5410`.

For a real batch of 4 drawings (`seed 1`), every image gets its own time: `t = [0.5067, 0.1194, 0.7962, 0.3596]`. The shapes are `x_t (4, 1, 28, 28)`, `t (4,)`, `target (4, 1, 28, 28)`.

The point is that the correct answer is just `x1 − x0`. Nothing has to be simulated to know what the network should say.


### (c) `sample_ode`: generating by following the speeds

```python
@torch.no_grad()
def sample_ode(model, n, labels, n_steps=50, method="euler", device=DEVICE, seed=None, ...):
    model.eval()
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    x = torch.randn((n, *img_shape), device=device, generator=gen)
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        tb = t0.expand(n)
        if method == "euler":
            v = model(tb, x, labels);                                   nfe += 1
            x = x + dt * v
        elif method == "midpoint":
            v1 = model(tb, x, labels);                                  nfe += 1
            x_mid = x + 0.5 * dt * v1
            v2 = model((t0 + 0.5 * dt).expand(n), x_mid, labels);        nfe += 1
            x = x + dt * v2
```

New PyTorch idioms:

- `@torch.no_grad()` is a decorator: a line above a function that changes how it runs. This one turns off gradient tracking (see Cell 13) because we are only using the model, not training it. That saves memory and time.
- `model.eval()` switches the model to "use" mode. It matters for layers like `Dropout` that behave differently while training.
- `torch.Generator().manual_seed(seed)` is PyTorch's version of `np.random.default_rng(seed)`. Same seed, same noise, same drawing.
- `(n, *img_shape)` unpacks `(1, 28, 28)` into the tuple, giving `(n, 1, 28, 28)`.
- `torch.linspace(0, 1, n_steps + 1)` gives evenly spaced times. With 5 steps: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
- `t0.expand(n)` repeats one time for all `n` images, e.g. `tensor(0.2).expand(3)` gives `[0.2, 0.2, 0.2]`.
- NFE ("number of function evaluations") counts network calls per image. That is the real cost of generating.

#### Euler and midpoint, simply

Walking in fog with a compass that tells you which way to go:

- Euler: read the compass, walk `dt` in that direction, read again. One reading per step.
- Midpoint: read the compass, walk half a step, read again there, then go back and take the full step using the second reading. Two readings per step, but it handles curved paths better.

If the path is perfectly straight, both arrive at the same place. That is why the notebook compares them at equal NFE (equal compass readings), not at equal steps.

One real pixel (cat, row 14, column 14), epoch-60 model, seed 101, 10 Euler steps (`dt = 0.1`):

```text
step 1:  t=0.0  x=1.270  v=-1.673  -> x + 0.1*v = 1.103
step 2:  t=0.1  x=1.103  v=-1.586  -> x + 0.1*v = 0.944
step 3:  t=0.2  x=0.944  v=-1.442  -> x + 0.1*v = 0.800
...
step 10: t=0.9  x=0.496  v=0.294   -> x + 0.1*v = 0.526      (ends up as ink, > 0)
```

Sample output, a full call: 3 images (cat, dog, fish), seed 101, 50 Euler steps, epoch-60 model on CPU:

```text
>>> im, nfe = sample_ode(model60, 3, torch.arange(3), n_steps=50, seed=101)
>>> im.shape, nfe
(torch.Size([3, 1, 28, 28]), 50)
>>> im.min(), im.max()
(-1.036, 1.161)          # slightly outside [-1, 1]; later code clamps it
>>> (im > 0).float().mean((1, 2, 3))       # ink fraction per image
tensor([0.1862, 0.1926, 0.1314])            # close to the real 0.198 / 0.181 / 0.140 from Cell 9
>>> _, nfe = sample_ode(model60, 3, torch.arange(3), n_steps=5, method="midpoint", seed=101); nfe
10                       # midpoint: 2 calls per step
```

`torch.arange(3)` is `[0, 1, 2]`, i.e. one cat, one dog and one fish. Section (b), the loss and update step, lives inside `train_model` in Cell 15.


---

## Cell 13 (code): `# === shape and gradient checks`

```python
probe = VelocityUNet(n_classes=len(CATS)).to(DEVICE)
xb = torch.from_numpy(DS["train_x"][:8]).to(DEVICE)
yb = torch.from_numpy(DS["train_y"][:8]).to(DEVICE)
x_t, t_s, target = flow_matching_batch(xb)
pred = probe(t_s, x_t, yb)
assert pred.shape == xb.shape == target.shape
```

- `torch.from_numpy(array)` wraps a NumPy array as a PyTorch tensor (no copy). `.to(DEVICE)` moves it to the CPU or GPU.
- `probe(t_s, x_t, yb)` calls the model's `forward`. The `assert` confirms the prediction has the same shape as the target, which is needed for the loss to make sense.

### Gradients, simply (first time you see `.backward()`)

```python
loss = torch.mean((pred - target) ** 2); loss.backward()
```

- The loss is one number: how wrong the predictions are on average (mean squared error).
- `loss.backward()` works out, for every one of the 1.8 million weights, "if I nudged this weight up a little, would the loss go up or down, and by how much?" That answer is the weight's gradient, stored in `p.grad`. Training then nudges every weight the downhill way (Cell 15).
- Analogy: you are blindfolded on a hill and want to reach the bottom. The gradient is the slope under your feet in each direction.

```python
n_with_grad = sum(1 for _, p in probe.named_parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0)
print(f"class embedding gets a gradient: {probe.class_embed.weight.grad.abs().sum().item() > 0}")
```

This counts how many weight tensors got a non-zero gradient, and specifically whether the class-embedding table did. `.item()` turns a one-number tensor into a plain Python number.

Notebook output (my rerun gives the same):

```text
x1 (8, 1, 28, 28) -> x_t (8, 1, 28, 28)  t (8,)  target (8, 1, 28, 28)  pred (8, 1, 28, 28)

weights receiving a non-zero gradient: 2 of 85
class embedding gets a gradient: False
  (if that were False the conditioning would be dead: ...)
```

The cell's own message says False would mean dead conditioning, and the result is False. This is not a sign that the trained model is broken. It is a flaw in the check:

1. The output layer is zero-initialised. Every signal flowing backwards into the rest of the network has to pass through the output layer's weights, which are all 0, so it gets multiplied by 0. Only the output layer itself gets a gradient: those are the "2 of 85" (`out_conv.weight` and `out_conv.bias`, which I confirmed by name).
2. Even without that, `DS["train_y"][:8]` is `[0 0 0 0 0 0 0 0]`, all cats (the data is sorted by class, see Cell 8). So the check could never test the dog or fish rows.

I ran the same check on the trained epoch-60 model: all 85 weight tensors get a gradient, and the class embedding does too. Only row 0 (cat) is non-zero, `[0.0864, 0, 0, 0]`, because of point 2. A working version of the check would take one training step first, and use a batch with all three classes.

The last loop checks the ends of the path: at `t = 0`, `x_t` equals the noise exactly; at `t = 1`, it equals the drawing. Both differences print as `0.00e+00`.

---

## Cell 14 (markdown): `### The three important code sections, in my own words`

Restates (a) `flow_matching_batch`, (b) the training step inside `train_model`, and (c) `sample_ode`, giving the inputs and outputs of each. The key claims, all of which match the code:

- `t` is one value per image, not per pixel.
- The target needs no simulation.
- The loss line is the whole flow-matching method.
- `eval_loss` uses a fixed set of `t` values so checkpoints compare fairly.
- NFE is returned so the samplers can be compared at equal cost.


---

## Cell 15 (code): `# === training loop`

### `make_loaders`: serving the data in batches

```python
tr = TensorDataset(torch.from_numpy(ds["train_x"]), torch.from_numpy(ds["train_y"]))
return (DataLoader(tr, batch_size=128, shuffle=True, drop_last=True, generator=g, ...),
        DataLoader(ev, batch_size=256, shuffle=False, ...))
```

- `TensorDataset(images, labels)` pairs item `i` of each, so asking for item 5 gives `(image 5, label 5)`.
- `DataLoader` hands these out in batches. `shuffle=True` mixes up the order each epoch. `drop_last=True` throws away the final incomplete batch, so every batch is exactly 128.
- An epoch is one full pass over the training data.

Sample output (rerun, `num_workers=0`):

```text
>>> len(tr_loader), len(ev_loader)
(75, 10)                     # 9600 // 128 = 75 batches; 2400 / 256 = 9.4 -> 10 (last one smaller)
>>> x1, y = next(iter(tr_loader))
>>> x1.shape, x1.dtype, y.shape
(torch.Size([128, 1, 28, 28]), torch.float32, torch.Size([128]))
>>> y[:10]
tensor([0, 1, 2, 1, 0, 1, 1, 0, 0, 2])     # shuffled: classes are mixed now
```

`next(iter(loader))` just grabs the first batch, a common trick to peek at the data.

### `eval_loss`: a fair held-out score

```python
ts = torch.linspace(0.02, 0.98, n_t, device=device)
for x1, y in loader:
    for t_val in ts:
        x0 = torch.randn(x1.shape, device=device, generator=gen)
        ...
        tot += torch.mean((pred - (x1 - x0)) ** 2).item() * x1.shape[0]
        n += x1.shape[0]
return tot / n
```

Training picks random times, but here every held-out drawing is scored at the same 8 times, `[0.02, 0.157, 0.294, 0.431, 0.569, 0.706, 0.843, 0.98]`, with seeded noise. Two checkpoints are then compared on exactly the same test. Multiplying by `x1.shape[0]` and dividing by `n` gives a proper average even though the last batch is smaller.

### `train_model`: the training loop, section (b)

```python
model = VelocityUNet(...).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr_loader))

for ep in range(1, epochs + 1):
    model.train()
    for x1, y in tr_loader:
        if augment_hflip:
            flip = torch.rand(x1.shape[0], device=device, generator=gen) < 0.5
            x1 = torch.where(flip.view(-1, 1, 1, 1), x1.flip(-1), x1)
        x_t, t, target = flow_matching_batch(x1, generator=gen)
        loss = torch.mean((model(t, x_t, y) - target) ** 2)     # <-- the CFM loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
```


Line by line:

- `AdamW` is the optimiser: the rule that turns gradients into weight changes. `lr = 2e-4` (0.0002) is the learning rate, i.e. how big each nudge is. `weight_decay` gently pulls weights towards 0 to avoid extreme values.
- `CosineAnnealingLR` slowly lowers the learning rate along a cosine curve, from 0.0002 down to 0 by the last step. Big steps early, careful steps at the end. `T_max = 60 × 75 = 4,500` steps.
- `model.train()` switches to training mode (the opposite of `model.eval()`).
- The flip: `torch.rand(B) < 0.5` is True for about half the images, and `x1.flip(-1)` mirrors images left-to-right (`-1` is the last axis, the width). `torch.where(cond, a, b)` picks `a` where `cond` is True, otherwise `b`, the same as `np.where` in Part A. A cat facing left is still a cat, so this gives free extra variety.
- The four lines that make up every PyTorch training step:
  1. `opt.zero_grad()` clears the old gradients. PyTorch adds new gradients to old ones, so they must be reset each step.
  2. `loss.backward()` computes new gradients.
  3. `clip_grad_norm_(..., 1.0)` caps the total gradient size at 1 so one bad batch cannot throw the weights off.
  4. `opt.step()` changes the weights. `sched.step()` then lowers the learning rate slightly.

Sample output (tiny illustration): the cosine schedule with `T_max = 4` instead of 4,500:

```text
lr at steps 0..4: [0.0002, 0.000171, 0.0001, 0.0000293, 0.0]
```

Sample output (tiny illustration): the `flip` / `torch.where` step on two 2×2 "images" with `flip = [True, False]`:

```text
>>> a[0, 0]                               # original
tensor([[0., 1.],
        [2., 3.]])
>>> torch.where(flip.view(-1,1,1,1), a.flip(-1), a)[:, 0]
tensor([[[1., 0.],                        # image 0: flipped (columns swapped)
         [3., 2.]],
        [[0., 1.],                        # image 1: unchanged
         [2., 3.]]])
```

Sample output (tiny illustration, not the real training): 3 training steps on the same 8 drawings with a fresh model, `torch.manual_seed(0)`:

```text
step 1 loss 1.8003   grad norm before clip 10.33
step 2 loss 1.7664   grad norm before clip 9.83
step 3 loss 1.7247   grad norm before clip 10.91
loss of predicting zero everywhere = mean(target²) = 1.8003
```

- Step 1's loss equals the "predict zero" loss exactly. That is the zero-initialised output layer at work.
- The loss then drops a little each step.
- The gradient norm is about 10, so clipping to 1 really does shrink every update here.

The real run did this 4,500 times. Its first logged step had loss 1.185, and the final epoch averaged 0.292.

### Saving and loading checkpoints

```python
torch.save({"model": model.state_dict(), "epoch": ep, "config": log["config"], ...}, ck)
...
ck = torch.load(path, map_location=device, weights_only=False)
model = VelocityUNet(n_classes=cfg["n_classes"], base=cfg["base"], emb_dim=cfg["emb_dim"]).to(device)
model.load_state_dict(ck["model"]); model.eval()
```

- `state_dict()` is a dictionary of every weight tensor, by name. `torch.save` writes it, plus some notes, to a `.pt` file.
- To load, you build an empty model with the same settings and then pour the saved weights in with `load_state_dict`.
- `map_location=device` lets a model trained on a GPU load on a CPU, which is exactly what happened in this run.

Sample output (the real `ckpt_epoch060.pt`):

```text
>>> list(ck.keys())
['model', 'epoch', 'config', 'train_loss', 'eval_loss', 'categories']
>>> ck["epoch"], ck["train_loss"], ck["eval_loss"]
(60, 0.291679234902064, 0.29217179692039885)
>>> list(ck["model"].keys())[:2], len(ck["model"])
(['time_embed.1.weight', 'time_embed.1.bias'], 85)       # 85 named weight tensors
```


---

## Cell 16 (code): `# === baseline: load or train`

```python
BASE_CFG = dict(epochs=60, batch_size=128, lr=2e-4, ...)
NEEDED = [os.path.join(BASE_DIR, f"ckpt_epoch{e:03d}.pt") for e in (5, 15, 60)]
if os.path.exists(BASE_LOG_PATH) and all(os.path.exists(p) for p in NEEDED):
    base_log = json.load(open(BASE_LOG_PATH))
else:
    _, base_log = train_model(DS, BASE_DIR, **BASE_CFG)
model60, ck60 = load_checkpoint(os.path.join(BASE_DIR, "ckpt_epoch060.pt"))
```

- `f"{e:03d}"` pads with zeros to 3 digits, so 5 becomes `005` and gives `ckpt_epoch005.pt`.
- `all(...)` is True only if every checkpoint file exists. If they all do, it loads the training log instead of retraining.
- `**BASE_CFG` unpacks the dictionary into keyword arguments, the same as writing `train_model(DS, BASE_DIR, epochs=60, batch_size=128, ...)`.
- `_, base_log = ...` ignores the first returned value, the model.

Notebook output (abridged):

```text
baseline checkpoints found -> loading, not retraining.
  n_train_original    : 9600
  params_total        : 1838081
  epochs              : 60
  ...
  device              : cuda
  gpu                 : NVIDIA GeForce RTX 3060
  total train time    : 8.10 min
  final train loss    : 0.29168
  final eval loss     : 0.29217
```

These values are read from the saved log (`training_log.json`), so they describe the original training run on an RTX 3060 GPU, not this CPU session. The log holds 60 epoch entries and 180 step entries (one every 25 steps). Its first epoch shows `train_loss 1.015, eval_loss 0.596`.

---

## Cell 17 (code): `# === training curves`

```python
ep_df = pd.DataFrame(base_log["epochs"])
st_df = pd.DataFrame(base_log["steps"])
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
```

Three charts from the saved log:

1. Training loss per logged step (thin line) with the epoch average on top. `ep_df["epoch"] * (st_df["step"].max() / ep_df["epoch"].max())` converts epoch numbers to step numbers so both lines share one x-axis (4,500 / 60 = 75 steps per epoch).
2. Train vs held-out loss per epoch. `ax.axvline` draws dotted lines at the checkpoints (5, 15, 60).
3. Held-out minus train loss, an overfitting check. If the model were memorising, the held-out loss would climb above the train loss.

Notebook output:

```text
held-out loss never exceeds train loss by more than 0.0070 -> no sign of overfitting at this size.
```

`(ep_df['eval_loss']-ep_df['train_loss']).max()` gives the largest gap, 0.0070. The cell rightly warns that this loss is an error on speeds, so a lower loss does not by itself mean nicer drawings.

---

## Cell 18 (code): `# === noise -> drawing trajectory`

```python
labels_show = torch.tensor([0, 0, 1, 1, 2, 2], device=DEVICE)
imgs, nfe, traj = sample_ode(model60, n_show, labels_show, n_steps=50, method="euler",
                             seed=SEED_LIST[0], return_trajectory=True, traj_every=6)
start = torch.randn((n_show, 1, IMG, IMG), generator=torch.Generator(device=DEVICE).manual_seed(SEED_LIST[0]), ...)
stages = [start] + traj
ts_lbl = [0.0] + [min(1.0, (i * 6 + 1) / 50) for i in range(len(traj))]
```

- This generates 2 cats, 2 dogs and 2 fish and saves snapshots along the way, to show noise turning into drawings.
- `start` re-creates the starting noise with the same seed. `sample_ode` itself does not save the image before step 1, and the same seed gives exactly the same `x0`.
- `[start] + traj` joins lists. The column titles `ts_lbl` give the time at each snapshot: a snapshot taken after step `i` is at time `(i + 1) / 50`.


Sample output (rerun):

```text
>>> len(traj), traj[0].shape
(10, torch.Size([6, 1, 28, 28]))
>>> [i for i in range(50) if i % 6 == 0 or i == 49]     # which steps are saved
[0, 6, 12, 18, 24, 30, 36, 42, 48, 49]
>>> ts_lbl
[0.0, 0.02, 0.14, 0.26, 0.38, 0.5, 0.62, 0.74, 0.86, 0.98, 1.0]
```

The rule `i % traj_every == 0 or i == n_steps - 1` saves every 6th step plus the last one. `%` is the remainder: `12 % 6 == 0`. The figure has 11 columns (start + 10 snapshots). The notebook prints `NFE per image for this path: 50`.

---

## Cell 19 (markdown): `## 4–6. Controlled comparisons and scoring`

Describes the protocol:

- The same 32 seeds are used everywhere, giving 96 drawings per setting.
- Only one factor changes at a time.
- Recognisability is the share of generated drawings that a separate classifier, trained only on real held-out drawings, labels as the class that was asked for.

---

## Cell 20 (code): `# === the scoring CNN`

### The evaluation classifier, simply

You cannot check thousands of drawings by eye, so a small "judge" network is trained to look at a doodle and say cat, dog or fish. Then you ask: when I asked the generator for a cat, did the judge say cat? The judge only ever sees real held-out drawings during its own training, so it has no connection to the generator.

```python
class DrawingCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14 -> 7
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))
```

- `nn.Sequential(a, b, c)` chains layers: the output of `a` feeds `b`, and so on.
- `nn.ReLU` is an activation that turns negatives into 0.
- `nn.MaxPool2d(2)` keeps the biggest value in each 2×2 square, halving the size.
- `AdaptiveAvgPool2d(1)` averages each channel down to one number.
- `Flatten` turns `(B, 128, 1, 1)` into `(B, 128)`.
- `Dropout(0.3)` randomly switches off 30% of the numbers during training only, which reduces overfitting.
- `Linear(128, 3)` gives 3 scores, one per class.

Sample output (the saved classifier on 3 real held-out drawings, one per class):

```text
  Conv2d             (3, 32, 28, 28)
  ReLU               (3, 32, 28, 28)
  MaxPool2d          (3, 32, 14, 14)
  Conv2d             (3, 64, 14, 14)
  ...
  AdaptiveAvgPool2d  (3, 128, 1, 1)
```


### Logits, softmax, argmax and cross-entropy

```text
true labels:  [0 1 2]                     (a real cat, a real dog, a real fish)
logits     tensor([[ 0.5639, -0.0074, -3.3199],
                   [-0.3640, -0.4744, -1.0387],
                   [-2.0000, -1.3949,  2.4342]])
softmax    tensor([[0.6308, 0.3563, 0.0130],
                   [0.4159, 0.3724, 0.2118],
                   [0.0115, 0.0210, 0.9675]])
argmax     tensor([0, 0, 2])
```

- Logits are the raw scores the network outputs: bigger means "more likely this class". They can be any number.
- Softmax turns them into probabilities that add up to 1 across each row. For the fish, the judge is 96.75% sure.
- `.argmax(1)` is the position of the biggest value in each row, i.e. the predicted class. The second drawing, a real dog, is predicted 0 (cat). That is a real mistake, and the reason the "ceiling" below is not 100%.
- Training uses `F.cross_entropy(logits, labels)`, the average of `−log(probability given to the true class)`. Here that is `−(log 0.6308 + log 0.3724 + log 0.9675)/3 = 0.494`, which I checked by both routes.

### Training the judge, and the ceiling

```python
perm = np.random.default_rng(555).permutation(n_ev)
cut = int(0.8 * n_ev)
ci, vi = perm[:cut], perm[cut:]
```

The 2,400 held-out drawings are split again: 1,920 to train the judge, 480 to test it.

`train_classifier` does the same zero_grad → backward → step loop as Cell 15, but runs it by hand over shuffled indices (`torch.randperm`) rather than with a DataLoader. `range(0, len(perm) - batch_size + 1, batch_size)` steps through the list in blocks of 128 and skips the last partial block. Since `eval_classifier.pt` already exists, the notebook loads it instead.

```python
with torch.no_grad():
    acc = (model(xv).argmax(1) == yv).float().mean().item()
```

- `with torch.no_grad():` is the block version of the decorator in Cell 12: no gradients inside it.
- `(pred == true)` gives True/False, `.float()` turns that into 1.0/0.0, and `.mean()` gives the accuracy.

Notebook output:

```text
scoring CNN accuracy on REAL held-out drawings: 0.8271
per-class ceiling on real drawings: {'cat': 0.8397, 'dog': 0.7396, 'fish': 0.9097}
```

The judge is right on 82.7% of real doodles. A generated set cannot fairly be expected to score above that, so 0.8271 is the "ceiling". The saved history shows accuracy climbing from 0.529 at epoch 1 to 0.827 at epoch 30. The test split has 156 cats, 169 dogs and 155 fish.


---

## Cell 21 (code): `# === measuring helpers`

Six helper functions. Nothing prints except a confirmation line.

`recognisability(images, labels)` runs the judge, takes `argmax`, and returns the share correct (`overall_accuracy`). It also returns the average probability the judge gave to the requested class (`mean_confidence`) and per-class versions of both. `.clamp(-1, 1)` first cuts values to [−1, 1], because generated images can overshoot slightly. `prob[torch.arange(n), labels]` picks, from row `i`, the column `labels[i]`: "the probability of the class I asked for".

`variety(images)` is the average distance between every pair of drawings in a set. If the model drew the same cat every time, it would be 0.

```python
x = images.reshape(len(images), -1)        # each image -> one row of 784 numbers
d = torch.cdist(tsr, tsr)                  # table of distances between every pair
iu = torch.triu_indices(len(tsr), len(tsr), offset=1)
return float(d[iu[0], iu[1]].mean())
```

Sample output, the 3 generated images from Cell 12 (seed 101):

```text
>>> torch.cdist(x, x)                        # distance from image i to image j
tensor([[ 0.0000, 23.4801, 22.6505],
        [23.4801,  0.0000, 22.2035],
        [22.6505, 22.2035,  0.0000]])
>>> torch.triu_indices(3, 3, offset=1)       # the pairs above the diagonal: (0,1) (0,2) (1,2)
tensor([[0, 0, 1],
        [1, 2, 2]])
>>> variety(im.numpy())                      # (23.48 + 22.65 + 22.20) / 3
22.778
```

The distance is the ordinary straight-line distance across 784 pixels. Only the upper triangle is used, so each pair counts once and the zeros on the diagonal are skipped. For scale, a fully blank image and a fully inked one are `sqrt(784 × 2²) = 56` apart.

`nearest_train_distance(gen, train)`: for each generated drawing, the distance to the closest of the 9,600 training drawings. `.min(dim=1)` returns both the smallest value and its position. For the 3 images above it gives `[15.0, 18.6, 12.9]`, so none is a near-copy.

`generate_config(model, n_steps, method)` loops over the 32 seeds. Each seed generates `y = [0, 1, 2]` (one per class), and the results are glued together. Sample output with just 2 seeds and 10 steps:

```text
>>> ims, lbs, sds, nfe, wall = generate_config(model60, 10, "euler", seeds=SEED_LIST[:2])
>>> ims.shape, lbs, sds, nfe
((6, 1, 28, 28), [0 1 2 0 1 2], [101 101 101 202 202 202], 10)
```

With all 32 seeds, that is 96 images.

`score_config(images, labels, tag)` collects every measure into one row (a dictionary). It includes the standard error of the score: `sqrt(acc × (1 − acc) / 96)`. For `acc = 0.7708` that is 0.0429, and the 95% range is `acc ± 1.96 × 0.0429` = 0.687 to 0.855. That matches `recog_ci_low` / `recog_ci_high` in the results table.

`save_grid` and `montage` draw grids of drawings. `save_grid` also saves every image to a `.npz` file (`np.savez_compressed`), so nothing is hand-picked.

---

## Cell 22 (code): `# === reproducibility check`

```python
a, _ = sample_ode(model, 3, y, n_steps=50, method=method, seed=seed)
b, _ = sample_ode(model, 3, y, n_steps=50, method=method, seed=seed)
c, _ = sample_ode(model, 3, y, n_steps=50, method=method, seed=seed + 1)
return bool(torch.equal(a, b)), bool(not torch.equal(a, c)), float((a - b).abs().max())
```

The same seed must give exactly the same pictures, and a different seed different ones. `torch.equal` is True only if every number matches.

My rerun on CPU gives `True 0.0` for the same seed, `False` for seed 102 (largest pixel difference 2.16), matching the notebook's `True` / `True` (the second value there is the opposite test: "differs"). Generation is random only through the starting noise; the stepping itself is a fixed recipe.


---

## Cell 23 (code): `# === COMPARISON 1 (training factor): checkpoint epoch`

```python
for epn in CKPT_EPOCHS:                                   # [5, 15, 60]
    mdl, _ = load_checkpoint(os.path.join(BASE_DIR, f"ckpt_epoch{epn:03d}.pt"))
    ims, lbs, sds, nfe, wall = generate_config(mdl, 50, "euler")
    row = score_config(ims, lbs, tag)
    tinfo = next(c for c in base_log["checkpoints"] if c["epoch"] == epn)
    row.update({"epoch": epn, "train_loss": tinfo["train_loss"], ...})
```

Research question Q1: does training longer help? Each checkpoint generates the same 96 drawings (same seeds, 50 Euler steps) and is scored.

- `next(c for c in ... if ...)` returns the first item in the list that matches the condition. Here, it is the log entry for this epoch.
- `row.update({...})` adds more keys to the dictionary.
- `del mdl` frees the model from memory before loading the next one.

Notebook output:

```text
epoch   5: recognisability 0.7708 (+/-0.043) | confidence 0.6639 | variety 22.35 | eval loss 0.3570
epoch  15: recognisability 0.8125 (+/-0.040) | confidence 0.7088 | variety 22.11 | eval loss 0.3164
epoch  60: recognisability 0.7708 (+/-0.043) | confidence 0.7135 | variety 22.23 | eval loss 0.2922
```

How to read it:

- 0.7708 = 74 of 96 drawings judged as the requested class. The `+/-0.043` is one standard error (Cell 21).
- The held-out loss falls (0.357 → 0.292, an 18% drop) and confidence rises.
- Recognisability does not clearly change: 0.77 → 0.81 → 0.77, all within about one standard error of each other.

---

## Cell 24 (code): `# === secondary training comparison: is the limit data or training time?`

Loads (or would train) a second model on a quarter of the data, `build_dataset(n_per_cat=1000)`, giving 2,400 training drawings, with the same settings. It generates and scores the same 96 seeds, then puts both runs in a two-row table. `.loc[ck_df.epoch == 60, "recognisability"].iloc[0]` means "the recognisability value in the row where epoch is 60"; `.iloc[0]` takes the first (only) match.

Notebook output:

```text
                                run  n_train_original  eval_loss  recognisability  mean_variety  train_time_min
    baseline (9,600 train drawings)              9600     0.2922           0.7708       22.2270          8.1003
quarter data (2,400 train drawings)              2400     0.3295           0.8646       22.0389          2.1059
```

With a quarter of the data, the held-out loss is worse (0.330 vs 0.292). But recognisability is higher (0.865 vs 0.771), and training took about a quarter of the time (2.1 vs 8.1 min), because 60 epochs over less data is fewer steps. Cell 30's text says the opposite on both points (see the issues list).

---

## Cell 25 (code): `# === COMPARISON 2 (inference factor): steps and solver`

```python
STEP_GRID = [2, 5, 10, 25, 50, 100]
for method in ["euler", "midpoint"]:
    for ns in STEP_GRID:
        ims, lbs, sds, nfe, wall = generate_config(model60, ns, method)
        row = score_config(ims, lbs, tag)
        row.update({..., "ms_per_image": 1000 * wall / len(ims)})
```

Q2: the same epoch-60 model and the same seeds; only the sampler changes. That is 12 configurations. `ms_per_image` is wall time in milliseconds per image.

Notebook output (abridged):

```text
euler     steps=  2  NFE/img=   2  recog=0.8438  conf=0.6718  variety= 14.99    78.9 ms/img
euler     steps= 10  NFE/img=  10  recog=0.8021  conf=0.7279  variety= 21.55   250.4 ms/img
euler     steps= 50  NFE/img=  50  recog=0.7708  conf=0.7135  variety= 22.23  1403.5 ms/img
euler     steps=100  NFE/img= 100  recog=0.7604  conf=0.7113  variety= 22.31  3791.7 ms/img
midpoint  steps=  5  NFE/img=  10  recog=0.7812  conf=0.7167  variety= 22.41   399.3 ms/img
midpoint  steps= 50  NFE/img= 100  recog=0.7604  conf=0.7086  variety= 22.41  1751.6 ms/img
midpoint  steps=100  NFE/img= 200  recog=0.7604  conf=0.7081  variety= 22.41  1224.1 ms/img
```

The clear effect is variety at 2 Euler steps: 14.99 against about 22 elsewhere, a one-third drop. The drawings are "unfinished" and too similar to each other. The timings were measured on CPU in this run and are noisy. For example, midpoint with 200 NFE (1224 ms) came out faster than midpoint with 100 NFE (1752 ms), which cannot reflect the real cost.

### Matched NFE, the fair comparison

```python
for ns in STEP_GRID:
    e = inf_df[(inf_df.method == "euler") & (inf_df.nfe_per_image == ns)]
    m = inf_df[(inf_df.method == "midpoint") & (inf_df.nfe_per_image == ns)]
    if len(e) and len(m):
        d = m.recognisability.iloc[0] - e.recognisability.iloc[0]
        se = float(np.sqrt(e.recog_stderr.iloc[0] ** 2 + m.recog_stderr.iloc[0] ** 2))
        matched.append({..., "resolvable at 95%": bool(abs(d) > 1.96 * se)})
```

- For each network-call budget, it finds the Euler row and the midpoint row with that NFE. These exist only for 10, 50 and 100 (e.g. Euler 10 steps vs midpoint 5 steps). `if len(e) and len(m)` skips budgets where one is missing.
- The standard error of a difference is `sqrt(se₁² + se₂²)`. For NFE 10: `sqrt(0.0407² + 0.0422²) = 0.0586`.
- `resolvable at 95%` asks whether the difference is bigger than 1.96 standard errors.

Notebook output:

```text
 NFE per image  euler steps  euler recog  midpoint steps  midpoint recog  difference (mid - euler)  stderr of difference  resolvable at 95%
            10           10       0.8021               5          0.7812                   -0.0208                0.0586              False
            50           50       0.7708              25          0.7604                   -0.0104                0.0611              False
           100          100       0.7604              50          0.7604                    0.0000                0.0616              False
```

Every difference is far smaller than its uncertainty, so at equal cost neither solver is measurably better.

---

## Cell 26 (code): `# === comparison figures`

Four charts, built with the same matplotlib calls as before:

1. Recognisability by checkpoint epoch, with 95% error bars (`ax.errorbar(..., yerr=1.96*stderr)`) and the 0.8271 ceiling as a dashed line. `ax.twinx()` adds a second y-axis on the right for the held-out loss.
2. Recognisability against step count (log x-axis).
3. Variety against NFE.
4. Milliseconds per image against NFE (log–log).

No text output.

---

## Cell 27 (code): `# === matched grids for the report`

Calls `montage` three times. Each one puts configurations in stacked row-blocks, with the same seed in every column, so you compare like with like by eye:

- epochs 5 / 15 / 60
- Euler 2 / 10 / 50 steps
- Euler 10 vs midpoint 5 (both 10 NFE)

`store_ck[5]` and `store_inf["euler2"]` are the `(images, labels, seeds)` tuples saved in Cells 23 and 25.

---

## Cell 28 (code): `# === 7. copying probe`

```python
gen_d, gen_i = nearest_train_distance(gen_ims, DS["train_x"])
real_idx = np.random.default_rng(3).choice(len(DS["eval_x"]), 200, replace=False)
real_d, _ = nearest_train_distance(DS["eval_x"][real_idx], DS["train_x"])
```

Did the model just memorise training drawings? For each of the 96 epoch-60 drawings, it finds the closest training drawing. As a fair yardstick, it does the same for 200 real held-out drawings the model never saw. If generated drawings were copies, they would sit much closer to the training set than genuinely new real drawings do.

Notebook output:

```text
                                      set   n  mean distance     min  5th pct     max
        generated (epoch 60, 96 drawings)  96        16.0457 10.0311  11.4386 21.1687
real HELD-OUT drawings (never trained on) 200        15.6896  8.7994  10.7198 23.6829
```

Generated drawings average 16.05 from their nearest training drawing, against 15.69 for real unseen ones. They are not closer, which is evidence against copying. The figure shows the 6 closest pairs: `np.argsort(gen_d)[:6]` lists the positions sorted from smallest distance up and takes the first 6. `fig.add_axes([left, bottom, width, height])` places each small picture by hand inside the right-hand panel.

---

## Cell 29 (code): `# === consolidated results`

```python
full = pd.concat([ck_df.assign(comparison="1: checkpoint"),
                  pd.DataFrame([row_s]).assign(comparison="1b: dataset size"),
                  inf_df.assign(comparison="2: sampler")], ignore_index=True, sort=False)
full = full[[c for c in keep if c in full.columns]]
```

- `.assign(comparison="...")` adds a column with the same value in every row, labelling where each row came from.
- `pd.concat` stacks the three tables: 3 + 1 + 12 = 16 rows. This is the same pattern as Part A, Cell 19.
- `[c for c in keep if c in full.columns]` keeps the wanted columns in the wanted order, skipping any that do not exist.

It saves `C_all_results.csv` and prints the real-data reference values. In the printed table, `NaN` means "not applicable": checkpoint rows have no `ms_per_image`, and sampler rows have no training time.

---

## Cell 30 (markdown): `## 8. What the results mean`

The written interpretation for the report. Its main points:

- Q1: held-out loss fell about 18% (0.357 → 0.292, confirmed) but recognisability did not measurably move. Lower loss is not better pictures.
- Q2: about 5 steps are enough. 2 steps loses about a third of the variety (confirmed: 14.99 vs ~22.3). Euler and midpoint cannot be told apart at equal NFE.
- Generation is random only through the starting noise.
- Copying: generated drawings are not closer to the training set than unseen real drawings.

Several numbers in this cell do not match the outputs above (see the issues list): the dataset-size paragraph, the "84%" ceiling, the "96% of it" figure, and the "40-fold" cost.

---

## Cell 31 (code): `# === reproduction record`

Builds one big nested dictionary: data manifest, model size, training settings, scorer accuracy, seed list, every comparison table, and the environment. It saves it as `C_reproduction_record.json`.

- `size_df.to_dict("records")` turns a table into a list of row dictionaries, which JSON can store.
- `json.dump(..., default=float)` tells `json` to convert anything it does not recognise (like a NumPy number) with `float()`. That replaces the manual `float(...)` calls used in Part A.

Notebook output (abridged):

```text
PART C COMPLETE

parameters trained  : 1,838,081 (all, from scratch)
training time       : 8.1 min on cpu
outputs per config  : 96 (32 per class)
configurations      : 16
scorer ceiling      : 0.8271 on real held-out drawings
```

"8.1 min on cpu" is wrong. The 8.1 minutes is from the saved log, which records training on an NVIDIA RTX 3060 (Cell 16). The word "cpu" comes from `record['environment']['gpu']`, which describes the machine running the notebook now. That machine has no GPU, so it falls back to `'cpu'`.

---

## Issues found in the notebook

I did not change anything in the notebook. Each issue was checked against the saved outputs or by rerunning the code.

1. **The Cell 13 gradient check fails, and the reason given is misleading.** It prints `class embedding gets a gradient: False` and `2 of 85`. Its own message says False would mean dead conditioning. The real cause is the zero-initialised output layer, which blocks every gradient except its own on a fresh model. The test batch is also all cats (`train_y[:8]` = all 0). On the trained model all 85 tensors get gradients. Fix: run the check after one optimiser step, on a batch with all three classes. Otherwise anyone reading the output sees a "failed" check.
2. **Cell 30's dataset-size paragraph contradicts Cell 24's output.** It says quartering the data "at identical training time" lowers recognisability. Cell 24 shows recognisability rose (0.8646 vs 0.7708), and training time was 2.1 vs 8.1 min. Only the worse held-out loss (0.330 vs 0.292) matches.
3. **Cell 30 quotes stale numbers.**
   - It says the scorer is right "about 84%" of the time; Cell 20 gives 0.8271.
   - It says epoch 60 reaches "about 96%" of the ceiling; it is 0.7708 / 0.8271 = 93%.
   - It says the cost rises "roughly 40-fold"; 3791.7 / 78.9 = 48.
   - It says recognisability from 2 to 100 Euler steps is "within one margin of error"; the spread is 0.8438 − 0.7604 = 0.083, about 2 standard errors.
4. **Cell 31 prints "training time: 8.1 min on cpu".** The training ran on an RTX 3060 (Cell 16); "cpu" describes the current session. It should read the device from `base_log["config"]["gpu"]`.
5. **"Ink fraction" means two things.** Cell 7 counts raw > 0 (cat median 0.334). Cell 9, `score_config` and Cell 29 count rescaled > 0, i.e. raw > 127.5 (cat mean 0.198). Both are reasonable, but the same name for both is confusing, and Cell 7's "0.16 to 0.43" is not comparable with the generated-drawing ink numbers.
6. **"Margin of error" is used for a standard error.** Cell 3's "±0.051" and Cell 30's "±0.041" are single standard errors. A 95% margin would be about 1.96 times that. Cell 30's "differences below about 0.09 are not resolvable" is roughly right, so only the wording is off.
7. **The CPU timings in Cell 25 are noisy** (midpoint 200 NFE faster than midpoint 100 NFE). They are fine as rough orders of magnitude, but not for precise cost claims.

---

## Quick reference: Python and PyTorch terms used above

| Term | Meaning | First seen |
|---|---|---|
| tensor | PyTorch's array; like a NumPy array but can live on a GPU and track gradients | Cell 3 |
| shape `(B, C, H, W)` | batch size, channels, height, width; e.g. `(8, 1, 28, 28)` | Cell 8 |
| `DEVICE`, `.to(DEVICE)` | where the maths runs (`"cpu"` or `"cuda"` GPU) and moving things there | Cell 3 |
| `mmap_mode="r"` | read a big `.npy` file lazily, row by row, without loading it all | Cell 6 |
| walrus `:=` | assign and test in one expression | Cell 6 |
| `reshape` / `.view` | change the shape without changing the numbers; `-1` = "work it out" | Cell 8, Cell 11 |
| broadcasting | size-1 axes stretch to fit, e.g. `(B,1,1,1) × (B,1,28,28)` | Cell 11 |
| `nn.Module`, `__init__`, `forward` | how a network is defined: layers in `__init__`, computation in `forward`; call it as `model(x)` | Cell 11 |
| `nn.Conv2d`, `nn.Linear`, `nn.Embedding` | convolution (local image patterns), fully connected layer, lookup table | Cell 11 |
| activation (`SiLU`, `ReLU`) | a simple bend that lets layers stack into something more than a straight line | Cell 11 |
| skip connection, `torch.cat(..., dim=1)` | stack early detailed features next to later ones along the channel axis | Cell 11 |
| `torch.randn` / `torch.rand` | bell-curve noise / uniform 0–1 numbers | Cell 12 |
| `torch.Generator().manual_seed(s)` | seeded random source, same seed gives same numbers | Cell 12 |
| `@torch.no_grad()`, `with torch.no_grad():` | turn off gradient tracking when only using a model | Cell 12, Cell 20 |
| `model.eval()` / `model.train()` | "use" mode / "training" mode | Cell 12, Cell 15 |
| NFE | network calls per image, the real cost of generating | Cell 12 |
| `loss.backward()`, `p.grad` | compute gradients (the slope for every weight) | Cell 13 |
| `.item()` | one-number tensor to a plain Python number | Cell 13 |
| `TensorDataset`, `DataLoader`, epoch, batch | serve the data in shuffled batches; one epoch = one full pass | Cell 15 |
| `opt.zero_grad()`, `opt.step()` | clear old gradients; update the weights | Cell 15 |
| learning rate, `CosineAnnealingLR` | size of each update; schedule that lowers it smoothly to 0 | Cell 15 |
| `state_dict`, `torch.save`, `load_state_dict` | save and restore all weights | Cell 15 |
| `**dict` | unpack a dictionary into keyword arguments | Cell 16 |
| logits, softmax, argmax | raw class scores; turned into probabilities; position of the largest | Cell 20 |
| `F.cross_entropy` | the classification loss: average of −log(probability of the true class) | Cell 20 |
| `torch.cdist`, `torch.triu_indices` | all pairwise distances; the index pairs above the diagonal | Cell 21 |
| standard error | how much a score would wobble by chance: `sqrt(p(1−p)/n)` | Cell 21 |
