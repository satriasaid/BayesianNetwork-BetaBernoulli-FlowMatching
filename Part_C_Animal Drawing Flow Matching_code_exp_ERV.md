# Part C: Animal Drawing Flow Matching, explained in plain words

This is an easier version of `Part_C_Animal Drawing Flow Matching_code_exp.md`. It explains `PartC02_Animal Drawing Flow Matching.ipynb` cell by cell, and it uses one story all the way through: an art student who learns to turn TV static into doodles.

How to use this file:

- The cell numbers (Cell 1, Cell 2, ...) are the same as in the original explainer, so you can jump between the two files.
- Only Cell 2 has a notebook id. For every other cell, the heading quotes the cell's first comment line (for example `# === the network`). Press Ctrl+F in the notebook and search for that text.
- Each code cell has up to four parts: what it does in one sentence, an analogy, the key lines in plain words, and a sample output with every number explained.
- Only the most important lines of code are quoted. The notebook has the full code.
- The numbers come from the notebook's saved outputs and from real runs of its code with the saved models. Nothing here is made up, and nothing was retrained.

---

## The whole project in plain words

### The story: an art student and TV static

Picture an old TV that is not tuned to any channel. The screen is full of random grey dots, which is called "static" or "snow". Now picture an art student who is given a screen full of that static and asked: "Turn this into a doodle of a cat."

The student cannot do it in one go. Instead they learn a skill: look at the picture as it is right now, and decide which way to nudge every dot. Some dots should get darker, some lighter, some should stay put. After a small nudge the picture looks slightly less like static. Look again, nudge again. After enough small nudges, a cat appears.

That is exactly what this notebook builds:

| In the story | In the notebook |
|---|---|
| The TV static at the start | Random noise, a picture of random numbers |
| The finished doodle | A 28 by 28 pixel drawing of a cat, dog or fish |
| "Which way should each dot move, and how much?" | The velocity (speed and direction for every pixel) |
| The student's skill | A neural network called a U-Net |
| Taking many small nudges | Euler sampling (or "midpoint" sampling, a slightly smarter way to step) |
| Being told "draw a cat" | The class label (cat = 0, dog = 1, fish = 2) |
| A judge who looks at the finished doodle and guesses the animal | The evaluation classifier (a second, separate network) |

### Where the drawings come from

Google ran an online game called Quick, Draw!. Players had 20 seconds to doodle a word like "cat". Millions of doodles were saved and shared for free. Each doodle is a tiny grey picture, 28 pixels wide and 28 pixels tall, so 784 pixels in total. Each pixel is a number from 0 (white paper) to 255 (full black ink).

The notebook uses three animals: cat, dog and fish. It takes 4,000 doodles of each. Of those, 3,200 per animal (9,600 in total) are used for learning, and 800 per animal (2,400 in total) are kept aside and never shown to the student. Those are "held out" for fair testing later.

### How the student learns (training)

This is the clever part. To teach the student, you need examples of the right answer. They are easy to make:

1. Take a real cat doodle.
2. Make a screen of random static.
3. Imagine a straight-line fade from the static to the cat, like a film that slowly dissolves from one picture into another.
4. Freeze the film at a random moment, say 30% of the way through. The frame is a blurry mix: 70% static and 30% cat.
5. Ask the student: "From this frame, which way should every dot move?" The right answer is simple: straight towards the cat.

The student guesses, you compare the guess with the right answer, and you correct the student a tiny bit. Repeat this thousands of times with different doodles, different static and different moments. That is training.

### How the student draws something new (generating)

Give the student fresh static they have never seen, and say "draw a fish". They nudge, look, nudge, look, for maybe 50 small steps. At the end there is a new fish doodle that was never in the data.

### How we check the drawings (evaluation)

Nobody can look at thousands of doodles by hand. So the notebook trains a second network, the judge. The judge has only ever seen real doodles. It looks at a doodle and says "cat", "dog" or "fish". If we asked for a cat and the judge says "cat", that drawing counts as recognisable.

The notebook also checks two more things:

- Variety: are the drawings different from each other, or is it the same cat every time?
- Copying: is the student just tracing doodles it saw during training?

### The two questions the notebook asks

- Q1: does more training help? The notebook compares the student after 5, 15 and 60 rounds of practice.
- Q2: how few nudges can you get away with? The notebook tries 2, 5, 10, 25, 50 and 100 steps, and compares the plain "Euler" way of stepping with the "midpoint" way.

The short answers (explained in Cells 23 to 30): more training made the student more exact at predicting the arrows, but the judge did not find the drawings clearly more recognisable. About 5 steps are enough, but 2 steps makes the drawings too similar to each other. Euler and midpoint do equally well when given the same amount of work.

---

### The core idea with just one pixel

Pretend a picture is only one pixel. In this notebook, −1 means white paper and +1 means black ink.

- The static gives this pixel the value −0.6 (fairly white).
- In the real cat doodle, this pixel is 1.0 (ink).

The straight-line fade from static to cat:

```text
moment t   value of the pixel = (1 − t) × static + t × drawing
0.00       −0.6                      <- pure static
0.25       0.75 × (−0.6) + 0.25 × 1.0 = −0.45 + 0.25 = −0.2
0.50       0.2
0.75       0.6
1.00       1.0                       <- the finished doodle
```

The right arrow (the velocity) is "how far you still have to travel over the whole trip":

```text
arrow = drawing − static = 1.0 − (−0.6) = 1.6
```

It is the same at every moment, because the path is a straight line.

If the student has learned "arrow = 1.6", then 4 small steps of size 0.25 get there exactly:

```text
start   −0.6
step 1  −0.6 + 0.25 × 1.6 = −0.2
step 2  −0.2 + 0.25 × 1.6 =  0.2
step 3   0.2 + 0.25 × 1.6 =  0.6
step 4   0.6 + 0.25 × 1.6 =  1.0   <- ink, done
```

A real picture is the same thing done for all 784 pixels at once. There is one catch. When the student starts from new static, they do not know which cat they are heading for. So they learn the average good direction from each blurry frame, and that turns out to be enough.

### A few words you will meet everywhere

- **Tensor**: PyTorch's word for a table of numbers. It can be a list, a grid, or a stack of grids. It is like a NumPy array, but it can run on the graphics card.
- **Shape**: the size of a tensor along each direction. The shape `(64, 1, 28, 28)` means 64 pictures, 1 colour channel (grey only), 28 rows and 28 columns.
- **Batch**: a handful of pictures processed together, for example 128 at a time. It is faster than one at a time.
- **Model or network**: the student. It is a big formula with about 1.8 million adjustable numbers, called weights.
- **Training**: adjusting those weights so the student's guesses get closer to the right answers.
- **Seed**: a starting number for the random-number maker. The same seed always gives the same "random" static, so results can be repeated exactly.
- **GPU** (or "cuda"): the graphics card. It does this kind of maths much faster than the normal processor (the "CPU").

There is a full glossary at the end.

---

## Cell 1 (markdown): Title, goal, data

What it says: the goal is to build a small model from scratch that turns random noise into cat, dog and fish doodles. It uses Quick, Draw! data, 4,000 doodles per animal, split 80% for learning and 20% held out. If saved models (checkpoints) already exist, the notebook loads them instead of training again.

A **checkpoint** is a saved copy of the student's brain (all the weights) at a certain point in training. It is like saving a video game, so you can continue later without starting over.

---

## Cell 2 (code, `id 5dd5461e`): Install the packages

```python
%pip install -r requirements_frozen.txt
```

What this cell does, in one sentence: it installs the exact versions of the libraries the notebook needs.

- A library is a ready-made toolbox of code written by someone else (NumPy, PyTorch, and so on).
- `requirements_frozen.txt` is the shopping list, with exact version numbers.
- If you use the project's `.venv` kernel, everything is already installed and you can skip this cell.

---

## Cell 3 (code): `# === setup and folders`

What this cell does, in one sentence: it loads the toolboxes, decides where files live, picks the GPU or CPU, and fixes the settings used everywhere else.

Analogy: setting up your desk before you start drawing. Pens out, paper in the right drawer, and the rules for the exercise written on a sticky note.

Key lines:

```python
import torch
from torch import nn
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CATEGORIES   = ["cat", "dog", "fish"]
N_PER_CAT    = 4000
TRAIN_FRAC   = 0.8
SEED_LIST = [101, 202, 303, ..., 3232]       # 32 seeds
```

In plain words:

- `import torch` loads PyTorch, the toolbox for neural networks. `nn` is its box of network building parts.
- `DEVICE = "cuda" if ... else "cpu"` means "use the graphics card if there is one, otherwise the normal processor".
- `CATEGORIES` is the list of animals. Their positions become their labels: cat = 0, dog = 1, fish = 2.
- `N_PER_CAT = 4000` doodles per animal, and `TRAIN_FRAC = 0.8` means 80% of them are for learning.
- `SEED_LIST` holds 32 fixed seeds. Each seed makes one cat, one dog and one fish, so every test uses 32 × 3 = 96 drawings. Because the same 32 seeds are used in every comparison, any difference comes from the setting being tested, not from luck. It is like testing two ovens with the same 32 cake recipes.
- A small helper, `pick_dir`, finds the folders `results/partC` (outputs), `data/quickdraw` (doodles) and `checkpoints` (saved models). If a folder does not exist yet, it creates it.
- Two `cudnn` lines tell the graphics card to give exactly the same answer every time.

Sample output (notebook output):

```text
device   : cpu
torch    : 2.3.0+cu121 | numpy 1.26.4 | python 3.10.12
32 generation seeds -> 96 outputs per configuration; margin of error at that size is +/-0.051 (worst case)
```

What the numbers mean:

- `cpu`: the computer that last ran the notebook had no graphics card, so it used the normal processor. With the project's new `.venv` on this machine, the same line now says `cuda` and `NVIDIA GeForce RTX 3060`.
- `96 outputs`: 32 seeds × 3 animals.
- `+/-0.051`: with only 96 drawings, a score can wobble by about 5 percentage points just by chance. It comes from `sqrt(0.25 / 96) = 0.051`. Strictly, this is one standard error, not a full 95% margin of error. See the issues list at the end.

---

## Cell 4 (markdown): `## 1. Method and research questions`

What it says: this is the method written as formulas, which is the same story as "The whole project in plain words" above. Here are the four formulas, translated:

| Formula in the notebook | In plain words |
|---|---|
| `x_t = (1−t)·x0 + t·x1` | The film frame at moment `t`: a mix of static `x0` and drawing `x1` |
| `u_t = x1 − x0` | The right arrow: drawing minus static. It is the same at every moment |
| loss = average of `(guess − (x1 − x0))²` | How wrong the student's arrows are, on average |
| `x ← x + Δt · v` | One small nudge: move a little (`Δt`) along the student's arrow `v` |

It also states the two research questions (Q1: training length, Q2: number of steps and stepping method), with the author's predictions written down before running anything. That is good scientific practice.

---

## Cell 5 (markdown): `## 2. Data: download, inspect, filter and split`

What it says: the data preparation plan, in four steps.

1. Pick a fixed random set of doodles.
2. Throw away extreme ones: almost blank pages, or pages scribbled almost solid black.
3. Change the pixel scale from 0…255 to −1…+1.
4. Split 80/20 into learning and held-out sets before doing anything else.

The only "augmentation" is a random left-right mirror flip, and only on the learning set. Augmentation means making extra variety for free. A cat facing left is still a cat.

---

## Cell 6 (code): `# === download the dataset`

What this cell does, in one sentence: it downloads the cat, dog and fish doodle files (if they are not already on disk) and records a fingerprint of each one.

Analogy: picking up three boxes of drawings from the library, then writing down each box's serial number so anyone can check they have the same boxes.

Key lines:

```python
if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
    return path                             # already have it, skip the download
urllib.request.urlretrieve(url, tmp)        # download to "cat.npy.part"
os.replace(tmp, path)                       # then rename to "cat.npy"
PATHS = {c: download_category(c) for c in CATEGORIES}
```

In plain words:

- If the file is already there and bigger than 1 MB, it is kept. The size check catches a half-finished download.
- Otherwise it downloads into a temporary name first, and renames it only when the download is complete. That way you never get a broken file that looks finished.
- `sha256_of` computes a fingerprint (a "SHA-256 hash"). It is a long code that changes completely if even one byte of the file changes.
- `PATHS` ends up as a dictionary (a lookup list): `{"cat": ".../cat.npy", "dog": ..., "fish": ...}`.

Sample output (notebook output):

```text
  cat.npy already present (97 MB)
  dog.npy already present (119 MB)
  fish.npy already present (105 MB)
  cat   21a281839d3f2eef601d57d2338a4eafdf24649f8d0a0e42d3ec3e595911463e
```

What the numbers mean: the files were already downloaded, at 97, 119 and 105 megabytes. The long code after `cat` is the fingerprint of the cat file.

What is inside one file (a rerun of the code):

```text
>>> arr = np.load(PATHS["cat"], mmap_mode="r")
>>> arr.shape
(123202, 784)
```

This means 123,202 cat doodles. Each is a row of 784 numbers (28 × 28 pixels laid out in one long line), each from 0 (white) to 255 (ink). `mmap_mode="r"` means "do not load the whole file into memory, just read the rows I ask for". It is like reading only the pages you need from a big book instead of photocopying the whole thing.

---

## Cell 7 (code): `# === inspect ink distribution`

What this cell does, in one sentence: before deciding which doodles count as "too blank" or "too scribbly", it measures how much ink normal doodles have.

Analogy: before setting a "too short / too tall" rule for a ride, you first measure a lot of people to see what normal heights look like.

Key lines:

```python
idx = np.sort(np.random.default_rng(1).choice(arr.shape[0], 20_000, replace=False))
frac = (np.asarray(arr[idx]) > 0).mean(axis=1)
```

In plain words:

- Pick 20,000 random doodles of each animal. `replace=False` means no doodle is picked twice.
- `arr[idx] > 0` marks every pixel that has any ink as True.
- `.mean(axis=1)` averages along each row, which gives one number per doodle: the fraction of its 784 pixels that have ink.
- A percentile is a cut-off point. The "1st percentile" (p1) is the value that only 1% of doodles fall below. The median is the middle value: half the doodles are below it and half above.

Sample output (notebook output):

```text
category  drawings in file  ink p1  ink median  ink p99  below 0.05  above 0.55
     cat            123202  0.1798      0.3342   0.4745           0           5
     dog            152159  0.1684      0.2934   0.4796           0          16
    fish            134150  0.1620      0.2449   0.3712           3           3
```

What the numbers mean:

- `ink median 0.3342` for cats: a typical cat doodle has some ink on about a third of its pixels.
- `ink p1 0.1798`: only 1% of cats have less than 18% ink.
- `below 0.05` / `above 0.55`: how many of the 20,000 doodles are almost blank (under 5% ink) or almost solid (over 55% ink). For cats, that is 0 and 5. So the filter `0.05 to 0.55` used in Cell 8 throws away only a handful of extreme doodles.

---

## Cell 8 (code): `# === build the dataset`

What this cell does, in one sentence: it picks 4,000 doodles per animal, removes the extreme ones, rescales the pixels to −1…+1, and splits them into a learning set and a held-out set.

Analogy: sorting a pile of drawings into two folders. The big folder is for practice. The small folder is locked away for the final exam, so the student cannot have seen the exam drawings before.

Key lines, in order:

```python
oversample = min(total, int(n_per_cat * 1.6) + 500)          # pick 6,900 candidates
keep = (frac >= INK_MIN) & (frac <= INK_MAX)                 # True = passes the ink rule
kept = cand[keep][:n_per_cat]                                # keep the first 4,000 that pass
perm = rng.permutation(len(kept))                            # shuffle 0..3999
tr_sel, ev_sel = perm[:n_train], perm[n_train:]              # first 3,200 / last 800
imgs = kept.reshape(-1, 1, IMG, IMG).astype(np.float32) / 255.0
imgs = imgs * 2.0 - 1.0                                      # 0..1 -> -1..+1
```

In plain words:

- It picks a few more doodles than needed (4,000 × 1.6 + 500 = 6,900), so there are still 4,000 left after the filter.
- `keep` is a True/False list: True for doodles that pass the ink rule. `cand[keep]` keeps only those rows. This is the same "mask" idea as in Part A.
- `permutation` shuffles the numbers 0 to 3,999, like shuffling a deck of cards. The first 3,200 cards go to learning and the last 800 to the exam.
- `reshape(-1, 1, 28, 28)` turns each long line of 784 numbers back into a little picture: 1 colour channel, 28 rows, 28 columns. The `-1` means "work out how many pictures there are yourself".
- The two rescaling lines turn 0 (white) into −1, mid grey into about 0, and 255 (ink) into +1. This puts the doodles on the same scale as the static, which is centred around 0.

A tiny example of the shuffle (seed 0, 6 items, 5 for learning):

```text
>>> perm = np.random.default_rng(0).permutation(6)
[3 2 5 4 0 1]
>>> perm[:5], perm[5:]
([3 2 5 4 0], [1])           # items 3,2,5,4,0 for learning, item 1 held out
```

Sample output (rerun of the cell):

```text
train_x (9600, 1, 28, 28) float32
train_y (9600,) int64
eval_x (2400, 1, 28, 28) float32
>>> np.bincount(DS["train_y"])
[3200 3200 3200]
```

What the numbers mean:

- `train_x (9600, 1, 28, 28)`: 9,600 learning pictures, 1 colour channel, 28 rows, 28 columns. `float32` means numbers with decimals.
- `train_y (9600,)`: 9,600 labels, one per picture (0, 1 or 2).
- `eval_x (2400, ...)`: the 2,400 held-out exam pictures.
- `[3200 3200 3200]`: exactly 3,200 cats, 3,200 dogs and 3,200 fish.

The labels are in order: rows 0 to 3,199 are cats, then dogs, then fish. This matters in Cell 13.

---

## Cell 9 (code): `# === look at the real data first`

What this cell does, in one sentence: it shows a grid of 36 real doodles (12 cats, 12 dogs, 12 fish) and measures how much ink each animal usually has.

Analogy: before teaching the student, you flip through the practice folder to see what the drawings actually look like.

Key lines:

```python
pool = DS["train_x"][DS["train_y"] == r]                  # only the pictures of animal r
axes[r, k].imshow(pool[j, 0], cmap="gray_r", vmin=-1, vmax=1)
ink = (DS["train_x"][DS["train_y"] == i] > 0).reshape(-1, IMG*IMG).mean(axis=1)
```

In plain words:

- `DS["train_y"] == r` is a True/False mask that keeps only one animal.
- `imshow` draws the picture. `gray_r` means −1 shows as white and +1 as black, like pen on paper.
- The ink line counts, for each doodle, the share of pixels darker than mid grey (above 0 after rescaling).

Sample output (notebook output):

```text
  cat   mean 0.1984  sd 0.0389
  dog   mean 0.1808  sd 0.0420
  fish  mean 0.1402  sd 0.0258
```

What the numbers mean:

- `mean 0.1984`: on average, about 20% of a cat doodle's pixels are dark ink. Fish use the least ink (14%).
- `sd 0.0389`: the standard deviation, which is how spread out the values are. Most cats are within about 4 percentage points of 20%.

These are used later as a comparison: generated cats should have roughly 20% ink too.

Why is this 0.198 for cats when Cell 7 said 0.334? The two cells use different rules. Cell 7 counts any trace of grey. Cell 9 counts only pixels darker than mid grey. Same name, different ruler (see the issues list).

---

## Cell 10 (markdown): `## 3. The model, the training loop, and the three key code sections`

What it says: a description of the student's brain, the U-Net.

- It shrinks the picture (28 → 14 → 7 pixels wide) to see the big shape, then grows it back (7 → 14 → 28) to put the detail back.
- "Skip connections" carry the fine detail across from the shrinking side to the growing side.
- The moment `t` and the animal label are turned into lists of numbers and fed into every part of the network.
- The last layer starts at all zeros, so a brand-new student predicts "don't move anything".
- There are 4 label slots, not 3. The 4th "null" slot is never used.

---

## Cell 11 (code): `# === the network`

What this cell does, in one sentence: it writes down the design of the student's brain. Nothing runs yet and nothing prints; this is only the blueprint.

Analogy: a recipe card. It lists the ingredients and the steps, but no cooking happens until someone uses it.

The blueprint has three parts: a clock reader (`SinusoidalTimeEmbedding`), a repeating building block (`ResBlock`), and the whole brain (`VelocityUNet`).

### How a network is written in PyTorch

```python
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, c_emb):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
    def forward(self, x, emb):
        ...
```

- `class ResBlock(nn.Module)` says "this is a network part". `nn.Module` gives it saving, loading and moving to the GPU for free.
- `__init__` lists the parts that have adjustable numbers (the weights). `super().__init__()` must come first; it sets up the PyTorch machinery.
- `forward` says what to do with an input. You never call `forward` yourself. Writing `block(x, emb)` calls it for you.

### Part 1: the clock reader (time embedding)

The student needs to know how far along the film it is. At `t = 0.1` the frame is almost all static, so big bold moves make sense. At `t = 0.9` it is almost finished, so only small touch-ups are needed. A single number like 0.3 is hard for a network to use, so it gets turned into a list of 128 numbers. That list is called an **embedding**: a way of writing one thing as a list of numbers that is easier for a network to read.

Analogy: a clock with many hands. The hour hand alone tells you roughly the time. Add a minute hand and a second hand and you know it precisely. Each pair of numbers in the embedding (a sine and a cosine, which are just smooth wave shapes) is like a hand spinning at a different speed.

A tiny example with 8 numbers instead of 128 (rerun of the code):

```text
>>> SinusoidalTimeEmbedding(8)(torch.tensor([0.0, 0.5, 1.0]))
t=0.0: [ 0.000,  0.000, 0.000, 0.000,  1.000,  1.000,  1.000, 1.000]
t=0.5: [-0.468, -0.938, 0.881, 0.050, -0.884, -0.347,  0.474, 0.999]
t=1.0: [ 0.827,  0.650, 0.834, 0.100,  0.562, -0.760, -0.551, 0.995]
```

What the numbers mean: each moment gets its own pattern of 8 numbers between −1 and +1. At `t = 0` every wave starts at its beginning, so the sines are 0 and the cosines are 1. With the real 128 numbers, the moments 0.30 and 0.31 end up clearly different (a distance of 6.46 apart), so the student can tell even very close moments apart.

### Part 2: the building block (ResBlock)

```python
h = self.conv1(F.silu(self.norm1(x)))
h = h + self.emb(F.silu(emb))[:, :, None, None]     # add the "time + animal" message
h = self.conv2(F.silu(self.norm2(h)))
return h + self.skip(x)
```

The parts used here:

| Part | What it does, in plain words |
|---|---|
| `nn.Conv2d` (a convolution) | Slides a small 3 by 3 window over the picture, like a magnifying glass, and looks for small patterns such as edges and corners |
| `nn.GroupNorm` | Keeps the numbers in a steady range so learning does not go wild |
| `F.silu` | An "activation": a gentle bend in the maths. Without bends, stacking many layers would be no better than one |
| `nn.Linear` | Mixes a list of numbers into a new list of numbers, using adjustable weights |

The two important lines:

- The `self.emb(...)` line adds a "message" to every pixel: "it is moment t, and you are drawing a cat". Every pixel of a channel gets the same nudge.
- `return h + self.skip(x)` is the "residual" idea. The block does not redraw the picture from scratch. It only learns a correction to add on top of what came in. This makes deep networks much easier to train, like editing a draft instead of rewriting the whole essay.

### Part 3: the whole brain (the U-Net)

```python
emb = self.time_embed(t) + self.class_embed(y)      # "what moment" + "which animal"
h1 = self.down1(h0, emb)                            # 28 x 28, detailed
h2 = self.down2(self.pool1(h1), emb)                # shrink to 14 x 14
m  = self.mid2(self.mid1(self.pool2(h2), emb), emb) # shrink to 7 x 7, the big picture
d2 = self.dec2(torch.cat([self.up2(m), h2], dim=1), emb)   # grow to 14, add back h2's detail
d1 = self.dec1(torch.cat([self.up1(d2), h1], dim=1), emb)  # grow to 28, add back h1's detail
```

Analogy: an artist who first squints at the whole page to see the overall shape ("there's a round head here, a tail there"), then leans in close to fix the fine lines. Shrinking the picture is the squint. Growing it back is leaning in. The skip connections (`torch.cat([..., h2])`) are like keeping the original sketch next to you while you work, so the fine lines are not lost.

- `self.class_embed(y)` looks up the animal in a small table with 4 rows (cat, dog, fish and the unused "null"), each 128 numbers long. This is how "draw a cat" reaches the network. It is called class conditioning.
- That animal list is added to the clock list, and the sum (`emb`) is passed into every block.

Sample output: the picture's shape at every stage, for 2 real doodles (a fresh, untrained network, seed 0):

```text
x              (2, 1, 28, 28)     2 pictures, 1 channel, 28 x 28
stem h0        (2, 64, 28, 28)    1 channel becomes 64 "feature maps"
pool1          (2, 64, 14, 14)    shrink to half size
down2 h2       (2, 128, 14, 14)   more feature maps
pool2          (2, 128, 7, 7)     shrink again: the bottom of the "U"
up2            (2, 128, 14, 14)   grow back
cat[up2,h2]    (2, 256, 14, 14)   skip connection: 128 + 128 stacked together
up1            (2, 64, 28, 28)    grow back to full size
out            (2, 1, 28, 28)     same shape as the input: one arrow per pixel
```

What the numbers mean:

- A "feature map" is one layer of the brain's notes about the picture. One map might light up where there are edges, another where there are curves. 64 maps means 64 different kinds of notes.
- The sizes go 28 → 14 → 7 → 14 → 28, down and back up like the letter U. That is where the name U-Net comes from.
- The output has the same shape as the input because it is an arrow (a velocity) for every pixel.

Two more lines:

```python
nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)
total = sum(p.numel() for p in model.parameters())
```

- The first line sets the last layer to all zeros, so a brand-new student says "move nothing" everywhere. That is a calm, safe starting point.
- The second line counts the adjustable numbers. The answer is 1,838,081, all learned from scratch.

---

## Cell 12 (code): `# === the three key pieces`

What this cell does, in one sentence: it defines two of the three most important functions. One makes practice questions for the student. The other lets the student draw by following its own arrows. The third piece, the correction step, is in Cell 15.

### (a) `flow_matching_batch`: making practice questions

Analogy: a teacher making a worksheet. They take a real drawing, smudge it with some static, and write the answer key: "the way back to the real drawing is this arrow".

```python
x0 = torch.randn(x1.shape, ...)        # fresh static, same size as the drawings
t  = torch.rand(x1.shape[0], ...)      # one random moment (0 to 1) per picture
t_b = t.view(-1, 1, 1, 1)              # the same moment for every pixel of that picture
return (1.0 - t_b) * x0 + t_b * x1, t, (x1 - x0)
```

In plain words:

- `torch.randn` makes static: random numbers that are mostly near 0, sometimes bigger (a bell curve).
- `torch.rand` makes a random moment between 0 and 1 for each picture. Note `randn` (bell curve) versus `rand` (any value from 0 to 1 equally likely).
- `t.view(-1, 1, 1, 1)` changes the moments' shape so that PyTorch applies one picture's moment to all of its 784 pixels.
- It gives back three things: the smudged frame, the moments, and the answer key (`x1 − x0`).

Sample output: a tiny 4-pixel "picture" (seed 0):

```text
x1      [ 1.0000, -1.0000, -1.0000,  1.0000]   the real drawing
x0      [ 1.5410, -0.2934, -2.1788,  0.5684]   random static
t       [0.4556]                               the moment: 45.56% of the way
x_t     [ 1.2945, -0.6154, -1.6417,  0.7651]   the smudged frame
target  [-0.5410, -0.7066,  1.1788,  0.4316]   the answer key: x1 − x0
```

Checking pixel 1 by hand:

- The frame is `(1 − 0.4556) × 1.5410 + 0.4556 × 1.0 = 0.8389 + 0.4556 = 1.2945`.
- The answer is `1.0 − 1.5410 = −0.5410`, meaning "go down by 0.54 over the whole trip".

The nice part: the answer key is just a subtraction. No complicated simulation is needed to know the right answer.

### (c) `sample_ode`: drawing by following the arrows

Analogy: the student at work on new static. Look at the picture, decide the arrows, nudge a little, and repeat.

```python
x = torch.randn((n, *img_shape), device=device, generator=gen)   # start: fresh static
ts = torch.linspace(0.0, 1.0, n_steps + 1)                       # the moments to stop at
for i in range(n_steps):
    dt = ts[i + 1] - ts[i]                     # size of one step
    if method == "euler":
        v = model(tb, x, labels)               # ask the student for the arrows
        x = x + dt * v                         # nudge a little along them
```

In plain words:

- It starts from fresh static. With the same seed, you always get the same static, so the same drawing.
- `torch.linspace(0, 1, 6)` gives evenly spaced moments: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` for 5 steps.
- Each step asks the student for arrows (`model(...)`) and moves a small amount (`dt`) along them.
- `@torch.no_grad()` above the function means "we are only using the student, not teaching it", which saves memory and time.
- NFE (number of function evaluations) counts how many times the student was asked. It is the real cost of drawing.

---

### Euler versus midpoint

Analogy: walking through fog with a compass.

- **Euler**: read the compass, walk one step that way, then read it again. That is one reading per step.
- **Midpoint**: read the compass, walk half a step, read it again there, then go back to where you started and take the full step using the second reading. That is two readings per step. It handles bendy paths better.

If the path is perfectly straight, both arrive at the same place. Because midpoint costs two readings per step, the fair comparison is at the same number of readings (same NFE), not the same number of steps.

One real pixel being drawn (cat, row 14, column 14, the trained 60-round student, seed 101, 10 Euler steps):

```text
step 1:  t=0.0  x=1.270  arrow=-1.673  -> 1.270 + 0.1 × (-1.673) = 1.103
step 2:  t=0.1  x=1.103  arrow=-1.586  -> 0.944
step 3:  t=0.2  x=0.944  arrow=-1.442  -> 0.800
...
step 10: t=0.9  x=0.496  arrow= 0.294  -> 0.526      ends above 0, so it is ink
```

What the numbers mean: this pixel started too dark (1.27), and the student kept pulling it lighter. Near the end it decided the pixel should be ink after all, and pushed it slightly back up. The final 0.526 is above 0, so in the finished drawing this pixel is a pen stroke.

A full call: 3 pictures (cat, dog, fish), seed 101, 50 Euler steps:

```text
>>> im, nfe = sample_ode(model60, 3, torch.arange(3), n_steps=50, seed=101)
>>> im.shape, nfe
(torch.Size([3, 1, 28, 28]), 50)
>>> (im > 0).float().mean((1, 2, 3))       # ink share per picture
tensor([0.1862, 0.1926, 0.1314])
```

What the numbers mean:

- 3 pictures of 28 × 28, made with 50 questions to the student per picture.
- The ink shares, 18.6%, 19.3% and 13.1%, are close to the real doodles from Cell 9 (19.8%, 18.1% and 14.0%). So the student uses about the right amount of ink.

---

## Cell 13 (code): `# === shape and gradient checks`

What this cell does, in one sentence: it does a quick safety check that the shapes fit together and that the "correction signal" can reach every part of the brain.

Analogy: before a long car trip, you check the tyres fit and the steering wheel actually turns the wheels.

Key lines:

```python
pred = probe(t_s, x_t, yb)                      # a fresh student guesses arrows for 8 pictures
assert pred.shape == xb.shape == target.shape   # guess, picture and answer must be the same size
loss = torch.mean((pred - target) ** 2); loss.backward()
```

- `assert` means "stop with an error if this is not true".
- The **loss** is one number saying how wrong the guesses are. Here it is the mean squared error: take each difference between the guess and the answer, square it (so negatives become positive), and average.
- `loss.backward()` works out the **gradient** for each of the 1.8 million weights. The gradient answers: "if I turn this weight up a tiny bit, does the loss go up or down, and by how much?"

Analogy for gradients: you are blindfolded on a hill and want to reach the bottom of the valley. You cannot see, but you can feel which way the ground slopes under your feet. The gradient is that slope. Training means taking a small step downhill, again and again.

Sample output (notebook output):

```text
x1 (8, 1, 28, 28) -> x_t (8, 1, 28, 28)  t (8,)  target (8, 1, 28, 28)  pred (8, 1, 28, 28)
weights receiving a non-zero gradient: 2 of 85
class embedding gets a gradient: False
```

What the numbers mean:

- The first line: 8 pictures go in and 8 arrow-pictures come out, all the same shape. The shape check passes.
- `2 of 85`: the brain has 85 groups of weights, and only 2 of them got a correction signal.
- `False`: the animal-label table got no correction signal.

This looks like a failure, but the trained student is fine. The check itself is flawed, for two reasons:

1. The last layer starts at all zeros (Cell 11). The correction signal has to travel backwards through that last layer to reach everything else, and anything times zero is zero. So only the last layer itself (the "2 of 85") gets a signal on a brand-new student. It is like a megaphone with the volume at zero: nobody behind it hears anything yet.
2. The 8 test pictures are the first 8 rows, and they are all cats (the data is sorted by animal, see Cell 8). So dogs and fish are never tested.

When the same check is run on the trained 60-round student, all 85 groups get a signal, including the animal table. See the issues list.

---

## Cell 14 (markdown): `### The three important code sections, in my own words`

What it says: a summary of the three key pieces in the author's own words. It matches the code:

- (a) `flow_matching_batch` makes practice questions. There is one moment per picture, not per pixel, and the answer is just a subtraction.
- (b) The training step (inside `train_model`, Cell 15). The whole method is one line: the loss.
- (c) `sample_ode` draws by following the arrows, and reports the cost (NFE) so that methods can be compared fairly.

---

## Cell 15 (code): `# === training loop`

What this cell does, in one sentence: it defines how the student practises. It hands out practice questions in batches, checks the guesses, corrects the brain a little, and saves progress.

Analogy: a practice session with flashcards. The teacher shuffles the deck, hands over 128 cards at a time, marks the answers, and the student adjusts. Going through the whole deck once is one **epoch** (one round). The notebook does 60 rounds.

### Serving the cards (`make_loaders`)

```python
DataLoader(tr, batch_size=128, shuffle=True, drop_last=True, ...)
```

- `DataLoader` deals the pictures out in batches of 128, shuffled differently every round.
- `drop_last=True` throws away the last small leftover batch, so every batch has exactly 128.

Sample output (rerun):

```text
>>> len(tr_loader), len(ev_loader)
(75, 10)
>>> x1, y = next(iter(tr_loader))
>>> x1.shape, y[:10]
(torch.Size([128, 1, 28, 28]), tensor([0, 1, 2, 1, 0, 1, 1, 0, 0, 2]))
```

What the numbers mean:

- `75`: 9,600 ÷ 128 = 75 batches per round. So 60 rounds is 60 × 75 = 4,500 practice steps.
- `10`: the 2,400 exam pictures are served 256 at a time, which is 9.4 batches, rounded up to 10 (the last one is smaller).
- `(128, 1, 28, 28)`: one batch is 128 pictures, 1 channel, 28 by 28.
- `[0, 1, 2, 1, ...]`: the animals are mixed now, because the deck is shuffled.

### A fair exam score (`eval_loss`)

During practice, the moments are random. For the exam score, every held-out picture is tested at the same 8 moments (0.02, 0.157, ..., 0.98) with the same seeded static. Every saved checkpoint therefore sits exactly the same exam, so their scores can be compared fairly.

### The practice step itself (`train_model`)

```python
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(tr_loader))
for x1, y in tr_loader:
    x1 = torch.where(flip.view(-1, 1, 1, 1), x1.flip(-1), x1)   # mirror about half the pictures
    x_t, t, target = flow_matching_batch(x1, generator=gen)    # make practice questions
    loss = torch.mean((model(t, x_t, y) - target) ** 2)       # how wrong are the arrows?
    opt.zero_grad(set_to_none=True)                           # 1. wipe old corrections
    loss.backward()                                           # 2. work out new corrections
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 3. cap them so none is huge
    opt.step(); sched.step()                                  # 4. apply them, shrink step size
```

In plain words:

- The **optimizer** (`AdamW`) is the rule that turns gradients into actual changes to the weights.
- The **learning rate** (`lr = 0.0002`) is how big each change is. It is like the size of your steps walking downhill: too big and you overshoot the valley, too small and you take forever.
- `CosineAnnealingLR` slowly shrinks the learning rate from 0.0002 down to 0 over the 4,500 steps. Big steps early, careful little steps at the end, like slowing down as you approach a parking space.
- The flip line mirrors about half of the pictures left-to-right, for free extra variety.
- The four numbered lines are the heart of every PyTorch training loop: clear the old corrections, compute new ones, cap them, apply them.

Sample output (a tiny illustration, not the real training): 3 practice steps on the same 8 pictures with a fresh student, seed 0.

```text
step 1 loss 1.8003   size of corrections before the cap 10.33
step 2 loss 1.7664   size of corrections before the cap 9.83
step 3 loss 1.7247   size of corrections before the cap 10.91
loss of guessing zero everywhere = 1.8003
```

What the numbers mean:

- `step 1 loss 1.8003` is exactly the same as "guess zero everywhere". That makes sense, because the brand-new student's last layer is all zeros.
- The loss goes down a little each step (1.80 → 1.77 → 1.72). The student is learning.
- The corrections are about 10 in size, and the cap is 1, so the cap really is shrinking every step here.

The real training did this 4,500 times. Its first logged step had a loss of 1.185, and the last round averaged 0.292.

### Saving and loading the brain (checkpoints)

```python
torch.save({"model": model.state_dict(), "epoch": ep, ...}, ck)    # save
model.load_state_dict(ck["model"]); model.eval()                  # load
```

- `state_dict()` is a list of every weight group by name. `torch.save` writes it to a `.pt` file. It is the video-game save.
- To load, the code builds an empty brain with the same design, then pours the saved weights into it.

Sample output (the real saved file `ckpt_epoch060.pt`):

```text
>>> ck["epoch"], ck["train_loss"], ck["eval_loss"]
(60, 0.291679234902064, 0.29217179692039885)
>>> len(ck["model"])
85
```

What the numbers mean: this save is from after round 60. The practice loss was 0.2917 and the exam loss was 0.2922. They are almost equal, which suggests the student is not just memorising. The brain has 85 named weight groups.

---

## Cell 16 (code): `# === baseline: load or train`

What this cell does, in one sentence: if the saved checkpoints already exist it loads them; otherwise it trains the student for 60 rounds (about 8 minutes on the RTX 3060).

Analogy: "Do I already have my saved game? Then load it. If not, play from the start."

```python
NEEDED = [os.path.join(BASE_DIR, f"ckpt_epoch{e:03d}.pt") for e in (5, 15, 60)]
if os.path.exists(BASE_LOG_PATH) and all(os.path.exists(p) for p in NEEDED):
    base_log = json.load(open(BASE_LOG_PATH))         # load the saved training diary
else:
    _, base_log = train_model(DS, BASE_DIR, **BASE_CFG)   # train from scratch
```

- `f"{e:03d}"` writes a number with 3 digits, so 5 becomes `005`, which gives `ckpt_epoch005.pt`.
- `all(...)` is True only if all three saves (rounds 5, 15 and 60) exist.

Sample output (notebook output, shortened):

```text
baseline checkpoints found -> loading, not retraining.
  params_total        : 1838081
  gpu                 : NVIDIA GeForce RTX 3060
  total train time    : 8.10 min
  final train loss    : 0.29168
  final eval loss     : 0.29217
```

What the numbers mean: the saves were found, so nothing was retrained. These numbers come from the saved training diary (`training_log.json`). The original training of 1,838,081 weights took 8.1 minutes on the RTX 3060, and ended with a practice loss of 0.292 and an exam loss of 0.292.

---

## Cell 17 (code): `# === training curves`

What this cell does, in one sentence: it draws three charts from the saved training diary, showing how the loss fell and whether the student started memorising.

Analogy: plotting your weekly practice-test scores to see if you are improving, and comparing practice scores with surprise-quiz scores.

The three charts:

1. The practice loss over the 4,500 steps.
2. Practice loss against exam loss, round by round. Dotted lines mark rounds 5, 15 and 60, where saves were made.
3. Exam loss minus practice loss. This is the **overfitting** check. Overfitting means memorising the practice cards instead of learning the skill. If that happened, the exam loss would climb well above the practice loss.

Sample output (notebook output):

```text
held-out loss never exceeds train loss by more than 0.0070 -> no sign of overfitting at this size.
```

What the number means: in every round, the exam loss was at most 0.0070 higher than the practice loss. That is a tiny gap, so the student is not memorising.

The cell also warns, correctly, that this loss measures how good the arrows are, not how nice the drawings look. A lower loss does not automatically mean better pictures (Cell 23 shows this).

---

## Cell 18 (code): `# === noise -> drawing trajectory`

What this cell does, in one sentence: it draws 2 cats, 2 dogs and 2 fish and takes snapshots along the way, so you can watch static turn into a drawing.

Analogy: a time-lapse video of the student drawing, from the first blob of static to the finished doodle.

```python
labels_show = torch.tensor([0, 0, 1, 1, 2, 2])
imgs, nfe, traj = sample_ode(model60, n_show, labels_show, n_steps=50, method="euler",
                             seed=SEED_LIST[0], return_trajectory=True, traj_every=6)
```

- `[0, 0, 1, 1, 2, 2]` asks for two of each animal.
- `return_trajectory=True, traj_every=6` means "also give me a snapshot every 6 steps".

Sample output (rerun):

```text
>>> len(traj), traj[0].shape
(10, torch.Size([6, 1, 28, 28]))
>>> ts_lbl
[0.0, 0.02, 0.14, 0.26, 0.38, 0.5, 0.62, 0.74, 0.86, 0.98, 1.0]
```

What the numbers mean:

- There are 10 snapshots, each holding the 6 pictures at that moment.
- `ts_lbl` is the column titles: the starting static (0.0), then 2%, 14%, 26% and so on, up to the finished drawing (1.0). That makes 11 columns in the figure.
- The notebook also prints `NFE per image for this path: 50`, meaning the student was asked 50 times per picture.

---

## Cell 19 (markdown): `## 4–6. Controlled comparisons and scoring`

What it says: the rules for a fair test.

- The same 32 seeds are used everywhere, so every setting is tested on the same 96 starting statics.
- Only one thing changes at a time.
- "Recognisability" is the share of generated drawings that the judge labels as the animal that was asked for. The judge was trained only on real held-out drawings.

---

## Cell 20 (code): `# === the scoring CNN`

What this cell does, in one sentence: it builds and trains (or loads) the judge, a small separate network that looks at a doodle and says "cat", "dog" or "fish".

Analogy: a judge at an art contest. The judge learned what cats, dogs and fish look like only from real doodles, and has never met the art student. So the judge cannot be biased towards the student's style.

```python
self.body = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 28 -> 14
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 14 -> 7
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))
```

In plain words:

- This is a **CNN** (convolutional neural network): a network made mostly of the sliding "magnifying glass" layers from Cell 11. It is the standard kind of network for looking at pictures.
- `nn.Sequential` chains the layers in order, like a conveyor belt.
- `ReLU` is a simple bend: negative numbers become 0.
- `MaxPool2d(2)` shrinks the picture by keeping the strongest value in each 2 by 2 square.
- `Dropout(0.3)` randomly ignores 30% of the numbers during the judge's training. This forces it not to rely on any one clue.
- `Linear(128, 3)` gives 3 final scores: one for cat, one for dog, one for fish.

### How the judge gives an answer

Sample output (the saved judge on 3 real exam doodles: a cat, a dog and a fish):

```text
scores        cat      dog     fish
picture 1   0.5639  -0.0074  -3.3199
picture 2  -0.3640  -0.4744  -1.0387
picture 3  -2.0000  -1.3949   2.4342
as %        cat      dog     fish
picture 1   63.1%    35.6%    1.3%
picture 2   41.6%    37.2%   21.2%
picture 3    1.2%     2.1%   96.8%
answer      [cat, cat, fish]
```

What the numbers mean:

- The raw scores (called **logits**) can be any number. Higher means "more likely this animal".
- **Softmax** turns them into percentages that add up to 100% per picture. For picture 3, the judge is 96.8% sure it is a fish.
- **Argmax** picks the biggest one, which is the judge's answer.
- Picture 2 was really a dog, but the judge said cat (41.6% vs 37.2%). The judge is not perfect, which is why its "ceiling" below is not 100%.

### How good is the judge?

The 2,400 exam doodles are split again: 1,920 to train the judge and 480 to test it.

Sample output (notebook output):

```text
scoring CNN accuracy on REAL held-out drawings: 0.8271
per-class ceiling on real drawings: {'cat': 0.8397, 'dog': 0.7396, 'fish': 0.9097}
```

What the numbers mean:

- `0.8271`: on real doodles the judge gets 82.7% right. This is the **ceiling**. Even perfect fake doodles could not be expected to score higher, because the judge makes mistakes on real ones too.
- The judge is best at fish (91%) and worst at dogs (74%). Dogs probably get confused with cats.

---

## Cell 21 (code): `# === measuring helpers`

What this cell does, in one sentence: it defines the measuring tools used in every comparison: recognisability, variety, closeness to the training doodles, and the error bars.

Analogy: the scorecard every contest entry gets.

The tools:

- **`recognisability`** asks the judge about each generated doodle. It returns the share where the judge named the animal that was asked for, plus the judge's average confidence in that animal.
- **`variety`** measures how different the doodles in a set are from each other, as the average distance between every pair. If the student drew the same cat every time, it would be 0.
- **`nearest_train_distance`** finds, for each generated doodle, the most similar of the 9,600 practice doodles. This is used to check for copying (Cell 28).
- **`generate_config`** draws the standard 96 doodles: 32 seeds, each giving a cat, a dog and a fish.
- **`score_config`** puts all the measurements for one setting into one row of a table.

What "distance" between two pictures means: line up the 784 pixels of each picture. For each pixel, take the difference, square it, then add all 784 squares and take the square root. This is the ordinary "ruler" distance, just in 784 directions instead of 2. For scale, an all-white picture (every pixel −1) and an all-black one (every pixel +1) differ by 2 at every pixel, so they are `sqrt(784 × 2²) = 56` apart.

Sample output: the 3 generated doodles from Cell 12 (seed 101):

```text
distance     cat    dog    fish
cat         0.00  23.48  22.65
dog        23.48   0.00  22.20
fish       22.65  22.20   0.00
variety = (23.48 + 22.65 + 22.20) / 3 = 22.78
closest practice doodle for each: [15.0, 18.6, 12.9]
```

What the numbers mean:

- Each picture is 0 away from itself (the diagonal). Each pair is counted once.
- The variety of 22.78 is their average distance apart.
- The closest practice doodles are 15.0, 18.6 and 12.9 away. That is not near zero, so none of the three is a copy.

### The error bar (standard error)

With only 96 doodles, a score wobbles by chance. The **standard error** measures how much:

```text
standard error = sqrt(score × (1 − score) / 96)
for a score of 0.7708: sqrt(0.7708 × 0.2292 / 96) = 0.0429
95% range: 0.7708 ± 1.96 × 0.0429 = 0.687 to 0.855
```

Analogy: if you flip a coin 96 times, you will not get exactly 48 heads. The standard error is how far from 48 you would typically land. The "95% range" is where the true score very likely sits.

---

## Cell 22 (code): `# === reproducibility check`

What this cell does, in one sentence: it checks that the same seed always gives exactly the same drawings, and that a different seed gives different ones.

Analogy: baking the same recipe twice with the same ingredients should give the same cake.

```python
a, _ = sample_ode(model, 3, y, n_steps=50, method=method, seed=seed)
b, _ = sample_ode(model, 3, y, n_steps=50, method=method, seed=seed)
c, _ = sample_ode(model, 3, y, n_steps=50, method=method, seed=seed + 1)
```

Sample output: running `a` and `b` (the same seed, 101) gives identical pictures, with a largest difference of 0.0. Running `c` (seed 102) gives different pictures, with a largest pixel difference of 2.16. The notebook prints `True` / `True`, which means "same seed gives the same result: yes" and "different seed gives a different result: yes".

This shows that the only randomness is the starting static. Following the arrows is a fixed recipe.

---

## Cell 23 (code): `# === COMPARISON 1 (training factor): checkpoint epoch`

What this cell does, in one sentence: it answers Q1 ("does more practice help?") by having the student draw the same 96 doodles after 5, 15 and 60 rounds, and scoring each set.

Analogy: testing the student with the same exam after 5 weeks, 15 weeks and 60 weeks of lessons.

```python
for epn in CKPT_EPOCHS:                                   # [5, 15, 60]
    mdl, _ = load_checkpoint(os.path.join(BASE_DIR, f"ckpt_epoch{epn:03d}.pt"))
    ims, lbs, sds, nfe, wall = generate_config(mdl, 50, "euler")
    row = score_config(ims, lbs, tag)
```

For each save: load that version of the student, draw the 96 doodles with 50 Euler steps, and score them.

Sample output (notebook output):

```text
epoch   5: recognisability 0.7708 (+/-0.043) | confidence 0.6639 | variety 22.35 | eval loss 0.3570
epoch  15: recognisability 0.8125 (+/-0.040) | confidence 0.7088 | variety 22.11 | eval loss 0.3164
epoch  60: recognisability 0.7708 (+/-0.043) | confidence 0.7135 | variety 22.23 | eval loss 0.2922
```

What the numbers mean:

- `recognisability 0.7708`: the judge recognised 74 of the 96 doodles (74 ÷ 96 = 0.7708), so 77%. The `+/-0.043` is the standard error from Cell 21.
- `confidence`: how sure the judge was, on average, about the animal that was asked for. It rises from 66% to 71%.
- `variety`: about 22 at every save, so the drawings stay just as varied.
- `eval loss`: the exam loss from Cell 15. It falls from 0.357 to 0.292, an 18% improvement.

The lesson: more practice made the arrows more exact (the loss fell), and the judge got a bit more confident. But the share of recognisable doodles went 77% → 81% → 77%. Those differences are within the wobble of about ±4%, so we cannot say the drawings got more recognisable. A better exam score on arrows is not the same as better drawings.

---

## Cell 24 (code): `# === secondary training comparison: is the limit data or training time?`

What this cell does, in one sentence: it trains (or loads) a second student that learned from only a quarter of the doodles (2,400 instead of 9,600), and compares it with the main student.

Analogy: does a student who practised with a smaller stack of flashcards do worse?

Sample output (notebook output):

```text
                                run  n_train_original  eval_loss  recognisability  mean_variety  train_time_min
    baseline (9,600 train drawings)              9600     0.2922           0.7708       22.2270          8.1003
quarter data (2,400 train drawings)              2400     0.3295           0.8646       22.0389          2.1059
```

What the numbers mean:

- `eval_loss`: the quarter-data student has a worse exam loss (0.330 vs 0.292). Its arrows are less exact.
- `recognisability`: but the judge recognised more of its doodles (86% vs 77%).
- `train_time_min`: it took 2.1 minutes instead of 8.1. With a quarter of the cards, each round is a quarter as long.

This is the same surprise as Cell 23: a worse loss did not mean worse-looking drawings. Cell 30's text describes this result the wrong way round (see the issues list).

---

## Cell 25 (code): `# === COMPARISON 2 (inference factor): steps and solver`

What this cell does, in one sentence: it answers Q2 ("how few nudges are enough, and is midpoint better than Euler?") by letting the same 60-round student draw the same 96 doodles with 6 step counts and 2 stepping methods, which is 12 settings in total.

Analogy: the same artist drawing the same 96 sketches, but sometimes allowed only 2 brush strokes and sometimes 100.

```python
STEP_GRID = [2, 5, 10, 25, 50, 100]
for method in ["euler", "midpoint"]:
    for ns in STEP_GRID:
        ims, lbs, sds, nfe, wall = generate_config(model60, ns, method)
```

Sample output (notebook output, shortened):

```text
euler     steps=  2  NFE/img=   2  recog=0.8438  variety= 14.99    78.9 ms/img
euler     steps= 10  NFE/img=  10  recog=0.8021  variety= 21.55   250.4 ms/img
euler     steps= 50  NFE/img=  50  recog=0.7708  variety= 22.23  1403.5 ms/img
euler     steps=100  NFE/img= 100  recog=0.7604  variety= 22.31  3791.7 ms/img
midpoint  steps=  5  NFE/img=  10  recog=0.7812  variety= 22.41   399.3 ms/img
midpoint  steps= 50  NFE/img= 100  recog=0.7604  variety= 22.41  1751.6 ms/img
```

What the numbers mean:

- `NFE/img`: how many times the student was asked per doodle. Euler asks once per step. Midpoint asks twice, so midpoint with 5 steps costs 10.
- `recog`: the share the judge recognised. It is between 76% and 84% everywhere. Strangely, 2 steps scores highest.
- `variety`: the clear effect. With only 2 steps it drops to 14.99, about a third lower than the roughly 22 everywhere else. The doodles are rushed and look too alike. That may also explain the high recognisability: simple, similar blobs are easy for the judge.
- `ms/img`: milliseconds per doodle, on the CPU. More steps take longer. These timings are rough and noisy (see the issues list).

### The fair match-up: same cost

Euler with 10 steps and midpoint with 5 steps both cost 10 questions. The notebook compares such pairs:

```text
 NFE  euler recog  midpoint recog  difference  stderr of difference  resolvable at 95%
  10       0.8021          0.7812     -0.0208                0.0586              False
  50       0.7708          0.7604     -0.0104                0.0611              False
 100       0.7604          0.7604      0.0000                0.0616              False
```

What the numbers mean:

- `difference`: midpoint minus Euler. For example, `-0.0208` means midpoint was 2 percentage points lower.
- `stderr of difference`: the wobble of that difference, about 6 percentage points. It comes from `sqrt(0.0407² + 0.0422²) = 0.0586`, which combines the two wobbles.
- `resolvable at 95%`: is the difference bigger than about 2 times its wobble? It is `False` every time. The differences are much smaller than the wobble.

The lesson: at the same cost, Euler and midpoint draw equally well. The arrows' paths are almost straight lines, so the smarter midpoint method has nothing to fix.

---

## Cell 26 (code): `# === comparison figures`

What this cell does, in one sentence: it draws four charts of the results from Cells 23 and 25.

1. Recognisability at rounds 5, 15 and 60, with error bars and the judge's ceiling (0.8271) as a dashed line.
2. Recognisability against the number of steps.
3. Variety against the cost (NFE). This is where the 2-step drop is easy to see.
4. Time per doodle against the cost.

There is no text output; the charts are saved in `results/partC/`.

---

## Cell 27 (code): `# === matched grids for the report`

What this cell does, in one sentence: it makes side-by-side picture grids where every column uses the same seed, so you can compare settings by eye.

The three grids:

- the student after rounds 5, 15 and 60
- Euler with 2, 10 and 50 steps
- Euler 10 steps against midpoint 5 steps (the same cost)

Analogy: before-and-after photos taken from exactly the same angle.

---

## Cell 28 (code): `# === 7. copying probe`

What this cell does, in one sentence: it checks whether the student just traced practice doodles instead of inventing new ones.

Analogy: a teacher checking a student's essay for plagiarism. You compare the essay with every book in the library and find the closest match. But you need a fair yardstick, so you also check some honest essays by other students. If the suspect essay is much closer to a library book than the honest ones are, it was probably copied.

```python
gen_d, gen_i = nearest_train_distance(gen_ims, DS["train_x"])          # the student's 96 doodles
real_idx = np.random.default_rng(3).choice(len(DS["eval_x"]), 200, replace=False)
real_d, _ = nearest_train_distance(DS["eval_x"][real_idx], DS["train_x"])   # 200 honest doodles
```

- For each of the 96 doodles from the 60-round student, find the closest of the 9,600 practice doodles.
- Do the same for 200 real exam doodles, which the student never saw. These are the honest essays.

Sample output (notebook output):

```text
                                      set   n  mean distance     min  5th pct     max
        generated (epoch 60, 96 drawings)  96        16.0457 10.0311  11.4386 21.1687
real HELD-OUT drawings (never trained on) 200        15.6896  8.7994  10.7198 23.6829
```

What the numbers mean:

- `mean distance 16.05`: on average, a generated doodle is 16.05 away from its closest practice doodle.
- Real unseen doodles are 15.69 away on average, which is about the same.
- `min 10.03`: even the generated doodle most like a practice doodle is still 10 away. The honest real doodles get even closer (8.80).

So the student's doodles are not closer to the practice set than honest new doodles are. There is no sign of copying.

---

## Cell 29 (code): `# === consolidated results`

What this cell does, in one sentence: it stacks all the result tables into one big table and saves it as `C_all_results.csv`.

```python
full = pd.concat([ck_df.assign(comparison="1: checkpoint"),
                  pd.DataFrame([row_s]).assign(comparison="1b: dataset size"),
                  inf_df.assign(comparison="2: sampler")], ignore_index=True, sort=False)
```

- `.assign(comparison="...")` adds a label column saying which experiment each row came from.
- `pd.concat` stacks the tables: 3 rows (checkpoints) + 1 row (quarter data) + 12 rows (samplers) = 16 rows.
- In the printed table, `NaN` means "does not apply". For example, the checkpoint rows have no timing column.

---

## Cell 30 (markdown): `## 8. What the results mean`

What it says: the written conclusions for the report.

- **Q1:** more training lowered the exam loss by about 18%, but the judge did not find the drawings measurably more recognisable. A better loss is not the same as better pictures.
- **Q2:** about 5 steps are enough. 2 steps loses about a third of the variety. Euler and midpoint cannot be told apart at the same cost.
- **Randomness:** the only randomness is the starting static.
- **Copying:** no sign of it.

Be careful if you copy text from this cell into a report. Several numbers in it do not match the actual outputs. They are listed in the issues section below.

---

## Cell 31 (code): `# === reproduction record`

What this cell does, in one sentence: it saves one big record of everything (data fingerprints, settings, results and computer details) as `C_reproduction_record.json`, and prints a summary.

Analogy: the lab notebook page that lets someone else repeat your experiment exactly.

Sample output (notebook output, shortened):

```text
PART C COMPLETE
parameters trained  : 1,838,081 (all, from scratch)
training time       : 8.1 min on cpu
outputs per config  : 96 (32 per class)
configurations      : 16
scorer ceiling      : 0.8271 on real held-out drawings
```

What the numbers mean: 1,838,081 weights were learned, 96 doodles were scored per setting, there are 16 settings, and the judge's ceiling is 82.7%. The line "8.1 min on cpu" is misleading. The 8.1 minutes was measured on the RTX 3060 graphics card. "cpu" only describes the computer that re-ran the notebook later.

---

## Things in the notebook that are wrong or confusing

Nothing in the notebook was changed. These are worth knowing about, especially if you copy any text into your report.

1. **Cell 13's safety check looks like a failure, but it isn't.** It prints "class embedding gets a gradient: False" and "2 of 85", and its own message says that would mean the animal label does nothing. The real reasons are that the last layer starts at zero (the megaphone at zero volume) and that the 8 test pictures are all cats. On the trained student, all 85 weight groups get a signal. A better check would do one practice step first and use a mix of cats, dogs and fish.
2. **Cell 30 gets the quarter-data result backwards.** It says that with a quarter of the data, recognisability went down at the same training time. Cell 24 shows recognisability went up (0.8646 vs 0.7708), and training was shorter (2.1 vs 8.1 minutes). Only the worse exam loss matches.
3. **Cell 30 quotes old numbers:**

   | Cell 30 says | The actual output says |
   |---|---|
   | The judge is right "about 84%" of the time | 82.7% (Cell 20) |
   | Round 60 reaches "about 96%" of the ceiling | 0.7708 ÷ 0.8271 = 93% |
   | Drawing cost rises "roughly 40-fold" | 3791.7 ÷ 78.9 = 48-fold |
   | 2 to 100 Euler steps are "within one margin of error" | They differ by 0.083, about two standard errors |

4. **Cell 31 says "8.1 min on cpu".** The training ran on the RTX 3060 graphics card. "cpu" describes the computer that later re-ran the notebook.
5. **"Ink fraction" means two different things.** Cell 7 counts any trace of grey (cats: 0.334). Cell 9 and the scoring code count only pixels darker than mid grey (cats: 0.198). It is the same name with a different ruler.
6. **"Margin of error" is used for one standard error.** The ±0.051 (Cell 3) and ±0.041 (Cell 30) figures are one standard error each. A proper 95% margin would be about twice as wide.
7. **The CPU timings in Cell 25 are noisy.** For example, midpoint with 200 questions per doodle came out faster than midpoint with 100. Treat the timings as rough sizes only.

---

## Glossary

| Word | Plain meaning | First seen |
|---|---|---|
| tensor | A table of numbers (list, grid or stack of grids) that PyTorch works with | Words section |
| shape | The size along each direction; `(64, 1, 28, 28)` = 64 pictures, 1 colour channel, 28 rows, 28 columns | Words section |
| batch | A handful of pictures processed together | Words section |
| seed | A starting number for the random-number maker; the same seed gives the same "random" numbers | Cell 3 |
| GPU / cuda, CPU | The graphics card (fast for this maths) / the normal processor | Cell 3 |
| checkpoint | A saved copy of the student's brain, like a video-game save | Cell 1 |
| static / noise | A picture of random numbers, the starting point for every drawing | Story |
| velocity / arrow | Which way and how much each pixel should move | Story |
| embedding | A way of writing one thing (a moment, an animal) as a list of numbers a network can use | Cell 11 |
| convolution (`Conv2d`) | A small window that slides over the picture looking for local patterns like edges | Cell 11 |
| feature map | One layer of the network's "notes" about the picture | Cell 11 |
| activation (SiLU, ReLU) | A small bend in the maths that lets stacked layers do more than one layer could | Cell 11 |
| residual block (ResBlock) | A block that learns a correction to add on top of its input, like editing a draft | Cell 11 |
| U-Net | A network that shrinks the picture to see the big shape, then grows it back, with skip connections | Cell 11 |
| skip connection | Passing detailed early notes straight across to the later part of the network | Cell 11 |
| weights / parameters | The adjustable numbers inside the network (1,838,081 here) | Cell 11 |
| class conditioning | Telling the network which animal to draw | Cell 11 |
| Euler step | Read the arrow, take one small step, repeat | Cell 12 |
| midpoint step | Peek half a step ahead, read the arrow there, then take the full step | Cell 12 |
| NFE | Number of times the network is asked per drawing, the real cost of drawing | Cell 12 |
| loss | One number saying how wrong the guesses are (here, the average squared difference) | Cell 13 |
| gradient | The slope for each weight: which way to turn it to make the loss smaller | Cell 13 |
| epoch | One full pass through all the practice pictures | Cell 15 |
| optimizer (AdamW) | The rule that turns gradients into actual weight changes | Cell 15 |
| learning rate | How big each weight change is | Cell 15 |
| augmentation | Making extra variety for free, here by mirror-flipping | Cell 5 |
| overfitting | Memorising the practice pictures instead of learning the skill | Cell 17 |
| CNN | A picture-reading network made mostly of convolutions (used for the judge) | Cell 20 |
| logits, softmax, argmax | Raw scores; scores turned into percentages; picking the biggest | Cell 20 |
| ceiling | The judge's accuracy on real doodles (82.7%), the best a fake set can fairly expect | Cell 20 |
| recognisability | Share of generated doodles the judge labels as the animal that was asked for | Cell 19 |
| variety | Average distance between pairs of doodles in a set; higher means more different | Cell 21 |
| standard error | How much a score typically wobbles by chance | Cell 21 |
| 95% range | Score ± about 2 standard errors, where the true value very likely sits | Cell 21 |

