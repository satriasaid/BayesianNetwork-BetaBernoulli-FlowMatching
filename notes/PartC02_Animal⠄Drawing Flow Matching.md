# Part C2: Animal Drawing with Flow Matching — A Step-by-Step Beginner's Guide

This guide explains `PartC02_Animal Drawing Flow Matching.ipynb` in simple English. We start with one number, calculate each step by hand, and then connect the idea to the notebook's images and Python code.

The result tables come from saved files in `results/partC/`. Examples marked **practice example** use small, made-up numbers to explain the method. They are not measurements from the trained model.

**Suggested reading pace:** Read Sections 1–5 first. Take a break after the one-pixel calculation. Then read training in Sections 6–8 and generation in Sections 9–11. Leave the result tables until those ideas feel familiar.

## 1. What are we trying to do?

We want a computer to make a small drawing of a **cat**, **dog**, or **fish**.

We give it two things:

1. A random starting image that looks like static.
2. A label saying which animal we want.

The computer changes the image a little at a time until it looks like an animal.

The starting image is called **noise**. The instructions for changing it are learned from real drawings.

The project has two different activities:

| Activity | What happens? | Are real example drawings supplied? |
|---|---|---|
| **Training** | The model practises predicting how an image should change. Its internal numbers are adjusted. | Yes. |
| **Generation**, also called sampling | The trained model uses its learned instructions to change new noise into a drawing. | No target drawing is supplied. |

During generation, the model is not given a finished cat to trace. It gets noise and the instruction “cat.”

## 2. An image is a table of numbers

Each image has 28 rows and 28 columns:

```text
28 × 28 = 784 pixels
```

A **pixel** is one small square in an image. A grayscale pixel stores one number, rather than separate red, green, and blue values.

The dataset stores pixel values between 0 and 255. The notebook rescales them:

```text
scaled value = (original value / 255) × 2 − 1

Original 0:
(0/255) × 2 − 1 = −1

Original 255:
(255/255) × 2 − 1 = 1

Original 127.5, a halfway value for illustration:
(127.5/255) × 2 − 1 = 0
```

In the notebook's displayed drawings, `−1` is white background and `+1` is dark ink because it uses the reversed grayscale colour map, `gray_r`.

A generated drawing is therefore a table of 784 numbers. The notebook displays that table as an image.

**Important:** Real training drawings are scaled into `[-1,1]`. Gaussian noise is not limited to that range. Intermediate images and generated values can also go outside it. The notebook clips values for display and classifier scoring, but does not clip them after every generation step.

### Where the drawings come from

The notebook uses Google's Quick, Draw! bitmap dataset, attributed to Google Creative Lab under CC BY 4.0. Its three classes are:

| Label in code | Meaning |
|---:|---|
| 0 | Cat |
| 1 | Dog |
| 2 | Fish |

It keeps 4,000 original drawings per class, after a loose ink filter, and splits them 80/20:

```text
For one class:
training drawings = 4,000 × 0.8 = 3,200
held-out drawings = 4,000 × 0.2 = 800

For all three classes:
training drawings = 3,200 × 3 = 9,600
held-out drawings = 800 × 3 = 2,400
```

**Held out** means the generator does not train on those drawings. They help check how it handles drawings outside its training set.

The split happens before augmentation. **Augmentation** means creating a simple variation of a training example. Here, the notebook sometimes flips a drawing left to right. Held-out drawings are not flipped for training.

The ink filter keeps drawings where between 5% and 55% of the raw pixels are above zero. For example, 157 nonzero pixels out of 784 gives:

```text
ink fraction = 157/784 ≈ 0.2003 = 20.03%
```

That drawing passes the filter. The filter removes very empty or very dense examples; it does not recognise animals.

![Examples of real training and held-out drawings](results/partC/C_real_data_grid.png)

## 3. Learn five symbols before looking at the formula

| Symbol | Say it as | Meaning |
|---|---|---|
| `x0` or `x_0` | “x zero” | The random starting image. |
| `x1` or `x_1` | “x one” | A real drawing used during training. |
| `t` | “time” | Progress from noise to drawing, between 0 and 1. |
| `x_t` | “x at time t” | An image partway between noise and the real drawing. |
| `y` | “label” | Which animal we want: cat, dog, or fish. |

Here, **time is progress**, not seconds on a clock:

```text
t = 0       start: all noise
t = 0.25    one quarter of the way along the training path
t = 0.50    halfway
t = 1       end: the real drawing
```

The model is written as `v_theta(t, x_t, y)`.

Read this as:

> Given the progress, the current image, and the animal label, predict how each pixel should change.

`theta` means all the model's learned internal numbers, or **parameters**. Unlike Part B, where theta was one unknown success probability, here it is a large collection of neural-network weights.

The output is called **velocity**. In this notebook, velocity means the rate at which each pixel value changes. It does not mean a cat physically walks across the picture.

## 4. The main training formula

The notebook builds a mixed image with:

```text
x_t = (1 − t) × x0 + t × x1
```

This is a weighted mixture of noise and a drawing.

At the start:

```text
t = 0
x_t = (1 − 0) × x0 + 0 × x1
    = x0
```

At the end:

```text
t = 1
x_t = (1 − 1) × x0 + 1 × x1
    = x1
```

Halfway:

```text
t = 0.5
x_t = 0.5 × x0 + 0.5 × x1
```

So we know exactly what the training image should be at any chosen time. We do not need to simulate the earlier times first.

The target velocity is:

```text
target velocity = x1 − x0
```

Why just subtract? The whole trip takes one unit of time:

```text
velocity = total change / total time
         = (x1 − x0) / (1 − 0)
         = x1 − x0
```

For a particular noise-and-drawing pair, this is constant along its straight training path.

## 5. Calculate one pixel by hand

**Practice example:** Ignore the other 783 pixels for a moment. Suppose:

```text
noise pixel:        x0 = −0.8
real drawing pixel: x1 = 0.6
chosen time:         t = 0.25
```

### Step 1: Find the mixed pixel

```text
x_t = (1 − 0.25) × (−0.8) + 0.25 × 0.6
    = 0.75 × (−0.8) + 0.15
    = −0.60 + 0.15
    = −0.45
```

At this stage, the pixel is `−0.45`.

### Step 2: Find the correct velocity

```text
target = x1 − x0
       = 0.6 − (−0.8)
       = 0.6 + 0.8
       = 1.4
```

The positive sign means the pixel value should increase.

Subtracting a negative number is adding. That is why the answer is 1.4, not −0.2.

### Step 3: Compare the model's answer

Suppose the model predicts velocity `1.1`. Its error is:

```text
error = prediction − target
      = 1.1 − 1.4
      = −0.3

squared error = (−0.3)² = 0.09
```

The square makes the error nonnegative. A prediction of 1.4 would give zero error.

### Step 4: Check other times

The same pair gives:

| Time `t` | Calculation | Mixed pixel |
|---:|---|---:|
| 0 | `1 × (−0.8) + 0 × 0.6` | −0.80 |
| 0.25 | `0.75 × (−0.8) + 0.25 × 0.6` | −0.45 |
| 0.50 | `0.50 × (−0.8) + 0.50 × 0.6` | −0.10 |
| 0.75 | `0.25 × (−0.8) + 0.75 × 0.6` | 0.25 |
| 1 | `0 × (−0.8) + 1 × 0.6` | 0.60 |

Every quarter of the trip changes the pixel by:

```text
0.25 × 1.4 = 0.35
```

The real notebook performs this arithmetic on all pixels at once.

## 6. How the loss combines many pixels

The **loss** is the number used to measure how wrong the velocity predictions are.

The notebook uses **mean squared error**:

```text
loss = average of (predicted velocity − target velocity)²
```

**Practice example:** Use a tiny two-pixel image:

```text
x0 = [−0.8,  0.2]
x1 = [ 0.6, −0.4]

target = x1 − x0
       = [0.6 − (−0.8), −0.4 − 0.2]
       = [1.4, −0.6]

Suppose the model predicts [1.1, −0.2].

Pixel 1 squared error = (1.1 − 1.4)² = 0.09
Pixel 2 squared error = (−0.2 − (−0.6))² = 0.16

loss = (0.09 + 0.16) / 2
     = 0.25 / 2
     = 0.125
```

For a real batch of 128 grayscale images, the mean is over:

```text
128 × 1 × 28 × 28 = 100,352 predicted pixel velocities
```

A loss of 0.125 is not 12.5% wrong animals. It is an average squared error on velocities.

### Why predicting an average can work

The model does not see the original noise-and-drawing pair separately. It sees only `t`, the mixed image, and the label. Different pairs can produce similar mixed images but different target velocities.

With squared error, the best prediction is an average of the possible target velocities for those inputs.

**Practice example:** Suppose two equally likely targets are 1 and 3:

```text
Predict 1: average squared error = ((1−1)² + (1−3)²)/2 = 2
Predict 2: average squared error = ((2−1)² + (2−3)²)/2 = 1
Predict 3: average squared error = ((3−1)² + (3−3)²)/2 = 2
```

The average, 2, gives the smallest error.

In the ideal flow-matching setup, the correct average velocity at each image and time carries the noise distribution to the drawing distribution. A finite trained model and numerical solver only approximate that ideal.

**A straight training path does not guarantee that every generated path is straight.** During generation, the predicted direction can change as the image and time change.

## 7. What does the U-Net do?

`VelocityUNet` is the main neural network. Its job is to predict a velocity image with the same shape as the input.

Its spatial sizes follow:

```text
28 × 28 → 14 × 14 → 7 × 7 → 14 × 14 → 28 × 28
```

The smaller views help combine information from different parts of a drawing. The larger views restore detail. **Skip connections** pass earlier details directly to later stages.

It also receives:

- A **time embedding**: a list of numbers representing the progress `t`. The code starts with sine and cosine features.
- A **class embedding**: a learned list of numbers representing cat, dog, or fish.

These let the same network make different predictions for different times and animal requests.

The saved model has **1,838,081 trainable parameters** and was trained from scratch. It uses 64 and 128 feature channels internally. A channel is a learned feature map; those channels are not colour channels in the input.

You do not need to calculate all those parameters by hand to understand the notebook. The one-pixel and two-pixel examples already show what its predictions are trained to do.

### Understanding the shape notation

```text
(B, 1, 28, 28)
```

means:

| Part | Meaning |
|---|---|
| `B` | Number of images in the batch. |
| `1` | One grayscale channel per image. |
| First `28` | Height. |
| Second `28` | Width. |

There is one sampled time per image. All 784 pixels in that image share that time.

## 8. Follow one training batch slowly

The main function is `train_model`.

1. Take 128 real training drawings and their labels.
2. Randomly flip some drawings horizontally.
3. Make a fresh noise image for each drawing.
4. Choose a random time between 0 and 1 for each drawing.
5. Build each mixed image using `(1−t)*x0 + t*x1`.
6. Calculate each target velocity using `x1−x0`.
7. Ask the network for its predicted velocities.
8. Calculate the mean squared error.
9. Calculate how the model's parameters contributed to that error.
10. Adjust the parameters, then repeat with another batch.

The central code is:

```python
x_t, t, target = flow_matching_batch(x1, generator=gen)
loss = torch.mean((model(t, x_t, y) - target) ** 2)

opt.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
sched.step()
```

| Code | Simple meaning |
|---|---|
| `zero_grad` | Clear the previous batch's stored gradients. |
| `backward` | Work out how changing the parameters would change the loss. |
| `clip_grad_norm_` | Limit very large gradients. |
| `opt.step` | Let the AdamW optimiser adjust the parameters. |
| `sched.step` | Update the learning rate according to the schedule. |

A **gradient** describes how the loss changes when a parameter changes. A **learning rate** controls the size of the adjustment.

An **epoch** is one pass through the training loader. The baseline has:

```text
9,600 images / 128 images per batch = 75 batches per epoch
60 epochs × 75 batches = 4,500 parameter updates
```

The initial learning rate is 0.0002. A cosine schedule gradually reduces it. Checkpoints are saved after epochs 5, 15, and 60. A **checkpoint** is a saved set of model parameters and related information.

The notebook loads existing checkpoints when they are present. Its saved baseline training time was about 8.10 minutes on an NVIDIA GeForce RTX 3060; your hardware may take a different time.

### Training epochs and generation steps are different

| Term | What changes? | Example |
|---|---|---|
| Training update | The model's parameters. | One batch causes one optimiser update. |
| Training epoch | The model works through the training loader. | The baseline trains for 60 epochs. |
| Generation step | The pixel values of a new image. | Generate with 50 Euler steps. |

Generating with 100 steps does not mean training the network for 100 more epochs.

### What is held-out loss?

The notebook also tests velocity predictions on drawings outside the generator's training set. It uses eight fixed times between 0.02 and 0.98 and repeatable noise to make checkpoint comparisons more consistent.

That checks velocity prediction on unseen drawings. It is still not a direct measure of how good the generated pictures look.

![Training and held-out velocity loss](results/partC/C_training_curves.png)

## 9. Generate a drawing with Euler steps

Now training is over. We have a fixed model and want a new drawing.

The function `sample_ode` starts with new noise. It repeatedly asks the model which way the pixels should move.

For `n_steps` equal steps:

```text
dt = 1 / n_steps
new pixel = current pixel + dt × predicted velocity
```

This is **Euler's method**. The same calculation is applied to every pixel.

### A full four-step practice example

For easy arithmetic, suppose a one-pixel model predicts velocity 1.4 at every step. Start at −0.8 and use four steps:

```text
dt = 1/4 = 0.25
change each step = 0.25 × 1.4 = 0.35

Start:            x = −0.80
Step 1, t=0.25:    x = −0.80 + 0.35 = −0.45
Step 2, t=0.50:    x = −0.45 + 0.35 = −0.10
Step 3, t=0.75:    x = −0.10 + 0.35 =  0.25
Step 4, t=1.00:    x =  0.25 + 0.35 =  0.60
```

This reaches 0.6 because we deliberately chose a constant, perfect velocity for the practice example.

In the real model, predictions can change at every step. It is not given a known endpoint during generation.

Smaller steps let the solver check and adjust direction more often. They cost more network calls and do not fix every model error.

![Noise gradually becomes an animal drawing](results/partC/C_trajectory_noise_to_drawing.png)

Read each row from left to right. It follows one image from noise toward a drawing.

## 10. Generate with midpoint steps

**Midpoint** asks for a direction twice during each step:

1. Predict the velocity at the current image.
2. Use it to estimate the image halfway through the step.
3. Predict the velocity at that halfway image and time.
4. Take the full step from the original image using this second velocity.

**Practice example:**

```text
Current pixel x = −0.8
Current time t = 0
dt = 0.25
First predicted velocity v1 = 1.4

Half-step time = 0 + 0.25/2 = 0.125
Half-step pixel = −0.8 + (0.25/2) × 1.4
                = −0.8 + 0.175
                = −0.625

Suppose the model predicts v2 = 1.2 at this halfway point.

New pixel = original pixel + dt × v2
          = −0.8 + 0.25 × 1.2
          = −0.8 + 0.3
          = −0.5
```

Notice that the full step starts from **−0.8**, not from the halfway estimate −0.625.

The velocities 1.4 and 1.2 are made up for this example. They show why a second look can change the update.

### Compare equal amounts of network work

**NFE** means **number of function evaluations**. Here, it counts calls to the velocity network along a generated image's path.

| Solver | Calls per step | Example | NFE per image |
|---|---:|---|---:|
| Euler | 1 | 10 steps | 10 |
| Midpoint | 2 | 5 steps | 10 |
| Midpoint | 2 | 10 steps | 20 |

So the fair comparison for a budget of 10 calls is **10 Euler steps against 5 midpoint steps**.

The notebook processes several images in a batch, so a call can handle several images together. NFE describes each image's path; it is not the number of images.

## 11. Why different noise makes different drawings

A **seed** controls the random-number generator.

With the same model, solver settings, label batch, seed, and deterministic execution setup, the notebook checks that generation repeats exactly. Changing the seed changes the starting noise and usually the resulting drawing.

No new random noise is added inside the Euler or midpoint loop. The randomness comes from the starting image.

For comparisons, the notebook uses 32 fixed seeds and requests all three animals:

```text
32 seeds × 3 classes = 96 drawings per configuration
```

Reusing the seeds makes comparisons easier: each setting starts from corresponding noise images.

The notebook also checks repeatability explicitly. Fixed seeds alone do not guarantee identical results across every hardware and software setup.

## 12. How do we measure the drawings?

The notebook uses several measures because one number cannot describe everything.

### 12.1 Recognisability: did a separate classifier choose the requested animal?

A second neural network, `DrawingCNN`, acts as a scorer. It does not generate drawings.

The 2,400 drawings held out from the generator are split again:

```text
Scorer training set = 2,400 × 0.8 = 1,920
Scorer test set = 2,400 × 0.2 = 480
```

The scorer trains on the first group and is checked on the second.

For each generated drawing, it predicts cat, dog, or fish. The notebook checks whether that matches the requested label.

**Manual calculation using the epoch-60 score:**

```text
76 matching labels out of 96 generated drawings
recognisability = 76/96 = 0.791666... = 79.17%
```

That means the classifier matched the request 76 times. It does not mean a human rated 79.17% of the drawings as beautiful.

The scorer's saved accuracy on real test drawings is **82.71%**. This is a useful reference, but **not a strict maximum** for generated scores. Generated examples could be easier for the classifier than real examples. The notebook sometimes calls this reference a “ceiling”; read that word cautiously.

### 12.2 Confidence: how much score did it give the requested class?

Suppose we asked for a cat and the scorer returned:

| Class | Scorer probability |
|---|---:|
| Cat | 0.60 |
| Dog | 0.30 |
| Fish | 0.10 |

It predicts cat, and the requested-class confidence is 0.60.

A prediction of `[0.95, 0.04, 0.01]` would also be correct, but more confident. This explains why confidence can improve while recognisability stays the same.

These are the classifier's probability estimates. They are not a guarantee of human agreement or perfectly calibrated probabilities.

### 12.3 Variety: how different are drawings within the same class?

The notebook calculates pixel distances between pairs of drawings within each animal class, then averages them. It finally averages the three class results.

**Practice example with two-pixel drawings:**

```text
A = [0, 0]
B = [1, 0]
C = [0, 1]

Distance A to B = sqrt((0−1)² + (0−0)²) = 1
Distance A to C = sqrt((0−0)² + (0−1)²) = 1
Distance B to C = sqrt((1−0)² + (0−1)²) = sqrt(2) ≈ 1.4142

Average pairwise distance = (1 + 1 + 1.4142)/3
                          ≈ 1.1381
```

`sqrt` means square root.

If every output were identical, every distance would be zero. Larger distances indicate more pixel variation, but random scribbles can also have large distances. More variety does not automatically mean better animals.

For 32 drawings of one class:

```text
Number of distinct pairs = 32 × 31 / 2 = 496
```

We divide by two because the pair A–B is the same pair as B–A.

### 12.4 Ink fraction

During generated-image scoring, the notebook counts pixels whose scaled value is above zero. This helps detect faint or overly dense outputs.

This threshold is different from the preprocessing filter, which counts raw pixels above zero **before** scaling. Therefore those two ink fractions should not be treated as the same measurement.

### 12.5 Uncertainty: 96 images are not a large sample

For recognisability `p = 76/96`, the notebook calculates:

```text
standard error = sqrt(p × (1−p) / 96)
               ≈ sqrt(0.7917 × 0.2083 / 96)
               ≈ 0.04145

Approximate 95% margin = 1.96 × standard error
                       ≈ 1.96 × 0.04145
                       ≈ 0.08124

Approximate 95% interval = 0.79167 ± 0.08124
                        ≈ [0.7104, 0.8729]
```

**Do not confuse standard error with a 95% margin.** The notebook sometimes uses “margin of error” for 0.041. Its saved interval columns actually use 1.96 times that value, about 0.081.

These are approximate uncertainty summaries. Because settings reuse seeds, comparisons are paired; the notebook's difference calculation does not account for that pairing. Avoid treating tiny score gaps as proof that one setting is better.

## 13. Q1: Does training longer help?

This comparison keeps the seeds and the sampler fixed at 50 Euler steps. It changes the training checkpoint.

Saved results, rounded:

| Training checkpoint | Held-out loss | Recognisability | Mean confidence | Variety |
|---|---:|---:|---:|---:|
| Epoch 5 | 0.3570 | 0.7708 | 0.6337 | 22.26 |
| Epoch 15 | 0.3164 | 0.7604 | 0.6718 | 22.17 |
| Epoch 60 | 0.2922 | 0.7917 | 0.7015 | 22.50 |

Calculate the loss improvement using the rounded values:

```text
Loss reduction = 0.3570 − 0.2922 = 0.0648
Relative reduction = 0.0648 / 0.3570 × 100 ≈ 18.15%
```

The model gets better at predicting velocities. Confidence also rises, and the variety measure stays similar.

But recognisability moves only from roughly 74/96 to 76/96. That small change does not establish a strong improvement in class recognition.

The important lesson is that **lower velocity loss and better class accuracy are different outcomes**. Look at the pictures and several measurements together.

![The same seeds at epochs 5, 15, and 60](results/partC/C_matched_grid_checkpoints.png)

Compare corresponding positions across checkpoint groups. The notebook describes later drawings as cleaner; use the grid to judge which details change and which failures remain.

### What happens with a quarter of the data?

| Run | Original training drawings | Held-out loss | Recognisability | Saved training time |
|---|---:|---:|---:|---:|
| Full data | 9,600 | 0.2922 | 0.7917 | 8.10 minutes |
| Quarter data | 2,400 | 0.3295 | 0.7812 | 2.11 minutes |

Both runs use 60 epochs. **They do not use equal training time or equal numbers of updates.** The smaller loader has fewer batches per epoch.

Also, the small-data run rebuilds its own data split and reports loss on that split. It is not a controlled comparison on one shared evaluation set.

The observed result is that this smaller-data setup has higher held-out loss and a slightly lower recognition score. It does not isolate dataset size from every other difference, so it cannot prove that data size alone caused the change.

## 14. Q2: How many generation steps do we need?

This comparison keeps the epoch-60 model and seeds fixed, and changes the Euler step count.

| Euler steps | Recognisability | Confidence | Variety | Milliseconds per image |
|---:|---:|---:|---:|---:|
| 2 | 0.7917 | 0.6331 | 14.85 | 0.82 |
| 5 | 0.7917 | 0.7024 | 20.64 | 1.88 |
| 10 | 0.7812 | 0.7059 | 21.73 | 3.65 |
| 25 | 0.7812 | 0.7003 | 22.29 | 8.98 |
| 50 | 0.7917 | 0.7015 | 22.50 | 17.82 |
| 100 | 0.8021 | 0.7000 | 22.61 | 35.22 |

Recognisability hardly changes, but two steps gives lower confidence and much less variety.

Calculate the variety drop from 50 steps to two steps:

```text
Difference = 22.50 − 14.85 = 7.65
Relative drop = 7.65/22.50 × 100 = 34%
```

Now calculate the time difference:

```text
100-step time / 2-step time = 35.22/0.82 ≈ 43
```

On this saved run, 100 steps took about 43 times as long per image as two steps, with little change in recognisability.

Around five Euler steps is a useful speed-quality compromise in this experiment. More steps still increase measured variety somewhat, so the best choice depends on which output qualities matter.

The timings are measurements from the saved hardware and batching setup, not promised timings for another computer.

![The same seeds with 2, 10, and 50 Euler steps](results/partC/C_matched_grid_steps.png)

### Does midpoint win at the same number of calls?

| NFE per image | Euler steps | Euler recognisability | Midpoint steps | Midpoint recognisability |
|---:|---:|---:|---:|---:|
| 10 | 10 | 0.7813 | 5 | 0.7708 |
| 50 | 50 | 0.7917 | 25 | 0.7813 |
| 100 | 100 | 0.8021 | 50 | 0.7917 |

Each score gap is about one drawing out of 96:

```text
1/96 ≈ 0.01042 = 1.04 percentage points
```

The notebook's approximate comparison does not resolve these differences at 95%. These results do not show a clear recognisability advantage for midpoint. They also do not prove the two methods are identical.

![Euler and midpoint with the same network-call budget](results/partC/C_matched_grid_solver_matched_nfe.png)

This experiment compares two solvers on this trained model. It does not test a diffusion baseline or establish that all flow-matching models need fewer steps than all diffusion models.

## 15. Is the model copying training pictures?

For each generated drawing, the notebook finds the nearest training drawing in pixel distance.

A simple practice example:

```text
Distances from one generated image to three training images:
3.2, 1.1, 2.7

Nearest-training distance = minimum = 1.1
```

It also finds nearest-training distances for 200 real held-out drawings. Those real drawings provide a reference for how close an unseen drawing can naturally be.

| Group | Mean distance to nearest training drawing |
|---|---:|
| 96 generated drawings | 16.60 |
| 200 real held-out drawings | 15.69 |

Generated drawings are farther away on average:

```text
16.60 − 15.69 = 0.91
```

This is evidence against simple pixel-level copying in the inspected outputs. It is not proof that memorisation is impossible. A shifted or slightly changed copy can have a large pixel distance.

![Nearest-training distances and the closest generated/training pairs](results/partC/C_memorisation_probe.png)

Look at both the distance distribution and the closest pairs. Two simple cat drawings can naturally share an outline without being the same drawing.

## 16. Connect the ideas to the code

| Name in the notebook | Plain-English job |
|---|---|
| `build_dataset` | Select, filter, scale, and split real drawings. |
| `SinusoidalTimeEmbedding` | Turn time into features the network can use. |
| `ResBlock` | Process image features with a residual connection and conditioning. |
| `VelocityUNet` | Predict pixel velocities from image, time, and animal label. |
| `flow_matching_batch` | Make noise, choose times, build mixtures, and calculate targets. |
| `train_model` | Repeat predictions and parameter updates. |
| `eval_loss` | Measure velocity error on held-out drawings. |
| `load_checkpoint` | Rebuild a model and load saved parameters. |
| `sample_ode` | Generate drawings with Euler or midpoint. |
| `DrawingCNN` | Classify drawings for evaluation. |
| `recognisability` | Measure requested-label matches and confidence. |
| `variety` | Measure average distance between image pairs. |
| `nearest_train_distance` | Find each drawing's closest training example. |
| `generate_config` | Generate all 96 outputs for one setting. |
| `score_config` | Collect that setting's evaluation measures. |
| `check_determinism` | Check same-seed repeatability and different-seed variation. |
| `save_grid` / `montage` | Save image grids for inspection and comparison. |

### A small detail in the gradient check

The final output layer starts with zero weights. On the very first backward pass, this can block nonzero gradients from reaching earlier layers, including the class embedding. After the output layer changes, gradients can reach those layers.

So a first-pass “class embedding gets a gradient: False” is not by itself proof that conditioning is permanently broken, despite the nearby notebook comment.

## 17. Notebook map

These are **cell positions starting at 1**, counting both text and code cells. They are not the execution numbers shown in brackets by Jupyter.

| Cells | What to read |
|---|---|
| 1–3 | Task, package installation, folders, settings, and random seeds. |
| 4 | Flow-matching method and the two research questions. |
| 5–9 | Download, inspect, filter, split, and display the data. |
| 10–11 | Model explanation and network classes. |
| 12 | Build training examples and generate with Euler or midpoint. |
| 13–14 | Shape checks, gradients, endpoints, and code explanation. |
| 15–16 | Training loop and loading or training the baseline. |
| 17–18 | Loss curves and noise-to-drawing trajectory. |
| 19–21 | Comparison protocol, scoring CNN, and measurement helpers. |
| 22 | Repeatability check. |
| 23 | Compare checkpoints at epochs 5, 15, and 60. |
| 24 | Compare the full-data and quarter-data runs. |
| 25 | Compare generation steps and solvers. |
| 26–27 | Summary plots and matched image grids. |
| 28 | Copying probe. |
| 29–31 | Combined results, interpretation, and reproduction record. |

To study the maths first, focus on cells 4, 12, 14, and 15. Then read the generation and evaluation sections.

## 18. Which saved files should I open?

| File in `results/partC/` | What it contains |
|---|---|
| `C_all_results.csv` | Combined measurements for the tested settings. |
| `C_comparison1_checkpoints.csv` | Epoch 5, 15, and 60 comparison. |
| `C_comparison1b_dataset_size.csv` | Full-data and quarter-data comparison. |
| `C_comparison2_sampler.csv` | Step counts, solver choices, scores, and timings. |
| `C_comparison2_matched_nfe.csv` | Euler and midpoint compared at equal calls. |
| `C_memorisation_distances.csv` | Generated and real held-out nearest-training distances. |
| `C_reproduction_record.json` | Settings, dataset records, hardware, and results. |
| `sample_grids/` | Per-setting picture grids and complete saved output arrays. |

![Overview of the main comparisons](results/partC/C_comparison_figures.png)

A CSV is a table you can open in a spreadsheet or text editor. JSON stores named settings and values. PNG files are images. The saved NPZ files contain image arrays and their labels and seeds.

## 19. Small practice questions

Try each question before reading its answer.

### Question 1: Mix one pixel

Noise is −1, the real pixel is 1, and time is 0.25. What is the mixed value?

```text
x_t = 0.75 × (−1) + 0.25 × 1
    = −0.75 + 0.25
    = −0.5
```

### Question 2: Find the target

For the same noise and real pixel, what is the target velocity?

```text
target = 1 − (−1) = 2
```

### Question 3: Calculate the loss

The target velocity is 2 and the model predicts 1.5. What is the one-pixel squared error?

```text
(1.5 − 2)² = (−0.5)² = 0.25
```

### Question 4: Take one Euler step

Start at −0.5. Use `dt=0.1` and predicted velocity 2.

```text
new x = −0.5 + 0.1 × 2 = −0.3
```

### Question 5: Count network calls

You can afford 20 calls per image. How many steps can each solver take?

```text
Euler:    20/1 = 20 steps
Midpoint: 20/2 = 10 steps
```

### Question 6: Interpret the score

The scorer matches the requested animal in 24 of 32 generated images. What is recognisability?

```text
24/32 = 0.75 = 75%
```

This measures class agreement with the scorer. It does not measure drawing beauty.

### Question 7: Does lower loss guarantee nicer pictures?

No. Loss measures velocity prediction error. Check the generated pictures, recognition, confidence, and variety too.

### Question 8: Does the generator need a real target drawing when making a new image?

No. It needs starting noise, an animal label, and its trained velocity network.

## 20. A short vocabulary list

| Word | Simple meaning |
|---|---|
| Noise | A random starting image. |
| Gaussian | The bell-shaped distribution used to draw the starting pixel values. |
| Conditional | Guided by extra information; here, the animal label. |
| Interpolation | Mixing two endpoints according to a chosen fraction. |
| Velocity | How fast each pixel value should change. |
| Flow matching | Training a network to predict target velocities along chosen paths. |
| Loss | A number measuring prediction error. |
| Parameter | An internal number learned by the model. |
| Batch | A group of examples processed together. |
| Epoch | One pass through the training loader. |
| Checkpoint | A saved model state. |
| ODE | An equation specifying how a state changes over time. Here, how image values change along the path. |
| Solver | A rule for taking steps along that equation, such as Euler or midpoint. |
| Inference / sampling | Using the trained model to generate a drawing. |
| NFE | Network calls along an image's generation path. |
| Held-out data | Examples excluded from a particular model's training. |
| Seed | A setting used to repeat random-number generation. |

The three calculations to remember are:

```text
Build a training image:  x_t = (1−t) × x0 + t × x1
Find its target:         target = x1 − x0
Generate with Euler:     new x = x + dt × predicted velocity
```

