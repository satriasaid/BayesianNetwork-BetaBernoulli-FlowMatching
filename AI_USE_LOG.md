# AI-Use and Development Log — COMP6010 Assignment

**Student:** Satria Pamungkas · **ID:** 21522611 · **Part C option:** C2 (flow matching)

**Tool used:** Claude (Anthropic), model Opus 5, accessed through the Claude Code CLI.
**Dates of use:** 9–10 September 2026.
**Scope of assistance:** explanation of flow matching for images, code scaffolding for the U-Net and
the sampler, notebook structure, and debugging. All mathematical reasoning, experimental design
decisions, verification checks and the claims in the report are my own responsibility, and every
number reported was produced by code that I ran on my own machine (RTX 3060).

Five interactions are recorded below, each with the prompt, the relevant part of the response, the
**check I performed**, and the resulting decision. Interaction 3 is an independently verified
correctness check; interaction 4 is a prediction-versus-observation comparison.

---

## Interaction 1 — Choosing the Part C option against my actual resources

**Prompt (paraphrased).** "I have an RTX 3060 (12 GB) and my prior coursework covers flow matching on
2-D two-moons (Practical 7). The assignment offers a from-scratch diffusion model on Oxford-IIIT Pet
photos (C1), from-scratch flow matching on Quick, Draw! (C2), or LoRA fine-tuning of Stable Diffusion
(C3/C4). Which fits my resources and prior knowledge, and what are the risks of each? Do not write
code yet."

**Response excerpt.** Recommended C2: the brief itself designates it the lower-compute route, the
28×28 grayscale data removes the need for a VAE or large U-Net, and it extends the two-moons
velocity-regression code directly. Noted that C1 has only ~200 images per category (the brief calls
it "demanding"), and that C3/C4 add a ~4 GB checkpoint download plus licence-recording obligations.

**My check.** I read the brief's own statements rather than accepting the summary: it does say "C2 is
the recommended starting point for limited hardware" and "no automatic bonus is awarded for choosing
a larger model or a harder option." I then ran a **feasibility pilot** myself before committing —
20 epochs on the full 12,000-drawing subset, which took 2.1 minutes and already produced
class-distinguishable output (cat ears, fish tails). Per-class ink fractions of the generated samples
matched the real data to within 0.01–0.02.

**Decision.** Chose C2. The pilot also told me a 60-epoch baseline plus a second full training run
and ~12 sampler configurations would all fit comfortably in the time available, so I designed the
experiment for two training comparisons rather than one.

---

## Interaction 2 — Where the class conditioning should enter the network

**Prompt (paraphrased).** "For a small conditional flow-matching U-Net on 28×28 images, explain the
options for injecting the time and class conditioning, and what would fail silently if I got it
wrong. Describe the tensor shapes."

**Response excerpt.** Suggested sinusoidal Fourier features for $t$ through a small MLP, a learned
`nn.Embedding` for the class, adding the two embeddings and injecting the sum into *every* residual
block via a per-block linear layer broadcast over spatial dimensions — shape `(B, C)` →
`(B, C, 1, 1)`. Warned that conditioning applied only at the input layer tends to be ignored, and
that a bug in the embedding path fails *silently*: training loss still falls (the model learns the
class-marginal velocity field) but all classes generate the same thing.

**My check.** I wrote an explicit gradient test rather than trusting this (in notebook 03, §3): after
one backward pass I assert `class_embed.weight.grad.abs().sum() > 0`, and print the count of
parameters receiving non-zero gradient. I also verified the interpolation endpoints numerically
(`max|x_t - x0|` at `t=0` and `max|x_t - x1|` at `t=1`, both ~1e-7). The real test of conditioning is
behavioural, and it is the recognisability measure in §6: a dead conditioning path would score at
chance (~0.33).

**Decision.** Adopted per-block additive conditioning. Kept the gradient assertion in the notebook as
a permanent guard, because the failure mode described is exactly the kind that would otherwise survive
to the report.

---

## Interaction 3 — Independently verified check: is my Part A likelihood-weighting estimator correct?

**Prompt (paraphrased).** "For this five-variable Bayesian network, explain likelihood weighting and
propose checks that would detect a wrong CPT lookup or a missing normalisation. Do not write the full
solution."

**Response excerpt.** Proposed three checks: (i) the mean weight should estimate $P(e)$, so compare
it to the exact evidence probability; (ii) dividing by $N$ instead of $\sum w$ yields the *joint*
$P(F=1, e)$ rather than the conditional, detectable by comparing against enumeration; (iii) counting
samples *unweighted* should collapse to the prior $P(F=1)$, so that value appearing is a signature of
forgotten weights.

**My verification — done independently of the suggestion.** I did not rely on the assistant's
reasoning for correctness. I wrote a **separate brute-force enumeration** over all 32 states, in a
standalone script with no shared code with the notebook, and confirmed the three exact values
$q_1 = 0.2315035800$, $q_2 = 0.2932938311$, $q_3 = 0.1686746988$ agree to 10 decimal places with the
notebook's enumeration. I then confirmed each suggested check numerically:

- mean weight $\approx P(e)$: for $q_2$, mean weight $0.1600$ against exact $P(e) = 0.159316$;
- the unweighted count returns $0.0800$ — exactly the prior $P(F=1) = 0.08$ — confirming check (iii)
  has real diagnostic power;
- and I added a check of my own that the assistant had not suggested: I deliberately **transposed the
  $H,F$ indices** in the $D$ CPT lookup and found the joint *still summed to 1*, so normalisation
  alone would not catch that bug, while the query values changed visibly. That check is in the
  notebook (A1) with the buggy variant included, because it shows the normalisation test is weaker
  than it looks.

**Decision.** Kept all three suggested checks and added the transposed-CPT test. The lesson recorded
in the notebook is that a passing normalisation check is necessary but not sufficient.

---

## Interaction 4 — Prediction versus observation: Euler vs midpoint integration

**Prompt (paraphrased).** "I predict that midpoint integration will clearly beat Euler at low ODE step
counts for my flow-matching model, because Practical 7 showed a large accuracy improvement from
second-order integration on two-moons. Justify or challenge this prediction, and propose a controlled
test."

**Response excerpt.** Challenged the prediction on two grounds. First, a fair comparison must be at
matched **network evaluations (NFE)**, not matched step count, since midpoint costs two evaluations
per step — "midpoint 10 vs Euler 10" silently gives midpoint double the compute. Second, a
second-order solver's advantage scales with trajectory *curvature*, and straight-line flow matching is
trained specifically to make trajectories nearly straight, so the advantage should be smaller than in
a setting where paths bend. Suggested reporting recognisability against NFE per image.

**What I predicted, then what I observed.** I recorded the prediction *before running*: a clear
midpoint advantage at low step counts. **Observed:** on the matched-NFE comparison
(`results/partC/C_comparison2_matched_nfe.csv`), the difference between Euler and midpoint is small —
much smaller than I expected — while the step-count comparison alone had made midpoint look better
simply because it was using twice the compute.

**Why my prediction was wrong, and my own diagnosis.** The discrepancy is not that the assistant was
right and I was wrong about integration theory; it is that Practical 7 and this experiment measure
different things. Practical 7 measured **round-trip reconstruction MSE**, which is dominated by
integration accuracy and does improve as $O(\Delta t^2)$. Here I measure whether the ODE endpoint
lands in the right region of image space — a much coarser criterion that a first-order solver already
satisfies once the path is close to straight. I had conflated "more accurate integration" with "better
samples". This is discussed as a failed expectation in §8 of notebook 03 and in the report.

**Decision.** Report both the per-step-count and the matched-NFE comparisons, and make matched-NFE
the basis of the conclusion. Added the NFE counter to `sample_ode` so the compute cost is always
recorded alongside the quality number rather than reconstructed afterwards.

---

## Interaction 5 — A suggestion I rejected: the data filter

**Prompt (paraphrased).** "Suggest a preprocessing filter for Quick, Draw! bitmaps to remove
unusable drawings."

**Response excerpt.** Suggested rejecting drawings with ink fraction below 0.02 or above 0.60 as
near-blank or near-solid, and initially I coded exactly those thresholds.

**My check.** I measured the actual ink-fraction distribution over 20,000 sampled drawings per
category before trusting the numbers. The bulk lies between roughly 0.16 and 0.43 (median ~0.24–0.33),
with minimum around 0.03 and maximum around 0.70. **The suggested thresholds rejected essentially
nothing** — 0 of 1,300 candidates in my first run — so the "filter" was decorative and would have
been misleading to disclose as a filter in the report.

**Decision.** Rejected the suggested values and recalibrated to [0.05, 0.55] against the measured
distribution, which removes a genuine handful of outliers (8 of 20,700 candidates, well under 1%) and
is honestly describable. The measured distribution table is reported in notebook 03 §2 alongside the
actual rejection counts, and I explicitly state that the filter is far too loose to constitute
selecting attractive drawings. This is recorded because it shows a suggestion can be plausible,
syntactically fine, and still wrong about the data.

---

## Summary of my own verification work

| Check | Where | Result |
|---|---|---|
| Joint distribution sums to 1 over all 32 states | nb 01, A1 | passes to 1e-12 |
| Exact query values re-derived by independent standalone enumeration | separate script | agree to 10 d.p. |
| Transposed-CPT bug still normalises to 1 (so normalisation is a weak test) | nb 01, A1 | confirmed; query check catches it |
| Vectorised vs scalar forward sampler agree in distribution | nb 01, A2 | marginals match exact to ~1e-3 |
| Unweighted likelihood-weighting counts collapse to the prior | nb 01, A3 | returns 0.0800 = P(F=1) |
| Mean weight estimates P(evidence) | nb 01, A3 | 0.1600 vs exact 0.159316 |
| Sequential vs pooled Beta updating identical | nb 02, B2 | assert passes, Beta(87,18) |
| Conjugate posterior vs brute-force numerical Bayes on a grid | nb 02, B2 | max pdf difference ~1e-9 |
| Batch ordering irrelevant (all 6 permutations) | nb 02, B2 | single posterior recovered |
| P(θ>0.8) analytic vs 200,000 Monte-Carlo draws | nb 02, B3 | within 1 standard error |
| Class embedding receives non-zero gradient | nb 03, §3 | passes |
| Interpolation endpoints exact at t=0 and t=1 | nb 03, §3 | ~1e-7 |
| Generated ink fraction vs real per class | nb 03, pilot + §6 | within 0.01–0.02 |
| Generated NN-distance vs real-held-out baseline | nb 03, §7 | reported, no evidence of copying |
| Same-seed sampling is bitwise reproducible | nb 03, §4 | **initially FAILED** (~7e-4 drift from cuDNN); fixed with `enable_deterministic()`, now asserted |
| Inference-only route works from a clean ZIP extract | clean extract test | checkpoint loads, samples generate |

**Note on responsibility.** No result, runtime, citation or figure in this submission was fabricated.
Every number in the report is traceable to a CSV or JSON file in `results/`, produced by the notebooks
in `notebooks/` on the hardware documented in the README.
