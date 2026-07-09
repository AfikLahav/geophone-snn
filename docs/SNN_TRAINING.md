# How Spiking Neural Networks Are Trained

*A pedagogical guide for people who already understand backpropagation and SGD but are new to spiking neural networks (SNNs).*

---

## 0. Orientation: what makes an SNN different

In a standard ANN, a neuron computes `y = σ(Wx + b)` — a continuous, differentiable nonlinearity applied to a weighted sum. Activations are real numbers that flow through the network in a single forward sweep.

A spiking neuron is different in two fundamental ways:

1. **It is stateful in time.** The neuron integrates input current into a *membrane potential* `U(t)` that decays (leaks) over time, like a leaky capacitor. It has memory.
2. **Its output is binary and event-driven.** When `U(t)` crosses a threshold `θ`, the neuron emits a *spike* (a `1`); otherwise it outputs `0`. After firing, the membrane is reset.

The most common model is the **Leaky Integrate-and-Fire (LIF)** neuron. In discrete time (the form used in nearly all deep-learning SNN frameworks), one timestep of a LIF neuron is:

```
U[t] = β · U[t-1] + W·X[t] − S[t-1]·θ        (membrane update with reset-by-subtraction)
S[t] = Θ(U[t] − θ)                            (spike generation; Θ is the Heaviside step)
```

where:
- `β = exp(−Δt/τ) ∈ (0,1)` is the **membrane decay factor** (τ is the membrane time constant); larger β = longer memory.
- `W·X[t]` is the input current at this timestep.
- `θ` is the firing **threshold** (often normalized to 1).
- `S[t] ∈ {0,1}` is the spike output.
- The `−S[t-1]·θ` term is the **reset**: when the neuron fires, its potential drops by θ. (See §7 for reset variants.)

This is the structure you must keep in mind for everything that follows. An SNN is essentially a recurrent network whose nonlinearity is a step function. Both of those properties — the recurrence-in-time and the step nonlinearity — shape how training works.

Two foundational, freely-available references underpin this whole guide:
- Neftci, Mostafa & Zenke (2019), *Surrogate Gradient Learning in Spiking Neural Networks*, IEEE Signal Processing Magazine — https://arxiv.org/abs/1901.09948
- Eshraghian et al. (2023), *Training Spiking Neural Networks Using Lessons From Deep Learning* (the snnTorch paper) — https://arxiv.org/abs/2109.12894

---

## 1. The core problem: the spike function is non-differentiable

### Why standard backprop needs a usable derivative

Backprop is just the chain rule applied through a computational graph. To update a weight `W`, you need:

```
∂L/∂W = ∂L/∂S · ∂S/∂U · ∂U/∂W
```

Every factor must be a finite, meaningful number. The middle factor, `∂S/∂U`, is the derivative of the spike-generating nonlinearity with respect to the membrane potential. In an ANN this is the slope of a sigmoid or ReLU — well-behaved and informative. In an SNN, `S = Θ(U − θ)` is the **Heaviside step function**, and its derivative is the problem.

### What the derivative of the spike actually is

The Heaviside step is flat everywhere except at the threshold, where it jumps from 0 to 1 instantaneously. Therefore:

- For all `U ≠ θ`: the function is constant, so `∂S/∂U = 0`. **The gradient is zero almost everywhere.**
- At `U = θ`: the function is discontinuous; its "derivative" is the **Dirac delta** — infinite, with zero width, and not a usable number.

So the true derivative is `∂S/∂U = δ(U − θ)`: zero almost everywhere and ill-defined (infinite) exactly at threshold.

### Why this *completely* breaks training

Plug that into the chain rule and the consequences are fatal in both regimes:

- **Away from threshold (the common case):** `∂S/∂U = 0`, so `∂L/∂W = 0`. The gradient signal is annihilated. No matter how wrong the output is, the weight receives **zero update**. This is the heart of the **dead neuron problem** — if a neuron isn't sitting exactly at threshold, no learning signal flows back through it (the snnTorch paper, arXiv 2109.12894, frames "neurons not firing → zero gradients" as the single most common reason SNNs fail to train).
- **At threshold:** the gradient is infinite, which would blow up any update.

The net effect: vanilla backpropagation applied to the hard spike produces a gradient that is **either 0 or ∞** — never the smooth, finite, informative signal that gradient descent needs. The optimizer is blind. This is *the* central obstacle that every direct SNN training method must work around.

> **Intuition.** Backprop asks "if I nudge the membrane potential a little, how does the output change?" For a step function the honest answer is "nothing changes (you're on a flat part)… unless you happen to be exactly at the cliff edge, in which case everything changes at once." Neither answer gives a direction to descend. Optimization needs a gentle slope, and the step function has none.

Sources: https://arxiv.org/abs/1901.09948 · https://arxiv.org/abs/2109.12894 · https://www.nature.com/articles/s41598-021-91786-z

---

## 2. Surrogate gradients — the key enabler

### The central trick

The breakthrough that made deep SNN training practical is the **surrogate gradient** (sometimes "pseudo-derivative" or "surrogate derivative"). The idea, formalized and popularized by **Neftci, Mostafa & Zenke (2019)** and introduced for multilayer nets by **Zenke & Ganguli's SuperSpike (2018)**, is disarmingly simple:

> **Use the hard step function on the forward pass, but substitute a smooth, well-behaved surrogate derivative on the backward pass.**

Concretely, you decouple the two passes:

- **Forward:** `S = Θ(U − θ)` — a real, binary spike. The network's actual computation and the loss are unchanged; you still get crisp 0/1 spikes and all the sparsity/energy benefits.
- **Backward:** replace the ill-defined `∂S/∂U = δ(U − θ)` with `∂S̃/∂U = g(U − θ)`, where `g` is a smooth bump-shaped function (a continuous relaxation of the delta). Now `∂L/∂W = ∂L/∂S · g(U−θ) · ∂U/∂W` is finite and nonzero for membrane potentials *near* threshold, so a gradient flows.

This is a deliberate, principled "lie" in the backward pass. The forward computation is exact; only the gradient is approximated. In autograd frameworks it's implemented as a custom function whose `forward()` returns the Heaviside output and whose `backward()` returns the surrogate `g` (this is exactly how snnTorch, Norse, SpikingJelly, and the SpyTorch tutorial do it — https://github.com/fzenke/spytorch).

It also generalizes the **straight-through estimator** used for binary/quantized ANNs: there, the binary forward is paired with an identity (`∂S/∂U = 1`) backward. Surrogate gradients are the same idea with a shaped, threshold-centered bump instead of a flat 1.

### Common surrogate functions

The surrogate `g` is typically the derivative of a smooth sigmoid-like function (a soft step). Below are the standard choices, with the exact backward-pass formulas and default hyperparameters from the snnTorch library (https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html). Let `x = U − θ` (distance from threshold).

| Surrogate | Backward-pass derivative `∂S̃/∂U` | Steepness param (default) | Notes |
|---|---|---|---|
| **ArcTan (atan)** | `1/π · 1 / (1 + (π·x·α/2)²)` | `α = 2` | Derivative of a scaled arctan (soft step). **snnTorch's default since 2023** — robust, cheap. |
| **Fast sigmoid / SuperSpike** | `1 / (1 + k·|x|)²` | `k = 25` | Derivative of the *fast sigmoid* `x/(1+k|x|)`. This is the original **SuperSpike** surrogate (Zenke & Ganguli 2018). Cheap — no `exp`. |
| **Sigmoid** | `k·e^{−kx} / (e^{−kx}+1)²` | `k = 25` | Derivative of the logistic sigmoid. The "textbook" smooth step. |
| **Triangular / piecewise-linear** | linear tent peaking at `x=0`, zero outside a window | width `1` | Cheapest of all; favored on neuromorphic hardware. The "boxcar"/rectangular variant is its close cousin. |
| **Straight-through estimator** | `1` everywhere | — | Constant gradient; ignores `x`. Simple baseline. |
| **SpikeRateEscape (Boltzmann)** | `k·e^{−β|x−1|}` | `β=1, k=25` | From the noisy-neuron / escape-noise interpretation of stochastic firing. |

The pattern is consistent: all the bump-shaped surrogates are **largest at `x = 0` (membrane right at threshold)** and **decay toward 0 as the membrane moves far from threshold**. This is the smooth stand-in for the delta function: "you get the most gradient when you're closest to flipping the spike."

### The width/steepness hyperparameter (α, k, slope) — why it matters

The single most important surrogate hyperparameter is the **steepness/width** — `α` for arctan, `k` (slope) for the sigmoids, window width for triangular. It controls a fundamental trade-off:

- **Large slope / narrow surrogate** → the bump is tall and thin, hugging the true delta closely. The backward pass is a *faithful* approximation of the real (degenerate) gradient, but only neurons with membrane potential *very* close to threshold receive any signal. Gradients become **sparse and prone to vanishing**; many neurons get ~0 gradient → harder optimization, risk of dead neurons.
- **Small slope / wide surrogate** → the bump is short and broad. Many neurons (even far from threshold) receive a gradient, so the signal is dense and optimization is well-conditioned. But the gradient is a *poor* approximation of the true spike behavior — this is the **gradient mismatch** (see §6). Too wide and you're essentially training a smooth ANN that bears little relation to the spiking forward pass; you can also get **exploding gradients** through depth×time.

So α/k interpolates between **faithful-but-vanishing** and **smooth-but-mismatched**. There is a sweet spot, and it is task- and depth-dependent. Practical guidance from the literature:

- Zenke & Vogels (2021), *The Remarkable Robustness of Surrogate Gradient Learning* (https://research-explorer.ista.ac.at/download/8253/11131/2021_NeuralComputation_Zenke.pdf), showed that learning is **surprisingly robust to the *shape*** of the surrogate (arctan vs sigmoid vs triangular barely matters) but **sensitive to its *scale*** (the steepness). Get the width roughly right and the exact functional form is secondary.
- Many modern methods make the steepness **adaptive** — warming it up over training epochs (start wide for dense early gradients, sharpen later for fidelity), or learning it per-layer (e.g. Learnable Surrogate Gradients; AdaLi at https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1795946/full).

Sources: https://arxiv.org/abs/1901.09948 · https://pmc.ncbi.nlm.nih.gov/articles/PMC6118408/ (SuperSpike) · https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html · https://research-explorer.ista.ac.at/download/8253/11131/2021_NeuralComputation_Zenke.pdf

---

## 3. Backpropagation-through-time (BPTT) for SNNs

The surrogate gradient fixes the *spatial* (layer-to-layer) gradient. But there's a second dimension: **time**. Because the membrane potential carries state from one timestep to the next, an SNN is a recurrent system, and you train it the way you train RNNs — with **backpropagation through time (BPTT)**.

### Unrolling the membrane dynamics

To process an input you run the network for `T` timesteps. At each step `t` the neuron updates `U[t] = β·U[t−1] + W·X[t] − S[t−1]·θ` and emits `S[t]`. Conceptually you **unroll** this recurrence into a feedforward graph of depth `T`: one copy ("instantiation") of every neuron per timestep, with the membrane state wired from each copy to the next. A single physical neuron unrolled over `T` steps becomes `T` nodes in the graph (Sinabs/SpikingJelly describe exactly this; https://sinabs.readthedocs.io/v1.2.1/tutorials/bptt.html).

Crucially, **even a purely feedforward SNN (no recurrent synapses) is recurrent in this sense**, because the membrane potential `U[t]` depends on `U[t−1]`. The state persistence *is* the recurrence.

### The computational graph and where gradients flow

After the forward run you compute a loss (e.g. on the spike count or membrane potential of the output neurons over all `T` steps). Backprop then flows through the unrolled graph along **two kinds of paths**:

1. The **explicit/spatial path**: through the spike `S[t]` and the weights — this is where the **surrogate gradient** from §2 is applied at every (neuron, timestep) pair.
2. The **implicit/temporal path**: through the membrane recurrence `U[t] → U[t−1]` — this carries gradient *backward in time*. Because the membrane decay `β` is a smooth multiplier, this path is differentiable directly (no surrogate needed); it's the `∂U[t]/∂U[t-1] = β` link.

The weights are **shared across all `T` timesteps** (the same `W` is reused each step). So, exactly as in RNN BPTT, you accumulate the per-timestep weight gradients and **sum them over time** to get the total update:

```
∂L/∂W = Σ_{t=1}^{T} (∂L/∂W)|_t
```

### Why it resembles — and costs like — RNN training

Because the graph is genuinely an unrolled recurrent net, SNN training inherits all the RNN baggage:

- **Memory cost ~ O(T):** every intermediate state (membrane potentials, spikes) over all `T` steps must be stored for the backward pass. Memory scales **linearly with T** (and with network size and batch). For an SNN with `N` neurons run for `T` steps, you store ~`O(N·T)` activations per sample. This is the dominant practical bottleneck — large `T` quickly exhausts GPU memory.
- **Compute cost ~ O(T):** forward and backward both scale linearly in `T` — and the *effective depth* of the credit-assignment problem is `depth × T`, which is why deep SNNs are so prone to vanishing/exploding gradients (§6).
- **Vanishing/exploding gradients over time:** the temporal path multiplies by `β` (and surrogate factors) at each of `T` steps, so gradients can shrink or blow up exactly as in vanilla RNNs.

Mitigations borrowed from RNN-land include **truncated BPTT** (only backprop through a window of recent timesteps; e.g. https://pmc.ncbi.nlm.nih.gov/articles/PMC10117667/) and **online approximations** like e-prop (§5) that avoid storing the whole unrolled graph. But the standard, highest-accuracy recipe today is **full BPTT + surrogate gradients**, run over a modest `T`.

> **One-line mental model:** *SNN training = RNN BPTT, where the only non-differentiable link (the spike) is patched with a surrogate derivative.* Everything else is the chain rule you already know.

Sources: https://openreview.net/pdf?id=B1xSperKvH · https://sinabs.readthedocs.io/v1.2.1/tutorials/bptt.html · https://pmc.ncbi.nlm.nih.gov/articles/PMC10117667/ · https://arxiv.org/abs/2109.12894

---

## 4. ANN-to-SNN conversion — the alternative paradigm

There is a completely different way to get a working SNN that sidesteps the non-differentiability problem entirely: **don't train the SNN at all — train a normal ANN and convert it.**

### The idea: rate equivalence

The conversion approach exploits a mathematical correspondence between a **ReLU activation** in an ANN and the **average firing rate** of an Integrate-and-Fire (IF) neuron. If you feed a constant input to a (non-leaky) IF neuron with threshold θ, its long-run firing rate is approximately proportional to the input current — a piecewise-linear, ReLU-like transfer function. So:

> A high ReLU activation value in the ANN ⇄ a high spike *rate* in the corresponding SNN neuron.

You therefore: (1) train a standard ANN with ReLUs (usually with constraints: no bias or limited bias, average pooling instead of max pooling, no batchnorm or fused-in batchnorm), then (2) copy the weights into an architecturally identical SNN of IF neurons, and (3) **calibrate the thresholds** so each layer's spike rates faithfully reproduce the ANN's activations.

### Weight / threshold normalization

The calibration step is critical. If thresholds are too high relative to the inputs, neurons under-fire (information is lost); too low, they saturate (over-fire). Two equivalent normalization strategies fix this (IJCAI 2021 conversion paper, https://www.ijcai.org/proceedings/2021/0321.pdf; ICLR 2020 hybrid paper, https://openreview.net/pdf?id=B1xSperKvH):

- **Weight normalization:** scale each layer's weights by a factor (derived from the max activation seen on training data) and set θ = 1.
- **Threshold balancing:** leave weights unchanged and set the threshold equal to that normalization factor.

The two are mathematically interchangeable. The scaling factor is often the **99.9th-percentile** ("robust") activation rather than the literal max, to avoid letting a single outlier activation crush the whole layer's dynamic range. Recent work further reduces conversion error by **calibrating offset spikes** (https://arxiv.org/pdf/2302.10685) or **converting a quantized ANN** so activations are already discrete (https://scispace.com/pdf/fast-snn-...).

### Pros and cons vs. direct surrogate-gradient training

**ANN-to-SNN conversion — pros:**
- Leverages the entire mature ANN toolchain (architectures, pretrained weights, well-understood training). Easy to reach high accuracy on large models like ResNet/VGG on ImageNet.
- No surrogate gradient, no BPTT, no `T`-scaling memory cost during training. Training is just ordinary ANN training.

**ANN-to-SNN conversion — cons:**
- **High latency.** Rate coding needs many timesteps for the average firing rate to converge to the target activation — historically hundreds to thousands of steps, though modern methods (calibration, quantized conversion, optimal threshold search) have pushed this down to tens. More steps = more energy and latency, which erodes the SNN's efficiency advantage.
- **Rate-coding only.** It throws away the SNN's ability to use *spike timing*; it only reproduces *rates*. It cannot natively learn temporal/event-driven features.
- **Conversion error.** The IF-rate ≈ ReLU equivalence is approximate (especially with leak, residual connections, or short `T`), so the converted SNN is usually slightly less accurate than the source ANN.
- **Poor on event/temporal data.** For neuromorphic sensors (DVS cameras, event audio) where the signal is inherently temporal, conversion underperforms.

**Direct surrogate-gradient + BPTT training — pros:** works at very low `T` (often 4–8 steps), can exploit temporal coding, trains end-to-end on event data, and tends to be far more energy-efficient at inference (fewer steps). **Cons:** expensive `O(T)` training, harder to tune, and historically harder to scale to very deep nets (the problems §6 solves).

### When to prefer which

- **Prefer conversion** when: you already have a strong ANN / want to reuse a big pretrained model, the task is static-image classification, accuracy at any latency is the priority, and you can tolerate higher `T`.
- **Prefer direct training** when: you need **low latency / low energy** (few timesteps), you're working with **event-based / temporal data**, or you want the network to actually *use* spike timing rather than just rates. Many systems now use a **hybrid**: convert to initialize, then fine-tune with surrogate-gradient BPTT to recover accuracy at low `T` (the ICLR 2020 hybrid approach, https://openreview.net/pdf?id=B1xSperKvH).

Sources: https://www.ijcai.org/proceedings/2021/0321.pdf · https://openreview.net/pdf?id=B1xSperKvH · https://arxiv.org/pdf/2302.10685

---

## 5. Biologically-plausible learning rules

The methods above (surrogate BPTT, conversion) are **machine-learning** approaches: they prioritize accuracy and borrow heavily from deep learning. A separate tradition asks "how could a *biological* network learn, using only locally available information?" These rules are more brain-like and more hardware-friendly (local, online), but generally **less accurate** on benchmark tasks — which is why they dominate neuromorphic/biological research but not leaderboards.

### STDP (Spike-Timing-Dependent Plasticity)

STDP is the canonical biological learning rule. A synapse's weight change depends on the **relative timing of pre- and post-synaptic spikes**:
- **Pre-before-post** (presynaptic spike causally precedes the postsynaptic one) → **potentiation** (strengthen). "Cells that fire together, in the right order, wire together."
- **Post-before-pre** (anti-causal) → **depression** (weaken).

The weight update is a function of `Δt = t_post − t_pre`, typically an exponentially decaying window. STDP is **unsupervised and purely local** — it uses only the two spike trains at that synapse, no global error signal, no backward pass. That makes it ideal for on-chip learning but means it has no direct mechanism to optimize a task loss. It's used for feature learning / clustering; reaching high supervised accuracy with pure STDP is hard. (Overview: https://www.sciencedirect.com/topics/mathematics/spike-timing-dependent-plasticity)

### R-STDP (Reward-Modulated STDP)

R-STDP adds a third factor — a global **reward/neuromodulatory signal** (dopamine-like) — that gates the STDP update. The synapse maintains an **eligibility trace** (a fading memory of recent STDP-eligible coincidences), and the actual weight change happens only when reward arrives: `Δw ∝ reward × eligibility_trace`. This turns unsupervised STDP into a **reinforcement learning** rule, letting SNNs learn from delayed, sparse rewards. It's a **three-factor learning rule** (pre-spike, post-spike, modulator). Good for RL/control on neuromorphic hardware; still weak vs. gradient methods on supervised benchmarks. (https://arxiv.org/pdf/2603.00710)

### e-prop (eligibility propagation)

**e-prop** (Bellec et al. 2020, *A solution to the learning dilemma for recurrent networks of spiking neurons*, Nature Communications — https://www.nature.com/articles/s41467-020-17236-y) is the most important bridge between biology and BPTT. Its key insight: the exact BPTT gradient for a weight **factorizes** into two parts:

```
dL/dw_ji  =  Σ_t  L_j[t]  ·  e_ji[t]
             └──────┘     └──────┘
          learning signal   eligibility trace
```

- The **eligibility trace** `e_ji[t]` captures the *local* interaction between neurons `i` and `j` and can be computed **forward in time** from local quantities only (it's a running accumulator at the synapse). This is the part that maps onto biological STDP-like traces.
- The **learning signal** `L_j[t]` is the top-down error reaching neuron `j`. Computing it *exactly* would require BPTT (propagating information backward from the future). **e-prop's approximation: ignore the future-dependent terms** and use only the instantaneous/direct error (e.g. broadcast from the output via fixed random or symmetric feedback weights).

Because of this, e-prop is **online and local**: no need to store or replay the entire unrolled graph, no backward pass through time. That makes it dramatically more memory-efficient and hardware-implementable than BPTT. The cost is that it's a *truncated/approximate* gradient, so accuracy is typically a bit below full BPTT — but it comes remarkably close, and far exceeds STDP. (Related: *Learning Precise Spike Timings with Eligibility Traces*, https://arxiv.org/pdf/2006.09988.)

### Why bio-plausible rules are less used for high accuracy

- **No exact long-range temporal credit assignment.** STDP/R-STDP have no mechanism to assign credit across many layers/timesteps for a global loss; e-prop approximates it but truncates future dependencies. BPTT computes it exactly.
- **Locality vs. optimality trade-off.** The very property that makes them biological/efficient — using only local information — is what limits their accuracy on hard supervised tasks.
- **Tooling.** Surrogate-gradient BPTT plugs directly into PyTorch/JAX autograd; bio-rules need custom machinery.

Their real payoff is **on-chip, online, low-power learning** (Loihi, SpiNNaker), not ImageNet accuracy.

Sources: https://www.nature.com/articles/s41467-020-17236-y · https://arxiv.org/pdf/2006.09988 · https://arxiv.org/pdf/2603.00710 · https://www.sciencedirect.com/topics/mathematics/spike-timing-dependent-plasticity

---

## 6. Deep-SNN training challenges & solutions

Once you commit to direct surrogate-gradient + BPTT training and try to go **deep**, a cluster of problems appears that shallow SNNs don't suffer from. This section is the "why deep SNNs were hard, and what fixed them" story.

### The compounded vanishing/exploding gradient problem

A deep SNN's effective credit-assignment depth is **layers × timesteps** (`depth × T`). Gradients must survive a product of many factors:
- per-layer surrogate factors `g(U−θ)` (each < 1 for the bump-shaped surrogates), and
- per-timestep membrane-decay factors `β` (< 1).

Multiply dozens of sub-1 (or, with a too-wide surrogate, sometimes >1) terms and gradients **vanish or explode** — and the problem is *more severe than in ANNs* because the tanh-like surrogates have bounded, often-small derivatives and there are two compounding dimensions (https://arxiv.org/pdf/2305.19725, https://arxiv.org/html/2606.11236).

### The gradient-mismatch issue

This is unique to surrogate training and worth stating plainly: **the gradient you backpropagate is not the true gradient of the spiking forward pass** — it's the gradient of a *smooth surrogate*. The forward net is hard-spiking; the backward net behaves as if it were soft. The discrepancy is **gradient mismatch**. In shallow nets it's benign, but it **compounds with depth**: small per-layer mismatches accumulate, biasing the update direction and causing under-optimized deep SNNs and accuracy collapse (https://arxiv.org/pdf/2406.19645, https://arxiv.org/html/2606.11236). The surrogate width (§2) directly controls this: wider = more mismatch but denser gradients; narrower = less mismatch but vanishing. Deep-SNN methods largely exist to keep this trade-off manageable across many layers.

### Solution 1: threshold-dependent BatchNorm (tdBN)

BatchNorm is the workhorse that lets deep ANNs train, but vanilla BN ignores the SNN's threshold and temporal dimension. **Zheng et al. (2021), *Going Deeper With Directly-Trained Larger Spiking Neural Networks* (AAAI)** — https://arxiv.org/pdf/2011.05280 / https://ojs.aaai.org/index.php/AAAI/article/view/17320 — introduced **threshold-dependent BatchNorm (tdBN)**:

- It normalizes the pre-synaptic input across **both the batch and the time dimension**, then rescales the variance to `(α·V_th)²` instead of the usual unit variance:

```
Î_c = α·V_th · (I_c − E[I_c]) / √(Var[I_c] + ε),     then  Ī_c = γ_c·Î_c + β_c
```

with `α` (default 1) tuned to prevent over-/under-firing. The key idea: **balance the input distribution against the firing threshold** so each layer's neurons fire in a healthy regime (not silent, not saturated). Combined with spatio-temporal BPTT (STBP), tdBN was the first method to **directly train SNNs up to ~50 layers** (previously direct training was stuck below ~10 layers).

### Solution 2: spiking residual blocks (SEW-ResNet, MS-ResNet)

Residual connections are *the* trick that let ANNs go to 100+ layers (identity shortcuts give gradients a clean highway). But naïvely porting ResNet to spikes fails — the shortcut/activation interaction reintroduces vanishing/exploding gradients and can break identity mapping.

- **Spiking-ResNet** (Hu et al. 2021): shortcut connects **membrane potential → after-activation spike**. Suffers gradient problems and can't cleanly represent identity.
- **SEW-ResNet — Spike-Element-Wise ResNet** (**Fang et al. 2021**, *Deep Residual Learning in Spiking Neural Networks*, NeurIPS — https://arxiv.org/pdf/2102.04159, code https://github.com/fangwei123456/Spike-Element-Wise-ResNet): the shortcut is applied **element-wise between the output spikes** of different blocks (e.g. with an ADD or IAND function). Fang et al. *prove* SEW-ResNet can implement identity mapping and **overcome vanishing/exploding gradients**, making it the **first method to directly train SNNs deeper than 100 layers** (validated on ImageNet, DVS-Gesture, CIFAR10-DVS).
- **MS-ResNet — Membrane-Shortcut ResNet** (Hu et al. 2021): the shortcut connects **membrane potential → membrane potential** across blocks. Because the residual path stays in the continuous membrane domain (it doesn't pass through the binary spike), gradients flow cleanly; this also enables very deep, often "full-spike" networks and has been pushed to extreme sparsity (e.g. *0.3 spikes per neuron*, https://www.nature.com/articles/s41467-024-51110-5).

The common theme: **route the residual/identity path so the gradient highway does not have to pass through the non-differentiable spike** (or, in SEW, so it provably preserves identity).

### Solution 3: learnable membrane time constant (PLIF)

**Fang et al. (2021), *Incorporating Learnable Membrane Time Constant to Enhance Learning of SNNs* (ICCV)** — https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf, code https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron — makes the **membrane time constant τ (equivalently the decay β) a *learnable parameter***, optimized by gradient descent alongside the weights. This is the **Parametric LIF (PLIF)** neuron.

Why it helps:
- τ is normally a hand-tuned hyperparameter that strongly affects dynamics; PLIF lets the network *learn* it, removing a fragile knob.
- τ is **shared within a layer but differs across layers**, giving each layer its own "time scale" / phase-frequency response — useful for multi-timescale temporal data.
- Empirically it reaches higher accuracy with **fewer timesteps** on both static (MNIST, CIFAR-10) and neuromorphic (N-MNIST, DVS-Gesture, CIFAR10-DVS) datasets.

### Solution 4: initialization

Initialization matters more in SNNs because a bad init immediately puts neurons in a dead (never-fires) or saturated (always-fires) regime, and §1's zero-gradient problem then prevents recovery. Practitioners scale initial weights so that the **initial membrane-potential distribution straddles the threshold** — i.e. a healthy fraction of neurons fire at moderate rates on the first forward pass. tdBN (above) effectively enforces this *during* training; good initialization enforces it *at the start*. (Discussion in https://arxiv.org/pdf/2201.02538.)

**Putting §6 together:** modern deep SNNs train successfully by combining (a) a sensibly-scaled surrogate to keep gradients flowing without too much mismatch, (b) tdBN-style normalization to keep firing regimes healthy, (c) spiking residual blocks (SEW/MS) to provide a gradient highway, (d) learnable dynamics (PLIF), and (e) careful initialization. Together these took directly-trained SNNs from <10 layers to 100+.

Sources: https://arxiv.org/pdf/2011.05280 · https://arxiv.org/pdf/2102.04159 · https://github.com/fangwei123456/Spike-Element-Wise-ResNet · https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf · https://arxiv.org/pdf/2305.19725 · https://arxiv.org/html/2606.11236 · https://www.nature.com/articles/s41467-024-51110-5

---

## 7. Practical knobs

This section collects the dials you actually turn when training an SNN, and the failure modes to watch for.

### The number of timesteps `T` — the master trade-off

`T` (how many timesteps you run per input) is the single most consequential hyperparameter. It trades off three things:

- **Accuracy:** more steps = more spikes = a richer, lower-variance estimate of rate-coded outputs → usually higher accuracy (with diminishing returns).
- **Latency:** inference can't finish until all `T` steps run → latency grows **linearly with `T`**. For real-time/edge applications you want `T` small.
- **Compute & energy & training memory:** forward/backward and BPTT memory all scale **O(T)** (§3). Energy at inference scales with total spike count, which tends to grow with `T`.

Direct surrogate training shines because it works at **small `T` (often 4–8)**, whereas converted SNNs historically needed `T` in the hundreds. A core research goal is **maximizing accuracy at minimum `T`** (PLIF, tdBN, temporal-efficient training such as https://arxiv.org/pdf/2202.11946 all target this).

### Spike-rate regularization — why you penalize spikes

Left alone, surrogate training has no reason to keep the network sparse; you can get neurons firing every timestep, which destroys the **energy efficiency** that is the entire point of SNNs (energy ∝ number of spikes on neuromorphic hardware). So you add a **spike-rate (activity) regularizer** to the loss — typically an **L1 or L2 penalty on per-neuron spike counts / firing rates** (the snnTorch paper, arXiv 2109.12894, and https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9047717/):

- **L1 on spike counts** → drives toward *spatial sparsity* (some neurons go fully silent) — fewer active neurons.
- **L2 on spike counts** → discourages *high* firing rates without killing neurons — smooths activity downward.

Two motivations: (1) **energy/sparsity** — fewer spikes = less compute/energy on event-driven hardware; (2) it can act as a **regularizer** improving generalization. The catch: over-regularize and you create **dead neurons** (next item). Modern "spike budgeting" methods make the regularization coefficient **adaptive** via feedback control toward a target spike rate — tightening on dense data, *loosening* (even encouraging more firing) on ultra-sparse data so features still propagate (https://arxiv.org/pdf/2602.12236). Networks as sparse as **~0.3 spikes/neuron** at competitive accuracy have been demonstrated (https://www.nature.com/articles/s41467-024-51110-5).

### Membrane reset between samples

Because the neuron is stateful, you must **reset the membrane potential (and spike history) to a known state between independent inputs/sequences** — otherwise the previous sample's residual potential contaminates the next, corrupting both forward outputs and BPTT gradients. There are also two **within-firing** reset mechanisms (the `−S·θ` term in §0), and the choice affects training:

- **Reset-by-subtraction (soft reset):** on firing, subtract θ from `U` (`U ← U − θ`). Preserves the supra-threshold "overshoot," so it loses less information and is the standard choice for accurate **conversion** and most direct training.
- **Reset-to-zero (hard reset):** on firing, clamp `U ← 0` (or to a fixed `U_reset`). Discards the overshoot — simpler, sometimes more biological, but lossier.

(Both are detailed in the snnTorch tutorials, https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html.)

### Optimizers and schedules that work

SNN training inherits the deep-learning optimizer toolkit, with a few tendencies:
- **Adam / AdamW** are the default and most robust for surrogate-gradient SNNs (the noisy, sometimes ill-scaled surrogate gradients benefit from adaptive per-parameter step sizes). Plain SGD+momentum works for deep conv-SNNs (ResNet-style) but is fussier.
- **Cosine-annealing or step LR schedules** plus a short **warm-up** are standard, mirroring ANN practice; warm-up is especially helpful given how sensitive early training is to firing-regime collapse.
- **Loss functions:** typically applied to the **accumulated output over time** — e.g. cross-entropy on summed/averaged output spike counts (rate readout), or on the output membrane potential, or time-to-first-spike for latency coding.
- Surrogate-specific schedules help: **warming up the surrogate width** (start wide → narrow) as discussed in §2.

### Common failure modes (and their tells)

- **Dead / silent neurons** *(the #1 failure).* Neurons that never fire → zero gradient (§1) → never recover → the layer (or whole net) stops learning, loss plateaus. **Causes:** weights/threshold mis-scaled (membrane never reaches θ), too-narrow surrogate, over-aggressive spike-rate regularization, bad init. **Fixes:** lower the threshold of at-risk layers, widen the surrogate, add/strengthen tdBN, reduce the rate-penalty coefficient, better init. *Diagnostic:* monitor per-layer firing rates during training and find where activity dies out (https://arxiv.org/pdf/2201.02538, https://arxiv.org/pdf/2109.11045).
- **Saturation / bursting / "epileptic" firing.** The opposite extreme: neurons fire every timestep. Output information collapses (everything saturates at max rate), energy explodes, and gradients can vanish. **Fixes:** raise threshold / normalize input (tdBN), add an L2 rate penalty, reduce input gain.
- **Vanishing/exploding gradients through depth×time** (§3, §6) — manifest as loss not moving (vanishing) or NaNs/divergence (exploding). **Fixes:** tdBN, spiking residuals (SEW/MS), gradient clipping, surrogate-width tuning, truncated BPTT.
- **Gradient mismatch** (§6) — deep net trains but plateaus well below expected accuracy. **Fixes:** narrower/adaptive surrogate, asymmetric/temporal-aware surrogates (https://arxiv.org/html/2606.11236).

Sources: https://arxiv.org/abs/2109.12894 · https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9047717/ · https://arxiv.org/pdf/2201.02538 · https://arxiv.org/pdf/2109.11045 · https://arxiv.org/pdf/2602.12236 · https://www.nature.com/articles/s41467-024-51110-5 · https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html

---

## 8. Putting it all together — the mental model

1. An SNN is a **recurrent network with a step-function nonlinearity**. Both properties drive training.
2. The step (spike) function has a **useless derivative** (0 almost everywhere, ∞ at threshold), so vanilla backprop sees only 0 or ∞ — it can't learn.
3. **Surrogate gradients** fix the spatial gradient: hard spike forward, smooth bump derivative backward. The **width/steepness** (α, k) trades faithfulness (vanishing) against smoothness (mismatch).
4. **BPTT** fixes the temporal gradient: unroll over `T` steps (RNN-style), share weights, sum gradients over time; pay `O(T)` in memory and compute.
5. **ANN-to-SNN conversion** is the alternative: train an ANN, normalize weights/thresholds for rate-equivalence, accept higher latency. Best for static images + reusing big models; worse for low-latency/event data.
6. **Bio-plausible rules** (STDP, R-STDP, e-prop) are local/online and hardware-friendly but trail BPTT on accuracy because they can't do exact long-range credit assignment.
7. Going **deep** needs extra machinery — **tdBN**, **spiking residuals (SEW/MS-ResNet)**, **PLIF**, careful init — to beat the compounded vanishing/exploding gradients and **gradient mismatch** over `depth × T`.
8. The practical dials are **`T`** (accuracy↔latency↔energy), **spike-rate regularization** (sparsity/energy), **membrane reset**, **Adam + warm-up/cosine**, and constant vigilance for **dead neurons and saturation**.

---

## References (papers and primary sources)

**Foundational / surrogate gradients**
- Neftci, Mostafa & Zenke (2019). *Surrogate Gradient Learning in Spiking Neural Networks.* IEEE Signal Processing Magazine. https://arxiv.org/abs/1901.09948
- Zenke & Ganguli (2018). *SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks.* Neural Computation. https://pmc.ncbi.nlm.nih.gov/articles/PMC6118408/ · https://arxiv.org/pdf/1705.11146
- Zenke & Vogels (2021). *The Remarkable Robustness of Surrogate Gradient Learning in Spiking Neural Networks.* https://research-explorer.ista.ac.at/download/8253/11131/2021_NeuralComputation_Zenke.pdf
- Eshraghian et al. (2023). *Training Spiking Neural Networks Using Lessons From Deep Learning* (snnTorch). https://arxiv.org/abs/2109.12894 · tutorials: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html · surrogate API: https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html
- SpyTorch tutorial (Zenke). https://github.com/fzenke/spytorch
- *Event-based backpropagation can compute exact gradients for SNNs.* Sci Rep (2021). https://www.nature.com/articles/s41598-021-91786-z

**BPTT / conversion**
- Rathi et al. (2020). *Enabling Deep SNNs with Hybrid Conversion and Spike Timing Dependent Backpropagation* (ICLR). https://openreview.net/pdf?id=B1xSperKvH
- Deng & Gu (2021). *Optimal ANN-SNN Conversion for Fast and Accurate Inference* (IJCAI). https://www.ijcai.org/proceedings/2021/0321.pdf
- *Bridging the Gap between ANNs and SNNs by Calibrating Offset Spikes.* https://arxiv.org/pdf/2302.10685
- *Efficient training of SNNs with temporally-truncated local BPTT.* https://pmc.ncbi.nlm.nih.gov/articles/PMC10117667/

**Biologically-plausible learning**
- Bellec et al. (2020). *A solution to the learning dilemma for recurrent networks of spiking neurons* (e-prop). Nature Communications. https://www.nature.com/articles/s41467-020-17236-y
- *Learning Precise Spike Timings with Eligibility Traces.* https://arxiv.org/pdf/2006.09988
- *Reward-Modulated Local Learning in Spiking Encoders (STDP).* https://arxiv.org/pdf/2603.00710
- *Spike-Timing-Dependent Plasticity overview.* https://www.sciencedirect.com/topics/mathematics/spike-timing-dependent-plasticity

**Deep-SNN training**
- Zheng et al. (2021). *Going Deeper With Directly-Trained Larger SNNs* (tdBN, AAAI). https://arxiv.org/pdf/2011.05280 · https://ojs.aaai.org/index.php/AAAI/article/view/17320
- Fang et al. (2021). *Deep Residual Learning in Spiking Neural Networks* (SEW-ResNet, NeurIPS). https://arxiv.org/pdf/2102.04159 · code: https://github.com/fangwei123456/Spike-Element-Wise-ResNet
- Fang et al. (2021). *Incorporating Learnable Membrane Time Constant* (PLIF, ICCV). https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf · code: https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron
- *High-performance deep SNNs with 0.3 spikes per neuron.* Nat Commun (2024). https://www.nature.com/articles/s41467-024-51110-5
- Reviews: *Direct Training High-Performance Deep SNNs* https://pmc.ncbi.nlm.nih.gov/articles/PMC11322636/ · *Direct Learning-Based Deep SNNs: A Review* https://arxiv.org/pdf/2305.19725 · *A2SG (gradient mismatch / vanishing)* https://arxiv.org/html/2606.11236

**Regularization / practical**
- *Improving Surrogate Gradient Learning via Regularization and Normalization.* https://arxiv.org/pdf/2201.02538
- *Backpropagation With Sparsity Regularization for SNN Learning.* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9047717/
- *Training Deep Spiking Auto-encoders without Bursting or Dying Neurons.* https://arxiv.org/pdf/2109.11045
- *Energy-Aware Spike Budgeting for Continual Learning.* https://arxiv.org/pdf/2602.12236
- *Temporal Efficient Training of SNN via Gradient Re-weighting.* https://arxiv.org/pdf/2202.11946
