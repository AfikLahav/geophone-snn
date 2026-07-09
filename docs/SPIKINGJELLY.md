# SpikingJelly: A Comprehensive Practical Guide

A working reference for building and training Spiking Neural Networks (SNNs) with **SpikingJelly**, the PyTorch-based SNN framework. Written to be both pedagogical (understand *why*) and practical (idiomatic code you can paste).

> **TL;DR mental model.** A spiking neuron is a stateful RNN cell whose output is a binary spike (0/1) instead of a real number. SpikingJelly gives you those neurons (`neuron`), the trick that makes the non-differentiable spike trainable (`surrogate`), spiking-aware versions of standard layers (`layer`), and the plumbing to run, reset, and accelerate the whole thing (`functional`, backends). You build a net like any `nn.Module`, run it for `T` time steps, read out the **average spike rate**, and let surrogate gradients + autograd do backprop-through-time for you.

---

## Table of Contents
1. [Origin & History](#1-origin--history)
2. [Module Structure (`activation_based`)](#2-module-structure-activation_based)
3. [The `step_mode` Concept](#3-the-step_mode-concept-single-step-vs-multi-step)
4. [Backends (`torch` vs `cupy`)](#4-backends-torch-vs-cupy)
5. [Building & Training a Model](#5-building--training-a-model-idiomatic-code)
6. [Practical Features & Gotchas](#6-practical-features--gotchas)
7. [Comparison to Other Frameworks](#7-comparison-to-other-frameworks)
8. [Sources](#sources)

---

## 1. Origin & History

### The paper
SpikingJelly is documented in:

> **W. Fang, Y. Chen, J. Ding, Z. Yu, T. Masquelier, D. Chen, L. Huang, H. Zhou, G. Li, Y. Tian.** "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence." *Science Advances* **9**(40):eadi1480 (2023). DOI: [10.1126/sciadv.adi1480](https://www.science.org/doi/10.1126/sciadv.adi1480).

Open-access copies: [arXiv:2310.16620](https://arxiv.org/abs/2310.16620) and [PMC10558124](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/).

**BibTeX:**
```bibtex
@article{doi:10.1126/sciadv.adi1480,
  author  = {Wei Fang and Yanqi Chen and Jianhao Ding and Zhaofei Yu and
             Timothee Masquelier and Ding Chen and Liwei Huang and
             Huihui Zhou and Guoqi Li and Yonghong Tian},
  title   = {SpikingJelly: An open-source machine learning infrastructure
             platform for spike-based intelligence},
  journal = {Science Advances},
  volume  = {9}, number = {40}, pages = {eadi1480}, year = {2023},
  doi     = {10.1126/sciadv.adi1480}
}
```

### Authors & lab
The project is led by **Wei Fang** with **Yonghong Tian** and **Guoqi Li** as corresponding authors. The core lab affiliations are **Peking University** (incl. the Institute for Artificial Intelligence and the Shenzhen Graduate School) and **Peng Cheng Laboratory** (Shenzhen). Co-authors span **CERCO UMR5549, CNRS–Université Toulouse 3** (Timothée Masquelier), **Institute of Automation, Chinese Academy of Sciences** (Guoqi Li), and **Shanghai Jiao Tong University**. The same group authored several of the techniques baked into the library — most notably the **Parametric LIF (PLIF)** neuron and the deep Spiking ResNet (SEW-ResNet) line of work.

### Motivation (why it exists)
Older SNN simulators fall into three buckets (paper's taxonomy):
- **Biophysical simulators** — NEURON, NEST, Brian2, GENESIS: very low-level biological detail, but **no automatic differentiation**, so no gradient-based deep learning.
- **Neuro–CS bridge** — Nengo, BindsNET: moderate neuron complexity, some GPU support.
- **Neuro–deep-learning bridge** — Norse, snnTorch, **SpikingJelly**: built on PyTorch, GPU + autodiff.

SpikingJelly's pitch is **full-stack integration**: neuromorphic dataset preprocessing, deep-SNN building blocks, surrogate-gradient *and* ANN2SNN training, biologically plausible learning rules (STDP), and neuromorphic-chip deployment — plus heavy simulation acceleration. The headline number: **deep SNN training accelerated up to 11×** (Spiking ResNet-18, T=32) and **up to ~2× faster inference** (T=128) vs other PyTorch SNN frameworks, via semi-automatically generated, fused CUDA kernels.

### Version history & the big API shift
- **First open-sourced: December 2019.** ~4 years of development by the time of the 2023 paper, by then "one of the most commonly used spiking deep-learning frameworks."
- **Versioning scheme:** four-zero style `0.0.0.0.X`. **Odd** `X` = dev builds (tracking GitHub/OpenI); **even** `X` = stable releases on PyPI. Documented lines include `0.0.0.0.4 / .6 / .8 / .10 / .12 / .14` and the current `latest`.
- **The `clock_driven` → `activation_based` rename (from `0.0.0.0.14`).** This is the single most important thing to know when reading old tutorials, Stack Overflow answers, or pre-2023 code:

  | Legacy (≤ 0.0.0.0.12) | Current |
  |---|---|
  | `spikingjelly.clock_driven` | `spikingjelly.activation_based` |
  | `spikingjelly.event_driven` | `spikingjelly.timing_based` |
  | `MultiStepLIFNode`, `MultiStep…` prefix classes | one class, set `step_mode='m'` |

  In the legacy API, single- vs multi-step was a **different class** (the `MultiStep` prefix). In the current API there is **one class per neuron** and you flip its `.step_mode` attribute (or call `functional.set_step_mode(net, 'm')`). Backward compatibility is good — usually a find-and-replace of the module path plus dropping the prefix is enough.

> **For your build:** always `from spikingjelly.activation_based import ...`. If a snippet you copied says `clock_driven` or `MultiStepLIFNode`, it's pre-2023; translate it using the table above.

(Sources: [paper/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/), [migrate_from_legacy docs](https://github.com/fangwei123456/spikingjelly/blob/master/docs/source/activation_based_en/migrate_from_legacy.rst), [PyPI](https://pypi.org/project/spikingjelly/), [Wei Fang homepage](https://fangwei123456.github.io/).)

---

## 2. Module Structure (`activation_based`)

Everything you build with lives under `spikingjelly.activation_based`. The key submodules:

| Submodule | What it gives you |
|---|---|
| `neuron` | Spiking neuron layers (stateful): `IFNode`, `LIFNode`, `ParametricLIFNode`, `QIFNode`, `EIFNode`, `IzhikevichNode`, … |
| `surrogate` | Surrogate gradient functions: `ATan`, `Sigmoid`, `PiecewiseQuadratic`, `S2NN`, `SoftSign`, etc. |
| `layer` | Spiking-aware, `step_mode`-aware wrappers of torch layers: `Linear`, `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `Dropout`, `Flatten`, `SeqToANNContainer`, `VotingLayer`, recurrent containers. |
| `functional` | Network-wide helpers: `reset_net`, `set_step_mode`, `set_backend`, `multi_step_forward`, `seq_to_ann_forward`. |
| `encoding` | Input encoders: `PoissonEncoder`, `LatencyEncoder`, `WeightedPhaseEncoder`, `GaussianTuning` (population coding). |
| `monitor` | Probes for recording spikes/voltages/gradients during fwd/bwd. |
| `learning` | Bio-plausible rules, e.g. STDP. |
| `model` | Pre-built architectures, e.g. Spiking ResNet, `parametric_lif_net`. |
| `ann2snn` | Convert a trained ReLU ANN into an SNN. |

### 2.1 `neuron` — the spiking cells

All neurons share a **three-step dynamics** (paper Eqs. 5–7), implemented as overridable methods:

1. **Charge** (`neuronal_charge`): update membrane potential `H[t] = f(V[t-1], X[t])`.
   - **IF (Integrate-and-Fire):** `H[t] = V[t-1] + X[t]` (no leak).
   - **LIF (Leaky IF):** `τ_m (V[t] − V[t-1]) = −(V[t-1] − V_reset) + X[t]`, i.e. the potential decays toward `V_reset` with time constant `τ`.
2. **Fire** (`neuronal_fire`): `S[t] = Θ(H[t] − V_threshold)` — Heaviside step → a **binary spike**. The forward uses the hard step; the backward uses the `surrogate_function`'s gradient.
3. **Reset** (`neuronal_reset`):
   - **Hard reset:** `V[t] = V_reset if S[t]==1 else H[t]`. (Default; better for directly-trained SNNs.)
   - **Soft reset:** `V[t] = H[t] − V_threshold·S[t]`. (Enabled by `v_reset=None`; preferred for ANN2SNN — lower fitting error vs ReLU.)

**Common constructor arguments** (shared across `IFNode`/`LIFNode`/`ParametricLIFNode`):

| Arg | Meaning | Typical / default |
|---|---|---|
| `tau` | Membrane time constant (LIF only). Larger = slower leak / longer memory. | `2.0` |
| `decay_input` | Whether the input `X[t]` is also scaled by the leak. Changes the exact charge equation. | `True` |
| `v_threshold` | Firing threshold. | `1.0` |
| `v_reset` | Potential after a spike. **Set to `None` for soft reset** (potential is *subtracted*, not clamped). | `0.0` |
| `surrogate_function` | Surrogate gradient used on the backward pass. | `surrogate.Sigmoid()` (use `ATan()` in practice) |
| `detach_reset` | If `True`, detach the reset term from the autograd graph. Stabilizes/cleans BPTT gradients and saves memory; widely used in deep SNNs. | `False` |
| `step_mode` | `'s'` (single-step) or `'m'` (multi-step). See §3. | `'s'` |
| `backend` | `'torch'`, `'cupy'`, or `'triton'`. Only matters for `step_mode='m'`. See §4. | `'torch'` |
| `store_v_seq` | In multi-step, also keep the **full** `[T,…]` voltage trace in `self.v_seq` (not just the last `v`). Needed to monitor membrane potential over time. | `False` |

**State** lives as a member variable (`self.v`, and `self.v_seq` if `store_v_seq=True`). This is why you **must call `reset_net`** between samples (§6) — the potential carries over otherwise.

`ParametricLIFNode` (**PLIF**, Fang et al. 2021) makes the **time constant learnable**: instead of a fixed `tau`, it learns a parameter `w` (init via `w = -log(init_tau - 1)`) and derives `τ = 1/sigmoid(w)`, **shared across neurons in a layer**. Use it when you don't want to hand-tune `tau` — it's a near drop-in for `LIFNode` (`init_tau` replaces `tau`). (Paper: ["Incorporating Learnable Membrane Time Constant…"](https://arxiv.org/pdf/2007.05785).)

```python
from spikingjelly.activation_based import neuron, surrogate

lif  = neuron.LIFNode(tau=2.0, v_threshold=1.0, v_reset=0.0,
                      surrogate_function=surrogate.ATan(),
                      detach_reset=True, step_mode='m', backend='cupy')

iff  = neuron.IFNode(surrogate_function=surrogate.ATan())          # no leak
plif = neuron.ParametricLIFNode(init_tau=2.0,                       # learnable tau
                                surrogate_function=surrogate.ATan())
```

### 2.2 `surrogate` — making the spike trainable

The spike is `Θ(x)` (Heaviside). Its true derivative is `+∞` at 0 and `0` everywhere else — gradients can't flow. The **surrogate gradient** trick: keep the hard step on the **forward** pass, but on the **backward** pass pretend the derivative is that of a smooth approximation. SpikingJelly implements this as a `torch.autograd.Function` so autograd handles it transparently.

A surrogate has a shape and a sharpness knob, usually **`alpha`**. Larger `alpha` → steeper, more step-like surrogate (gradient concentrated near threshold); smaller → smoother, gradient spread out.

- **`surrogate.ATan(alpha=2.0)`** — the de-facto default in practice.
  Forward (approx.) `≈ (1/π)·arctan(π·α·x/2) + 1/2`; backward `∂S/∂x = (α/2) / (1 + (π·α·x/2)²)`.
- **`surrogate.Sigmoid(alpha=4.0)`** — backward `= α·σ(αx)·(1−σ(αx))` where `σ` is the logistic sigmoid.
- Others: `PiecewiseQuadratic`, `PiecewiseExp`, `SoftSign`, `S2NN`, `QPseudoSpike`, `LeakyKReLU`, etc.

```python
from spikingjelly.activation_based import surrogate
sg = surrogate.ATan(alpha=2.0)   # pass to any neuron's surrogate_function=...
```
**Practical default:** `ATan()`. Tune `alpha` only if training is unstable (lower it) or under-fitting near threshold (raise it).

### 2.3 `layer` — why spiking-aware wrappers exist

You *could* use `torch.nn.Linear`. SpikingJelly provides `layer.Linear`, `layer.Conv2d`, `layer.BatchNorm2d`, `layer.MaxPool2d`, `layer.Dropout`, `layer.Flatten`, etc. for two reasons:

1. **`step_mode` awareness.** In multi-step mode the data tensor is `[T, B, …]`. A spiking-aware layer knows to treat the leading `T` axis correctly — for *stateless* ops (Linear/Conv) it folds `T` into the batch (`[T,B,…] → [T·B,…]`), runs **one** big parallel matmul/conv, then unfolds. That's a major speedup (see §3–§4). A plain `nn.Linear` would choke on the extra time axis.
2. **Correct stateful semantics.** Dropout must use the **same mask across all `T` steps** (otherwise you inject noise that breaks temporal credit assignment); `layer.Dropout` does this. BatchNorm needs time-aware statistics.

Other useful members:
- **`layer.SeqToANNContainer`** — wrap any stateless `nn.Module`(s) so they process a `[T,B,…]` sequence by merging `T` into the batch dim. (This is the mechanism behind the multi-step speedup.)
- **`layer.VotingLayer`** — the **readout** for population coding: assign several output neurons per class and average their votes (e.g. 10 neurons/class → average-pool groups of 10) to get one logit per class. More robust than a single neuron per class.
- Recurrent containers (`LinearRecurrentContainer`, `ElementWiseRecurrentContainer`) for explicit recurrence.

### 2.4 `functional` — network-wide operations

- **`functional.reset_net(net)`** — walks the module tree and calls `.reset()` on every module that has one, zeroing all neuron states. **Call this after every sample/batch.** (See [Discussion #175](https://github.com/fangwei123456/spikingjelly/discussions/175).)
- **`functional.set_step_mode(net, 'm')`** — flip an entire network to multi-step (or `'s'`) in one call, instead of touching each layer. *Caveat:* it won't recurse into neurons inside recurrent containers like `LinearRecurrentContainer` — set those manually.
- **`functional.set_backend(net, 'cupy', instance=neuron.BaseNode)`** — switch all (multi-step) neurons to the CuPy backend.
- **`functional.multi_step_forward(x_seq, module)`** — runs a single-step module over a `[T,…]` sequence with a time loop, concatenating outputs. (`seq_to_ann_forward` is the stateless-layer equivalent.)

### 2.5 `encoding` — turning real data into spikes
- **`encoding.PoissonEncoder()`** — treats each input value in `[0,1]` as a firing probability; on every call emits a fresh Bernoulli spike (rate coding). Call it **once per time step** inside the `T` loop.
- **`encoding.LatencyEncoder(T)`** — stronger inputs spike *earlier* (temporal coding).
- **`encoding.WeightedPhaseEncoder`**, **`GaussianTuning`** (population coding for continuous values, used in the RL example).

> **Note:** you don't *always* need an encoder. A common pattern ("direct/constant input encoding") is to feed the **same** real-valued input every step and let the first spiking layer do the encoding — often trains better than Poisson for static data.

(Sources: [neuron docs](https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.neuron.html), [README](https://github.com/fangwei123456/spikingjelly/blob/master/README.md), [paper/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/).)

---

## 3. The `step_mode` Concept: single-step vs multi-step

This is the central performance/usability concept. Every SpikingJelly module has a `step_mode` attribute.

### Single-step (`step_mode='s'`) — "step-by-step"
- A module's `forward(x)` processes **one** time step: input `X[t]` shaped `[B, …]` → output `Y[t]` shaped `[B, …]`.
- **You** write the `for t in range(T)` loop and feed steps one at a time.
- Internally the whole network is advanced one tick at a time (depth-first over layers, then next tick).
- **Memory: O(N)** — independent of `T`, because you don't keep every step's activations of every layer alive simultaneously by default. Great for **inference** and **very large `T`** (e.g. ANN2SNN). Also required for genuine step-coupled recurrence.

### Multi-step (`step_mode='m'`) — "layer-by-layer"
- A module's `forward(x_seq)` processes the **entire** sequence at once: input shaped `[T, B, …]` → output `[T, B, …]`.
- The framework runs the net **layer-by-layer** (breadth-first): each layer consumes all `T` steps before moving to the next layer.
- **Why it's faster:**
  - **Stateless layers** (Linear/Conv/BN) reshape `[T,B,…] → [T·B,…]` and run **one** kernel over all steps in **parallel** — no Python time loop, full GPU utilization.
  - **Stateful neurons** unroll their recurrence inside a **single fused CUDA kernel** (with the `cupy` backend), eliminating per-step kernel-launch overhead. The paper shows naive per-step ("RAW") training time grows ~quadratically with `T`, while fused multi-step stays ~linear → up to **11× faster** training.
- **Memory: O(T·N)** — you hold all `T` steps' activations for BPTT. The trade-off for speed.

### Switching the whole net
```python
from spikingjelly.activation_based import functional
functional.set_step_mode(net, step_mode='m')   # entire net → multi-step
```
Then feed a `[T, B, …]` tensor and get a `[T, B, …]` output in one forward call (no manual time loop). Switch back with `'s'` for memory-lean inference.

**Rule of thumb:** **train in `'m'`** (speed), **deploy/large-T in `'s'`** (memory). The two produce numerically equivalent results for feed-forward nets.

(Sources: [paper/PMC propagation patterns](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/), [migrate_from_legacy](https://github.com/fangwei123456/spikingjelly/blob/master/docs/source/activation_based_en/migrate_from_legacy.rst), [CUDA+LBL docs](https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/11_cext_neuron_with_lbl.html).)

---

## 4. Backends (`torch` vs `cupy`)

For **multi-step neurons only**, you pick how the neuron's recurrence is computed:

| Backend | What it is | When to use |
|---|---|---|
| `'torch'` | Pure PyTorch ops; the time loop / dynamics expressed in Python+ATen. | **Default. Coding, debugging, CPU, small models.** Portable, readable, works everywhere. |
| `'cupy'` | A **fused CUDA kernel** (one big kernel for the whole `[T,…]` unroll), generated semi-automatically and JIT-compiled/cached by [CuPy](https://cupy.dev/). | **GPU training of deep SNNs / large `T` / big batches.** Eliminates the overhead of launching many tiny per-step kernels. This is the source of the **11× training / ~2× inference** speedups. |
| `'triton'` | Triton-generated kernels (newer alternative to CuPy). | GPU acceleration without a CuPy install; experimental/alternative. |

**How the CuPy speedup works (paper):** SpikingJelly uses **semi-automatic CUDA code generation**. A base neuron CUDA-kernel template has "vacant" slots; a new neuron only supplies its charge equation and the two local derivatives (e.g. for IF: `H[t]=V[t-1]+X[t]`, `dH[t+1]/dV[t]=1`, `dH[t]/dX[t]=1`). The framework fills in the surrogate + backward code, **fuses many small ops into one large kernel**, CuPy compiles and **caches** the binary, and forward/backward both run fused. This is also what makes it cheap to **accelerate your own custom neuron** ("multilevel inheritance + semi-automatic code generation").

```python
from spikingjelly.activation_based import functional, neuron
functional.set_step_mode(net, 'm')                         # cupy needs multi-step
functional.set_backend(net, 'cupy', instance=neuron.BaseNode)
```
**Notes & gotchas:** `cupy` is **GPU-only** and requires a matching CUDA/CuPy install (`pip install cupy-cuda12x` for CUDA 12). The first run pays a one-time JIT-compile cost (then it's cached). Develop on `'torch'`, switch to `'cupy'` for the real training run.

(Sources: [paper/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/), [neuron docs](https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.neuron.html), [README](https://github.com/fangwei123456/spikingjelly/blob/master/README.md).)

---

## 5. Building & Training a Model (idiomatic code)

### Install
```bash
pip install spikingjelly            # stable, from PyPI
# from source (latest dev):
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly && pip install .
# optional GPU acceleration:
pip install cupy-cuda12x            # match your CUDA version
```

### 5.1 The minimal network (from the README)
```python
from torch import nn
from spikingjelly.activation_based import layer, neuron, surrogate

net = nn.Sequential(
    layer.Flatten(),
    layer.Linear(28 * 28, 10, bias=False),
    neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan()),
)
```
Use `layer.*` (not `torch.nn.*`) so the net is `step_mode`-aware.

### 5.2 The training-loop pattern (the part people get wrong)
The canonical recipe, distilled from the official `lif_fc_mnist.py` example:

```python
import torch, torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional, encoding

class SNN(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.layer = nn.Sequential(
            layer.Flatten(),
            layer.Linear(28 * 28, 10, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )
    def forward(self, x):
        return self.layer(x)

net = SNN(tau=2.0).cuda()
encoder   = encoding.PoissonEncoder()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
T = 20   # number of time steps

for img, label in train_loader:                 # img in [0,1], shape [B,1,28,28]
    img, label = img.cuda(), label.cuda()
    label_onehot = F.one_hot(label, 10).float()

    optimizer.zero_grad()

    # --- run T time steps, accumulate output spikes ---
    out_fr = 0.
    for t in range(T):
        encoded = encoder(img)                  # fresh Poisson spikes each step
        out_fr += net(encoded)                  # sum spikes over time
    out_fr = out_fr / T                          # -> firing RATE in [0,1], shape [B,10]

    # --- rate-coded readout: train the rate to match the one-hot target ---
    loss = F.mse_loss(out_fr, label_onehot)
    loss.backward()                              # surrogate grads flow through the spikes
    optimizer.step()

    # --- CRITICAL: clear neuron state before the next sample ---
    functional.reset_net(net)
```

**Why each piece:**
- **`for t in range(T)`** — an SNN's output is a *spike train*; one forward isn't enough, you integrate over `T`.
- **`out_fr += net(...)` then `/= T`** — the readout is the **average firing rate** per output neuron (rate coding). This real-valued vector is what the loss sees.
- **`F.mse_loss(out_fr, label_onehot)`** — push the correct class's neuron toward rate 1 and others toward 0. (Cross-entropy on `out_fr` also works and often trains a touch faster.)
- **`loss.backward()`** — autograd does **backprop-through-time**; at each spike the surrogate function supplies the gradient. **You write zero special backward code.**
- **`functional.reset_net(net)`** — neuron potentials are stateful members; without this, leftover charge from sample *i* contaminates sample *i+1*. **This is the #1 bug.**

### 5.3 Faster version (multi-step + CuPy, no Python time loop)
```python
functional.set_step_mode(net, 'm')                          # whole net -> multi-step
functional.set_backend(net, 'cupy', instance=neuron.BaseNode)

for img, label in train_loader:
    img, label = img.cuda(), label.cuda()
    # build a [T, B, ...] sequence. For static images, repeat across T (direct encoding):
    x_seq = img.unsqueeze(0).repeat(T, 1, 1, 1, 1)          # [T, B, 1, 28, 28]
    optimizer.zero_grad()

    y_seq  = net(x_seq)                                      # ONE call -> [T, B, 10]
    out_fr = y_seq.mean(0)                                   # average over time -> [B, 10]

    loss = F.cross_entropy(out_fr, label)
    loss.backward(); optimizer.step()
    functional.reset_net(net)
```
Same math, far fewer kernel launches → big speedup on GPU. Note the readout is `y_seq.mean(0)` (mean over the `T` axis).

### 5.4 A small conv SNN (more representative)
```python
net = nn.Sequential(
    layer.Conv2d(2, 16, 3, padding=1, bias=False),          # 2 = DVS polarity channels
    layer.BatchNorm2d(16),
    neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
    layer.MaxPool2d(2),

    layer.Conv2d(16, 32, 3, padding=1, bias=False),
    layer.BatchNorm2d(32),
    neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
    layer.MaxPool2d(2),

    layer.Flatten(),
    layer.Linear(32 * 8 * 8, 10 * 10, bias=False),
    neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
    layer.VotingLayer(10),                                   # 10 neurons/class -> 10 logits
)
functional.set_step_mode(net, 'm')
```

(Sources: [README minimal example & install](https://github.com/fangwei123456/spikingjelly/blob/master/README.md); [`lif_fc_mnist.py` example](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples) — Poisson encoder, `out_fr/T`, `mse_loss`, `reset_net`; [reset_net Discussion #175](https://github.com/fangwei123456/spikingjelly/discussions/175).)

---

## 6. Practical Features & Gotchas

### Monitors — see what your spikes are doing
SpikingJelly ships **five general-purpose monitors** that act like probes inserted into all layers of a given type.

```python
from spikingjelly.activation_based import monitor, neuron

# record the output spikes of every LIFNode in the net:
spk_mon = monitor.OutputMonitor(net, instance=neuron.LIFNode)

# record membrane potential over time (needs store_v_seq=True on the neurons):
v_mon   = monitor.AttributeMonitor('v_seq', pre_forward=False,
                                   net=net, instance=neuron.LIFNode)

# (optional) record gradients flowing back into neurons:
g_mon   = monitor.GradOutputMonitor(net, instance=neuron.LIFNode)

with torch.no_grad():
    net(x_seq)
print(spk_mon.records)            # list, one entry per monitored module
spk_mon.clear_recorded_data()     # reset between batches
spk_mon.remove_hooks()            # when done
```
The five types: **`OutputMonitor`**, **`InputMonitor`**, **`AttributeMonitor`** (any member, e.g. `v`/`v_seq`), **`GradInputMonitor`**, **`GradOutputMonitor`**. Pass a `function_on_…` transform to compute on the fly — e.g. give `OutputMonitor` a transform that returns `S.mean()` to log **firing rate** per layer (a key health metric — neurons that never fire or always fire are dead/saturated). (See [monitor docs / paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/), [Issue #631](https://github.com/fangwei123456/spikingjelly/issues/631).)

### The readout / voting layer
For classification, the readout is **spike rate**, not the last value. Two common choices:
- **1 neuron/class** + average over `T` (the minimal example).
- **N neurons/class** + `layer.VotingLayer(N)` (population coding). Each output spike is a vote; average the group → one logit per class. More robust, the standard for DVS-Gesture-style tasks.

### Common mistakes (ranked by how often they bite)
1. **Forgetting `functional.reset_net(net)`.** State leaks across samples → garbage/erratic accuracy. Reset **every** batch (after `optimizer.step()`, or before the next forward). *Reset is not differentiable; never expect to backprop across it.*
2. **`step_mode` / tensor-shape mismatch.** In `'s'` mode you must run a manual `for t` loop and feed `[B,…]`; in `'m'` mode you feed `[T,B,…]` once. Feeding a `[T,B,…]` tensor to a `'s'`-mode net (or vice-versa) gives shape errors or silently wrong results. Use `functional.set_step_mode(net,'m')` to flip the **whole** net consistently — and remember it skips neurons inside recurrent containers.
3. **`T` too small.** With too few time steps there aren't enough spikes for a meaningful rate → the net can't learn / accuracy is low. Start around **T=4–32** depending on task; raise `T` if under-fitting, lower it to save memory/compute. (Static images can need fewer steps with direct encoding; neuromorphic/event data often needs more.)
4. **Multi-step memory blow-up.** `'m'` keeps **O(T·N)** activations for BPTT. Large `T` × large model can OOM. Fixes: smaller `T`/batch, gradient checkpointing, or switch to `'s'` for inference.
5. **Mixing `torch.nn.*` with multi-step.** Use `layer.*` wrappers; plain `nn.Linear`/`nn.Conv2d` aren't `[T,B,…]`-aware. (Or wrap them in `layer.SeqToANNContainer`.)
6. **Dropout that re-samples per step.** Use `layer.Dropout` (shared mask across `T`), not `nn.Dropout`.
7. **CuPy without GPU/correct CUDA.** `backend='cupy'` errors on CPU or with a mismatched CuPy build. Develop on `'torch'`; switch for the GPU run.
8. **Expecting `tau` to update with `LIFNode`.** `tau` is a fixed hyperparameter in `LIFNode`. If you want a learnable time constant, use `ParametricLIFNode` (`init_tau=…`).

### Other handy features
- **`store_v_seq=True`** on neurons to inspect the whole voltage trajectory (needed by `AttributeMonitor('v_seq')`).
- **`detach_reset=True`** is a cheap, common stabilizer for deep SNNs — turn it on by default for conv stacks.
- **AMP / mixed precision** works (the official examples use `torch.cuda.amp`); combine with CuPy multi-step for max throughput.
- **`ann2snn`** lets you train a normal ReLU CNN and convert it (use **soft reset**, `v_reset=None`), useful when surrogate training is finicky.

---

## 7. Comparison to Other Frameworks

All of Norse, snnTorch, and SpikingJelly are **PyTorch-based with GPU + autodiff**; the differences are emphasis.

| Framework | Built on | Emphasis | Strengths | Weaknesses |
|---|---|---|---|---|
| **SpikingJelly** | PyTorch (+CuPy/Triton) | **Performance + full-stack** | Fastest training (fused CUDA kernels, **~11×**); neuromorphic datasets *and* chip deployment (Lava/Loihi, Lynxi, NIR); surrogate **and** ANN2SNN; STDP; big model zoo (Spiking ResNet, SEW-ResNet). | Larger API surface; the `clock_driven`→`activation_based` rename means stale tutorials abound; CuPy adds an install dependency. |
| **snnTorch** | PyTorch | **Education & ease** | Most widely used; excellent docs/tutorials; gentle learning curve; clean surrogate-gradient API. | Less raw speed; lighter on neuromorphic-chip deployment and large-scale tooling. |
| **Norse** | PyTorch | **Functional / composable** | JAX-style functional design (state separated from computation); rich set of biologically plausible neuron/synapse models; composable. | Functional style is less familiar; smaller community; performance below SpikingJelly's fused kernels. |
| **BindsNET** | PyTorch | **Bio learning rules / RL** | IF/LIF + Hebbian, STDP, reward-modulated STDP; good for ML/RL with local learning; CPU/GPU. | Not focused on deep gradient-trained SNNs / surrogate gradients; smaller for large-scale supervised SNNs. |
| **Lava** | Custom (Intel) | **Neuromorphic HW co-design** | First-class **Intel Loihi 2** deployment; hardware–software co-design; event-driven. | Steeper, hardware-oriented programming model; less of a general PyTorch deep-learning training tool (though Lava-DL exists). |

**Why SpikingJelly is a strong default for a deep-SNN project:**
- **Speed at scale.** Independent SNN-framework benchmarks consistently put **SpikingJelly with the CuPy backend** at or near the top for training/forward-backward throughput; the paper's 11× figure is the headline.
- **Full pipeline in one place.** Encode → build deep SNN → train (surrogate or ANN2SNN) → monitor → deploy to neuromorphic hardware, without stitching libraries together.
- **You won't outgrow it.** Pre-built deep architectures + semi-automatic CUDA generation mean even custom neurons get accelerated cheaply.
- **Active, well-cited, widely adopted** since 2019.

**When another tool might fit better:** **snnTorch** if you prioritize the gentlest learning curve and tutorials; **Norse** if you want a functional/composable API and broad neuron-model variety; **BindsNET** for STDP/Hebbian-driven or RL research; **Lava** if your endpoint is specifically Intel Loihi silicon.

(Sources: [paper/PMC framework taxonomy & Fig. 1E](https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/); SNN framework benchmark commentary (Open Neuromorphic, [arXiv:2408.00280 temporal-fusion](https://arxiv.org/pdf/2408.00280)); [snnTorch surrogate docs](https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html).)

---

## Sources
**Paper**
- Science Advances (paywalled landing): https://www.science.org/doi/10.1126/sciadv.adi1480
- arXiv preprint: https://arxiv.org/abs/2310.16620
- PMC full open-access HTML: https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/

**Code & docs**
- GitHub repo: https://github.com/fangwei123456/spikingjelly
- README (install, minimal example, BibTeX): https://github.com/fangwei123456/spikingjelly/blob/master/README.md
- Migration guide (`clock_driven`→`activation_based`): https://github.com/fangwei123456/spikingjelly/blob/master/docs/source/activation_based_en/migrate_from_legacy.rst
- Read the Docs (English tutorials + zh/en API): https://spikingjelly.readthedocs.io/
- `neuron` API: https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.neuron.html
- CUDA + layer-by-layer acceleration tutorial: https://spikingjelly.readthedocs.io/zh-cn/latest/activation_based_en/11_cext_neuron_with_lbl.html
- MNIST example dir (`lif_fc_mnist.py`): https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples
- PyPI (versions): https://pypi.org/project/spikingjelly/
- `reset_net` Discussion #175: https://github.com/fangwei123456/spikingjelly/discussions/175
- Monitor usage Issue #631: https://github.com/fangwei123456/spikingjelly/issues/631

**Related papers / context**
- PLIF (ParametricLIFNode): https://arxiv.org/pdf/2007.05785
- Wei Fang homepage: https://fangwei123456.github.io/
- snnTorch surrogate docs (comparison): https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html

*Compiled June 2026. Verify exact signatures against the version you `pip install`, since the `0.0.0.0.X` line evolves.*
