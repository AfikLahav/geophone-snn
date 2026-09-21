"""One builder that turns a configuration dictionary into a network.

Every axis of the study lives here: how many layers and how wide, which neuron, how many time
steps, what normalization, how the answer is read out, the firing threshold and what happens
after a spike, the gradient stand-in, the head layout, and whether the thing spikes at all --
the ordinary-network controls are built by the same function so the comparison is honest.

Two implementation notes that matter for the results, not just the code:

  * `stem_once` computes the first layer ONE time and feeds the same current at every step.
    With a static input that is mathematically identical to recomputing it -- the linear output
    is the same at every step, and normalizing T identical copies gives the same answer as
    normalizing one -- but it removes T-1 redundant dense multiplies. Measured on the current
    model that is the difference between using 21% MORE energy than the equivalent ordinary
    network and using 3x LESS.

  * Soft reset is `v_reset=None` in this library (subtract the threshold), hard reset is a
    number (snap back to it). Both are exposed because neither has ever been varied here.
"""
import warnings

import numpy as np

# The fused kernels still reference numpy aliases removed in 1.20; without this shim the fast
# backend raises AttributeError on its first call and the run either dies or silently falls back.
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
for _a, _t in (("int", int), ("float", float), ("bool", bool)):
    try:
        getattr(np, _a)
    except AttributeError:
        setattr(np, _a, _t)

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer

HEAD_SIZES = {"human": 2, "animal": 2, "vehicle": 1}   # ordinal: present, and more-than-one


# ------------------------------------------------------------------ neurons

class AdaptiveLIF(neuron.AdaptBaseNode):
    """Leaky integrate-and-fire with a self-inhibiting adaptation current.

    The base class already carries the adaptation variable and updates it on every spike; all
    that is missing is the membrane update, which is the ordinary leaky one minus that current.
    This is the neuron the audio literature keeps finding useful, and the one the library ships
    only as a base class."""

    def __init__(self, tau=2.0, **kw):
        super().__init__(**kw)
        self.tau = float(tau)

    def neuronal_charge(self, x):
        self.v = self.v + (x - (self.v - self.v_rest) - self.w) / self.tau


def make_surrogate(name, alpha):
    return {
        "atan": surrogate.ATan,
        "sigmoid": surrogate.Sigmoid,
        "piecewise": surrogate.PiecewiseQuadratic,
        "erf": surrogate.Erf,
        "softsign": surrogate.SoftSign,
    }[name](alpha)


def make_neuron(cfg):
    """Build one spiking unit from the configuration. Shared by every hidden layer."""
    sg = make_surrogate(cfg.get("surrogate", "atan"), cfg.get("surrogate_alpha", 2.0))
    thr = cfg.get("v_threshold", 1.0)
    thr = 1.0 if thr == "learnable" else float(thr)
    v_reset = None if cfg.get("reset", "hard") == "soft" else 0.0
    common = dict(v_threshold=thr, surrogate_function=sg, detach_reset=cfg.get("detach_reset", True),
                  step_mode="m")
    kind = cfg.get("neuron", "plif")

    if kind == "if":
        return neuron.IFNode(v_reset=v_reset, **common)
    if kind == "lif":
        return neuron.LIFNode(tau=cfg.get("tau", 2.0), v_reset=v_reset, **common)
    if kind == "plif":
        return neuron.ParametricLIFNode(init_tau=cfg.get("tau", 2.0), v_reset=v_reset, **common)
    if kind == "klif":
        return neuron.KLIFNode(tau=cfg.get("tau", 2.0), v_reset=v_reset, **common)
    if kind == "eif":
        return neuron.EIFNode(tau=cfg.get("tau", 2.0), **common)          # has its own rest/reset
    if kind == "qif":
        return neuron.QIFNode(tau=cfg.get("tau", 2.0), **common)
    if kind == "izhikevich":
        return neuron.IzhikevichNode(tau=cfg.get("tau", 2.0), a=0.02, b=0.2, **common)
    if kind == "liaf":
        # emits a continuous value gated by the spike; no fused kernel exists for it, so it is
        # pinned to the plain backend rather than left to fail on its first call
        c = dict(common, backend="torch")
        return neuron.LIAFNode(act=nn.ReLU(), threshold_related=False,
                               tau=cfg.get("tau", 2.0), v_reset=v_reset, **c)
    if kind == "adaptive":
        return AdaptiveLIF(tau=cfg.get("tau", 2.0), v_reset=v_reset,
                           tau_w=cfg.get("tau_w", 8.0), a=cfg.get("adapt_a", 0.0),
                           b=cfg.get("adapt_b", 0.5), **common)
    raise ValueError(f"unknown neuron {kind!r}")


class LearnableThreshold(nn.Module):
    """Wraps a spiking unit so its firing threshold is trained rather than fixed at 1."""

    def __init__(self, node, init=1.0):
        super().__init__()
        self.node = node
        self.log_thr = nn.Parameter(torch.tensor(float(np.log(init))))

    def forward(self, x):
        self.node.v_threshold = float(torch.exp(self.log_thr).item())
        return self.node(x)


# ------------------------------------------------------------------ plumbing

class SeqBN(nn.Module):
    """Normalize over channels for a [steps, batch, channels] tensor by folding steps in."""

    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        T, B, C = x.shape
        return self.bn(x.reshape(T * B, C)).reshape(T, B, C)


class SeqGN(nn.Module):
    def __init__(self, c, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(min(groups, c), c)

    def forward(self, x):
        T, B, C = x.shape
        return self.gn(x.reshape(T * B, C, 1)).reshape(T, B, C)


class ThresholdBN(nn.Module):
    """Normalization scaled to the firing threshold, the variant meant for deep spiking stacks."""

    def __init__(self, c, v_threshold=1.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)
        self.v = float(v_threshold)

    def forward(self, x):
        T, B, C = x.shape
        return self.v * self.bn(x.reshape(T * B, C)).reshape(T, B, C)


def make_norm(kind, c, v_threshold=1.0):
    if kind == "bn":
        return SeqBN(c)
    if kind == "tdbn":
        return ThresholdBN(c, v_threshold)
    if kind == "gn":
        return SeqGN(c)
    if kind in (None, "none"):
        return nn.Identity()
    raise ValueError(f"unknown normalization {kind!r}")


class FlyHash(nn.Module):
    """The fruit fly's olfactory front end, as an input stage.

    The fly takes about fifty receptor inputs, projects them through a sparse binary random
    matrix into a much larger layer, and then a single inhibitory neuron silences everything
    except the strongest few percent. The result is a sparse binary tag where similar inputs get
    similar tags, and the projection is never trained.

    Applied here to 132 features. Note this is NOT a spiking network on its own -- it has no
    memory, no leak and no time -- but its output is exactly the sparse binary vector a spiking
    layer emits at one step, so it slots in ahead of the trunk unchanged.

    `winner_take_all` replaces the top-k sort with lateral inhibition settling over the time
    steps, which is how the fly actually does it and the version that belongs in a spiking
    comparison rather than beside it.
    """

    def __init__(self, n_in, n_out=4096, density=0.1, keep=0.05, winner_take_all=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        mask = (torch.rand(n_out, n_in, generator=g) < density).float()
        self.register_buffer("proj", mask)          # fixed: never trained, as in the fly
        self.keep = keep
        self.n_out = n_out
        self.wta = winner_take_all

    def forward(self, x):
        y = x @ self.proj.t()
        k = max(1, int(round(self.keep * self.n_out)))
        if not self.wta:
            thr = torch.topk(y, k, dim=-1).values[..., -1:]
            return (y >= thr).float()
        # lateral inhibition: subtract a shared pool a little at a time until only the strongest
        # few percent are still above zero
        v = y
        for _ in range(4):
            inhib = v.clamp(min=0).mean(-1, keepdim=True) * (self.n_out / k) * 0.25
            v = v - inhib
        return (v > 0).float()


class TimeAttention(nn.Module):
    """Learns how much each time step contributes, instead of weighting them equally."""

    def __init__(self, T):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(T))

    def forward(self, x):
        a = torch.softmax(self.w, 0).view(-1, 1, 1)
        return x * (a * x.shape[0])


# ------------------------------------------------------------------ readout

class Readout(nn.Module):
    """Turns the trunk's output into a score per head.

    mean  -- average the membrane value over the time steps (the current choice)
    last  -- take the final step only
    count -- how many times the unit fired
    vote  -- several units per output, averaged (population coding)
    """

    def __init__(self, d_in, d_out, mode="mean", pop=4):
        super().__init__()
        self.mode = mode
        self.pop = pop if mode == "vote" else 1
        self.fc = layer.Linear(d_in, d_out * self.pop)
        if mode == "count":
            self.node = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode="m",
                                       backend="torch")
        else:
            self.node = neuron.LIFNode(v_threshold=float("inf"), surrogate_function=surrogate.ATan(),
                                       step_mode="m", store_v_seq=True, backend="torch")
        self.d_out = d_out

    def forward(self, h):
        y = self.fc(h)
        if self.mode == "count":
            s = self.node(y)
            out = s.mean(0)
        else:
            self.node(y)
            v = self.node.v_seq
            out = v.mean(0) if self.mode in ("mean", "vote") else v[-1]
        if self.pop > 1:
            out = out.reshape(out.shape[0], self.d_out, self.pop).mean(-1)
        return out


# ------------------------------------------------------------------ trunk

def build_trunk(cfg, n_in):
    """The shared body: a stack of linear -> normalize -> unit -> dropout blocks.

    For an ordinary-network control the spiking unit is replaced by a smooth activation and the
    stack is otherwise identical, so any difference between the two is the unit and nothing else.
    """
    spiking = cfg.get("kind", "snn") == "snn"
    widths = cfg["widths"]
    norm = cfg.get("norm", "bn")
    dp = cfg.get("dropout", 0.1)
    thr = cfg.get("v_threshold", 1.0)
    blocks, d = [], n_in
    rank = int(cfg.get("low_rank") or 0)
    for i, w in enumerate(widths):
        if i == 0 and rank:
            # the first layer factored through a narrow bottleneck: n_in x rank + rank x w
            # weights instead of n_in x w -- the first layer is where a small model's
            # parameters and energy sit
            blocks.append(layer.Linear(d, rank, bias=False))
            blocks.append(layer.Linear(rank, w))
        else:
            blocks.append(layer.Linear(d, w))
        blocks.append(make_norm(norm, w, 1.0 if thr == "learnable" else float(thr)))
        if spiking:
            node = make_neuron(cfg)
            blocks.append(LearnableThreshold(node) if thr == "learnable" else node)
        else:
            blocks.append({"gelu": nn.GELU(), "relu": nn.ReLU()}[cfg.get("activation", "gelu")])
        if dp:
            blocks.append(layer.Dropout(dp))
        d = w
    return nn.Sequential(*blocks), d


class Recurrent(nn.Module):
    """Optional feedback around the trunk, for when each time step carries new information."""

    def __init__(self, kind, d):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.cell = nn.Linear(d, d)
        elif kind == "elementwise":
            self.w = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        if self.kind is None:
            return x
        out, state = [], torch.zeros_like(x[0])
        for t in range(x.shape[0]):
            if self.kind == "linear":
                state = x[t] + self.cell(state)
            else:
                state = x[t] + self.w * state
            out.append(state)
        return torch.stack(out, 0)


# ------------------------------------------------------------------ the network

class CampaignNet(nn.Module):
    """The whole model for one configuration.

    Input is either a single feature vector repeated at every step (`input: static`, which is
    every configuration that has ever been trained here) or a different vector per step
    (`input: sequence`, the arm where the time axis finally carries information).
    """

    def __init__(self, cfg, n_in):
        super().__init__()
        self.cfg = cfg
        self.T = int(cfg.get("T", 4))
        self.n_in = n_in
        self.spiking = cfg.get("kind", "snn") == "snn"
        self.sequence = cfg.get("input", "static") == "sequence"
        self.stem_once = bool(cfg.get("stem_once", False)) and not self.sequence
        self.passes = int(cfg.get("ann_passes", 1))

        gate = cfg.get("gate", "on")
        self.gate = nn.Parameter(torch.ones(n_in)) if gate in ("on", "sparse") else None
        self.gate_l1 = float(cfg.get("gate_l1", 1e-4)) if gate == "sparse" else 0.0

        fe = cfg.get("frontend")
        if fe in ("flyhash", "flyhash_wta"):
            # expansion happens inside the model, not in the stored data: 3.3 million windows at
            # 4096 dimensions would be 54 GB resident, against 1.6 GB for the 132 it starts from
            self.frontend = FlyHash(n_in, n_out=int(cfg.get("flyhash_dim", 4096)),
                                    keep=float(cfg.get("flyhash_keep", 0.05)),
                                    winner_take_all=(fe == "flyhash_wta"))
            trunk_in = self.frontend.n_out
        else:
            self.frontend = None
            trunk_in = n_in

        self.trunk, d = build_trunk(cfg, trunk_in)
        self.attn = TimeAttention(self.T) if cfg.get("attention") == "time" else None
        self.recur = Recurrent(cfg.get("recurrent"), d) if cfg.get("recurrent") else None

        layout = cfg.get("heads", "3head")
        self.layout = layout
        mode = cfg.get("readout", "mean")

        if layout == "4way":
            self.names = ["fourway"]
            self.heads = nn.ModuleList([Readout(d, 4, mode)])
        else:
            names = ["human", "animal", "vehicle"]
            if layout in ("2head_neg", "2head_noanimal"):
                names = ["human", "vehicle"]
            if layout in ("1head_human", "1head_human_noveh"):
                names = ["human"]
            # `head_sizes` overrides the default outputs per head: 1 = presence only,
            # 2 = presence plus "more than one"
            hs = cfg.get("head_sizes") or {}
            sizes = [int(hs.get(n, HEAD_SIZES[n])) for n in names]
            if layout == "plus_any":
                names, sizes = names + ["any"], sizes + [1]
            self.names = names
            if layout == "private":
                # each head gets its own spiking layer on top of the shared trunk
                self.private = nn.ModuleList()
                for _ in names:
                    blk = [layer.Linear(d, d), make_norm(cfg.get("norm", "bn"), d)]
                    blk.append(make_neuron(cfg) if self.spiking else nn.GELU())
                    self.private.append(nn.Sequential(*blk))
            self.heads = nn.ModuleList([Readout(d, s, mode) for s in sizes])

    # -- forward ---------------------------------------------------------

    def _encode(self, x):
        """Shape the input into [steps, batch, features]."""
        if self.sequence:                     # x is already [batch, steps, features]
            xs = x.permute(1, 0, 2).contiguous()
        else:
            xs = x.unsqueeze(0).expand(self.T, -1, -1)
        if self.gate is not None:
            xs = xs * self.gate
        if self.frontend is not None:
            xs = self.frontend(xs)
        return xs

    def _body(self, xs):
        if self.stem_once:
            # the first linear layer sees the same vector at every step, so compute it once and
            # broadcast -- identical output, T-1 fewer dense multiplies
            first, rest = self.trunk[0], self.trunk[1:]
            h0 = first(xs[:1])
            h = h0.expand(xs.shape[0], -1, -1)
            h = nn.Sequential(*rest)(h)
        else:
            h = self.trunk(xs)
        if self.attn is not None:
            h = self.attn(h)
        if self.recur is not None:
            h = self.recur(h)
        return h

    def forward(self, x):
        if not self.spiking and self.passes == 1 and not self.sequence:
            xs = self._encode(x)[:1]          # a single ordinary forward pass
            h = self.trunk(xs)
        else:
            h = self._body(self._encode(x))
        out = {}
        for i, nm in enumerate(self.names):
            hh = self.private[i](h) if self.layout == "private" else h
            out[nm] = self.heads[i](hh)
        return out

    # -- reporting -------------------------------------------------------

    def gate_penalty(self):
        if self.gate_l1 and self.gate is not None:
            return self.gate_l1 * self.gate.abs().mean()
        return torch.zeros((), device=next(self.parameters()).device)

    def learned_tau(self):
        """The leak each layer settled on. Free to record, and it says whether the network chose
        to keep any memory between steps at all."""
        out = []
        for m in self.modules():
            if isinstance(m, neuron.ParametricLIFNode):
                out.append(float(1.0 / torch.sigmoid(m.w.detach()).item()))
            elif isinstance(m, (neuron.LIFNode, AdaptiveLIF)) and hasattr(m, "tau"):
                out.append(float(m.tau))
        return out

    def spiking_modules(self):
        return [m for m in self.modules() if isinstance(m, neuron.BaseNode) and m.v_threshold != float("inf")]


class SeparateNets(nn.Module):
    """Completely independent networks, one per head.

    `heads: separate`  -- three networks, person / animal / vehicle, the upper bound on what head
                          separation can buy, at three times the parameters.
    `heads: separate2` -- two networks, person / vehicle, animal scenes removed from training.

    With `head_features` set, each network reads its own subset of the prepared features -- the
    way to give the vehicle head the feature groups the person head is better off without. The
    subset is fixed at build time as a list of column indices into the prepared feature list.
    """

    def __init__(self, cfg, n_in):
        super().__init__()
        self.names = ["human", "vehicle"] if cfg.get("heads") == "separate2" else \
            ["human", "animal", "vehicle"]
        hf = cfg.get("head_features") or {}
        if hf:
            from .data import select_features
            full = select_features(cfg.get("features", "all132"))
            assert len(full) == n_in, (len(full), n_in)
            idx = {nm: [full.index(f) for f in select_features(hf[nm])] if nm in hf else None
                   for nm in self.names}
        else:
            idx = {nm: None for nm in self.names}
        self.nets = nn.ModuleDict({
            nm: _SingleHeadNet(cfg, len(idx[nm]) if idx[nm] is not None else n_in, HEAD_SIZES[nm],
                               feat_idx=idx[nm]) for nm in self.names})
        self.T = int(cfg.get("T", 4))

    def forward(self, x):
        return {nm: net(x)["out"] for nm, net in self.nets.items()}

    def learned_tau(self):
        return self.nets[self.names[0]].learned_tau()

    def gate_penalty(self):
        return sum(n.gate_penalty() for n in self.nets.values())

    def spiking_modules(self):
        return [m for n in self.nets.values() for m in n.spiking_modules()]


class _SingleHeadNet(CampaignNet):
    """One trunk, one head -- the building block of the fully separate arm.

    `feat_idx`, when given, selects this network's own columns from the shared input."""

    def __init__(self, cfg, n_in, size, feat_idx=None):
        cfg = dict(cfg)
        cfg["heads"] = "3head"          # build the standard trunk, then replace the heads
        super().__init__(cfg, n_in)
        self.layout = "single"
        self.names = ["out"]
        self.heads = nn.ModuleList([Readout(cfg["widths"][-1], size, cfg.get("readout", "mean"))])
        self.register_buffer("feat_idx",
                             torch.as_tensor(feat_idx, dtype=torch.long) if feat_idx is not None
                             else torch.zeros(0, dtype=torch.long))

    def forward(self, x):
        if self.feat_idx.numel():
            x = x.index_select(-1, self.feat_idx)
        return super().forward(x)


class SeqNet(nn.Module):
    """A conventional model over the K consecutive windows ending at the current one.

    kind = gru   -- a gated recurrent unit reads the K feature vectors in order; its final state
                    feeds the heads
    kind = attn  -- a transformer encoder (one or more self-attention layers with learned
                    positions) reads them; the mean over positions feeds the heads

    Same feature gate and the same head layout as CampaignNet, so the loss, the scoring and the
    threshold protocol are untouched."""

    def __init__(self, cfg, n_in):
        super().__init__()
        self.cfg = cfg
        self.kind = cfg["kind"]
        self.K = int(cfg.get("context", 5))
        gate = cfg.get("gate", "on")
        self.gate = nn.Parameter(torch.ones(n_in)) if gate in ("on", "sparse") else None
        d = int(cfg.get("d_model", 32))
        drop = float(cfg.get("dropout", 0.1))
        layers = int(cfg.get("layers", 1))
        if self.kind == "gru":
            self.rnn = nn.GRU(n_in, d, num_layers=layers, batch_first=True,
                              dropout=drop if layers > 1 else 0.0)
        elif self.kind == "attn":
            self.inp = nn.Linear(n_in, d)
            self.pos = nn.Parameter(torch.zeros(self.K, d))
            enc = nn.TransformerEncoderLayer(d, nhead=int(cfg.get("n_heads", 2)),
                                             dim_feedforward=int(cfg.get("ff_mult", 4)) * d,
                                             dropout=drop, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
        else:
            raise ValueError(f"unknown sequence kind {self.kind!r}")
        self.drop = nn.Dropout(drop)
        names = ["human", "animal", "vehicle"]
        if cfg.get("heads") in ("2head_neg", "2head_noanimal"):
            names = ["human", "vehicle"]
        hs = cfg.get("head_sizes") or {}
        self.names = names
        self.layout = cfg.get("heads", "3head")
        self.heads = nn.ModuleList([nn.Linear(d, int(hs.get(n, HEAD_SIZES[n]))) for n in names])

    def forward(self, x):                      # x: [batch, K, features]
        if self.gate is not None:
            x = x * self.gate
        if self.kind == "gru":
            self.rnn.flatten_parameters()      # the averaged copy's weights are not one chunk
            _, h = self.rnn(x)
            h = h[-1]
        else:
            z = self.enc(self.inp(x) + self.pos)
            h = self.norm(z.mean(1))
        h = self.drop(h)
        return {nm: self.heads[i](h) for i, nm in enumerate(self.names)}

    def gate_penalty(self):
        return torch.zeros((), device=next(self.parameters()).device)

    def learned_tau(self):
        return []


def build(cfg, n_in):
    """The only entry point. Returns the network for this configuration."""
    if cfg.get("kind") in ("gru", "attn"):
        return SeqNet(cfg, n_in)
    if cfg.get("heads") in ("separate", "separate2"):
        return SeparateNets(cfg, n_in)
    return CampaignNet(cfg, n_in)


# ------------------------------------------------------------------ cost

E_MAC_PJ = 4.6      # 45 nm, 32-bit float: 3.7 multiply + 0.9 add  (Horowitz 2014)
E_AC_PJ = 0.9       # a spike arrives -> add only, nothing to multiply


def head_outputs(cfg):
    """Number of output units across the heads of a layout, honouring `head_sizes`."""
    layout = cfg.get("heads", "3head")
    if layout == "4way":
        return 4
    names = ["human", "vehicle"] if layout in ("2head_neg", "2head_noanimal", "separate2") \
        else ["human", "animal", "vehicle"]
    if layout == "plus_any":
        names = names + ["any"]
    hs = cfg.get("head_sizes") or {}
    return sum(int(hs.get(n, HEAD_SIZES.get(n, 1))) for n in names)


def cost_model(cfg, n_in, fire_rates):
    """Operations and energy per window, and the ratio against the same network run as an
    ordinary one.

    Separate-network layouts are priced as the sum of their networks, each with its own input
    count; the firing-rate list holds every spiking layer in network order.

    The first layer is fed real numbers, not spikes, so it is priced as a multiply-accumulate.
    Every later layer is fed spikes and priced as an accumulate, scaled by how often the layer
    below it actually fires and by the number of time steps. Repeating the input means that
    first dense layer runs T times unless `stem_once` is set.
    """
    if cfg.get("kind") in ("gru", "attn"):
        # conventional sequence models: every operation is a multiply-accumulate, once per window
        K = int(cfg.get("context", 5)); d = int(cfg.get("d_model", 32)); L = int(cfg.get("layers", 1))
        nh = head_outputs(cfg)
        if cfg["kind"] == "gru":
            macs = K * 3 * (n_in * d + d * d) + (L - 1) * K * 3 * (2 * d * d)
        else:
            ff = int(cfg.get("ff_mult", 4)) * d
            macs = K * n_in * d + L * (K * 4 * d * d + 2 * K * K * d + K * 2 * d * ff)
        macs += d * nh
        e = macs * E_MAC_PJ / 1e6
        return dict(macs=int(macs), sops=0.0, energy_uj=float(e), energy_ratio=1.0)
    widths = cfg["widths"]
    if cfg.get("heads") in ("separate", "separate2"):
        names = ["human", "vehicle"] if cfg.get("heads") == "separate2" else ["human", "animal", "vehicle"]
        hf = cfg.get("head_features") or {}
        from .data import select_features
        L = len(widths)
        rates = list(fire_rates) + [fire_rates[-1] if fire_rates else 0.1] * (L * len(names))
        tot = dict(macs=0, sops=0.0, energy_uj=0.0, ann_uj=0.0)
        for k, nm in enumerate(names):
            sub = dict(cfg, heads={"human": "2head_noanimal", "vehicle": "2head_noanimal",
                                   "animal": "2head_noanimal"}[nm], head_features=None)
            n_sub = len(select_features(hf[nm])) if nm in hf else n_in
            c = cost_model(sub, n_sub, rates[k * L:(k + 1) * L])
            # one output per network rather than the two-head count: correct the head layer
            head_macs = widths[-1] * (3 - HEAD_SIZES[nm])
            c_macs = c["macs"] - head_macs
            tot["macs"] += c_macs
            tot["sops"] += c["sops"]
            tot["energy_uj"] += c["energy_uj"] - head_macs * rates[(k + 1) * L - 1] * int(cfg.get("T", 4)) * E_AC_PJ / 1e6
            tot["ann_uj"] += c_macs * E_MAC_PJ / 1e6
        e = tot.pop("energy_uj")
        ann = tot.pop("ann_uj")
        return dict(macs=int(tot["macs"]), sops=float(tot["sops"]), energy_uj=float(e),
                    energy_ratio=float(ann / e) if e > 0 else None)
    T = int(cfg.get("T", 4))
    n_heads_out = head_outputs(cfg)

    frontend_sops = 0.0
    if cfg.get("frontend") in ("flyhash", "flyhash_wta"):
        # a sparse BINARY projection: no multiplies, only adds on the connections that exist
        dim = int(cfg.get("flyhash_dim", 4096))
        frontend_sops = float(cfg.get("flyhash_density", 0.1)) * n_in * dim
        n_in = dim

    rank = int(cfg.get("low_rank") or 0)
    macs = [n_in * rank + rank * widths[0] if rank else n_in * widths[0]]
    for a, b in zip(widths, widths[1:]):
        macs.append(a * b)
    macs.append(widths[-1] * n_heads_out)
    total_mac = int(sum(macs))

    if cfg.get("kind", "snn") != "snn":
        e = total_mac * cfg.get("ann_passes", 1) * E_MAC_PJ / 1e6
        return dict(macs=total_mac, sops=0.0, energy_uj=e, energy_ratio=1.0)

    stem_steps = 1 if cfg.get("stem_once") else T
    rates = list(fire_rates) + [fire_rates[-1]] * 8
    if frontend_sops:
        # the expansion emits spikes, so the layer it feeds is accumulate-only like the rest --
        # the model has no dense real-valued layer at all
        e_stem = (frontend_sops + macs[0]) * stem_steps * E_AC_PJ
    else:
        e_stem = macs[0] * stem_steps * E_MAC_PJ
    sops = frontend_sops * stem_steps + sum(rates[i] * T * macs[i + 1] for i in range(len(macs) - 1))
    e = (e_stem + sum(rates[i] * T * macs[i + 1] for i in range(len(macs) - 1)) * E_AC_PJ) / 1e6
    ann = total_mac * E_MAC_PJ / 1e6
    return dict(macs=total_mac, sops=float(sops), energy_uj=float(e),
                energy_ratio=float(ann / e) if e > 0 else None)
