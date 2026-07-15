# -*- coding: utf-8 -*-
"""nnsight_engine.py — NNsight execution engine for the E1/E2 Table-1 interventions.

Why this exists
---------------
The four E1/E2 harnesses (``intervention_analysis{,_opt,_neo,_qwen}.py``) do not use the
HuggingFace forward pass. They hand-reimplement the transformer — reading raw weight
tensors off the modules (``attn.c_attn.weight.t()``) and recomputing attention from
scratch — and run the real forward only once per sentence, purely to fire two embedding
hooks whose output is discarded.

This module provides the alternative: run the **true** HF forward under NNsight and apply
each intervention as an activation edit inside ``.trace()``, reading attention
probabilities straight out of the model's own eager attention. The manual path is retained
as the reference implementation (``--engine manual``, still the default); this engine is
``--engine nnsight``. ``verify_parity`` cross-checks the two, which is what makes the
manual reimplementation auditable rather than merely trusted.

Contract
--------
:meth:`NNsightEngine.run_all` returns ``{intervention_key: [per-layer tensors]}`` with each
tensor ``[num_heads, seq, seq]`` on CPU — byte-for-byte the shape the manual
``run_all_interventions`` returns, so every downstream consumer
(``compute_bos_attention_metric``, the stats rows, the CSV/TXT writers) is unchanged.

Expressing weight edits as activation edits
-------------------------------------------
NNsight intervenes on module ``.input``/``.output``, not on weights. Interventions (b),
(i) and (j) are weight-space in the manual code, so they are re-expressed algebraically —
exactly, not approximately:

  (b) ``q = x @ Wq^T + bq``  ⇒  zeroing ``bq`` == subtracting ``bq`` from the query slice
      of the projection output.

  (i)/(j) ``k = x @ Wk^T + bk``. Zeroing the input-columns ``S`` of ``Wk`` gives
      ``x_masked @ Wk^T + bk == k - x[..., S] @ Wk[:, S]^T``.
      For GPT-2's fused Conv1D, ``Wk[:, S]^T`` is ``c_attn.weight[S, H:2H]``.

Both are verified against real weight edits by the smoke test (deviation ~1e-8).

NNsight 0.7 constraints this module is built around
---------------------------------------------------
* **Envoy accesses must follow execution order.** Within a block that is
  ``pre_ln -> (q/k proj) -> attn -> mlp``; touching ``mlp.output`` before reading
  ``attn.output`` raises ``MissedProviderError``. The trace body below is ordered
  accordingly and must stay that way.
* **``attn_implementation="eager"`` is load-bearing**, not vestigial. Under ``sdpa`` the
  eager attention path is skipped and ``attn.output[1]`` is ``None``. Asserted at init.
* **An explicit ``attention_mask`` must accompany ``position_ids``.** ``masking_utils``
  treats any position-id step != 1 as a packed-sequence boundary and ANDs a block-diagonal
  mask into the causal mask; Qwen's ``(h)`` (all-zero position ids) would otherwise
  degenerate to an identity mask, silently. We always pass an explicit mask.
* The trace body is captured by **source inspection** and AST-transformed, so it must live
  in a real source file (it does) and cannot be built dynamically.
"""

import functools
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from datasets_loader import DEFAULT_SEED

# ═══════════════════════════════════════════════════════════════════════════════
# Architecture spec
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ArchSpec:
    """Declarative description of one architecture's module tree and semantics.

    Paths are dotted and resolved with ``getattr`` chains, so the same engine drives all
    four families without per-architecture forks.
    """

    name: str                       # "gpt2" | "opt" | "neo" | "qwen"

    # --- module-tree navigation ---
    blocks_path: str                # decoder block list, relative to the model root
    pre_ln_path: str                # rel. to block; .output is the projection input `x`
    attn_path: str                  # rel. to block; .output[1] is the attention probs
    mlp_out_path: str               # rel. to block; .output is the FFN result
    wte_path: str                   # token-embedding module
    wpe_path: Optional[str]         # additive positional embedding; None for RoPE models

    # --- q/k edit surface ---
    qkv_layout: str                 # "fused" (GPT-2 Conv1D c_attn) | "separate"
    qkv_path: Optional[str] = None  # rel. to attn; "c_attn" when fused
    q_path: Optional[str] = None    # rel. to attn; "q_proj" when separate
    k_path: Optional[str] = None    # rel. to attn; "k_proj" when separate

    # --- semantics ---
    mlp_out_layout: str = "BSH"     # "BSH" | "NH" (OPT flattens to [B*S, H] before fc1)
    positional: str = "additive"    # "additive" | "rope_ids" (Qwen)
    needs_output_attentions: bool = False   # OPT nulls attn_weights unless asked
    massive_scope: str = "per_model"        # "per_model" | "per_sentence" (Qwen)
    dtype_mode: str = "post_load_to"        # "post_load_to" | "from_pretrained" (Qwen)


ARCH_SPECS: Dict[str, ArchSpec] = {
    # GPT-2: fused Conv1D QKV; probs unconditionally returned by GPT2Attention.
    "gpt2": ArchSpec(
        name="gpt2",
        blocks_path="transformer.h",
        pre_ln_path="ln_1",
        attn_path="attn",
        mlp_out_path="mlp",
        wte_path="transformer.wte",
        wpe_path="transformer.wpe",
        qkv_layout="fused",
        qkv_path="c_attn",
    ),
    # OPT: separate projections; fc2 emits [B*S, H] because OPTDecoderLayer reshapes
    # before the FFN; OPTAttention returns attn_weights=None unless output_attentions.
    "opt": ArchSpec(
        name="opt",
        blocks_path="model.decoder.layers",
        pre_ln_path="self_attn_layer_norm",
        attn_path="self_attn",
        mlp_out_path="fc2",
        wte_path="model.decoder.embed_tokens",
        wpe_path="model.decoder.embed_positions",
        qkv_layout="separate",
        q_path="q_proj",
        k_path="k_proj",
        mlp_out_layout="NH",
        needs_output_attentions=True,
    ),
    # GPT-Neo: GPTNeoAttention is a thin wrapper that returns the inner tuple unchanged,
    # so the outer `attn` envoy still exposes .output[1]. q_proj.bias is None -> (b) no-op.
    "neo": ArchSpec(
        name="neo",
        blocks_path="transformer.h",
        pre_ln_path="ln_1",
        attn_path="attn",
        mlp_out_path="mlp",
        wte_path="transformer.wte",
        wpe_path="transformer.wpe",
        qkv_layout="separate",
        q_path="attention.q_proj",
        k_path="attention.k_proj",
    ),
    # Qwen2.5: RoPE — no additive PE module, so positional interventions are position_ids
    # manipulations passed straight into the forward. Massive coords are per-sentence.
    "qwen": ArchSpec(
        name="qwen",
        blocks_path="model.layers",
        pre_ln_path="input_layernorm",
        attn_path="self_attn",
        mlp_out_path="mlp",
        wte_path="model.embed_tokens",
        wpe_path=None,
        qkv_layout="separate",
        q_path="q_proj",
        k_path="k_proj",
        positional="rope_ids",
        massive_scope="per_sentence",
        dtype_mode="from_pretrained",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Intervention plans
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EditPlan:
    """One intervention, expressed as the set of edits to apply during a trace."""

    key: str
    zero_bq: bool = False               # (b)
    pe_edit: Optional[str] = None       # "remove_first" | "zero_all"   (additive archs)
    te_edit: Optional[str] = None       # "zero_first"                  (f)
    pos_ids: Optional[str] = None       # "remove_first"|"swap01"|"zero_all"  (rope archs)
    mlp_edit: Optional[str] = None      # "zero_all" | "swap_epe" | "swap_pe"
    wk_zero: Optional[str] = None       # "massive" | "random"


def build_edit_plans(spec: ArchSpec) -> Dict[str, EditPlan]:
    """Per-architecture intervention table, keyed exactly like each harness's registry.

    The positional interventions (c)/(d)/(e)/(h) diverge by architecture family:

    * ``additive`` (GPT-2/OPT/Neo) — edit the positional-embedding module output, and for
      the swaps, edit the layer-0 MLP output (the manual ``modify_mlp_fn`` hook).
    * ``rope_ids`` (Qwen) — there is no additive PE to edit, so the same semantics are
      enacted on RoPE position ids, mirroring ``intervention_analysis_qwen.py``. Note (e)
      is deliberately identical to (d) there: RoPE has a single notion of position, so the
      raw/effective PE distinction collapses. Kept as a separate row to preserve Table-1
      column alignment.
    """
    plans = {
        "int_a": EditPlan("int_a"),
        "int_b": EditPlan("int_b", zero_bq=True),
        "int_f": EditPlan("int_f", te_edit="zero_first"),
        "int_g": EditPlan("int_g", mlp_edit="zero_all"),
        "int_i": EditPlan("int_i", wk_zero="massive"),
        "int_j": EditPlan("int_j", wk_zero="random"),
    }
    if spec.positional == "rope_ids":
        plans["int_c"] = EditPlan("int_c", pos_ids="remove_first")
        plans["int_d"] = EditPlan("int_d", pos_ids="swap01")
        plans["int_e"] = EditPlan("int_e", pos_ids="swap01")   # (e) == (d) under RoPE
        plans["int_h"] = EditPlan("int_h", pos_ids="zero_all")
    else:
        plans["int_c"] = EditPlan("int_c", pe_edit="remove_first")
        plans["int_d"] = EditPlan("int_d", mlp_edit="swap_epe")
        plans["int_e"] = EditPlan("int_e", mlp_edit="swap_pe")
        plans["int_h"] = EditPlan("int_h", pe_edit="zero_all")
    return plans


# Canonical order — matches the INTERVENTIONS registry in every harness.
INTERVENTION_ORDER = ["int_a", "int_b", "int_c", "int_d", "int_e",
                      "int_f", "int_g", "int_h", "int_i", "int_j"]


# ═══════════════════════════════════════════════════════════════════════════════
# (j) random-control determinism
# ═══════════════════════════════════════════════════════════════════════════════


def draw_random_wk_columns(num_layers: int, k: int, hidden_size: int,
                           seed: int = DEFAULT_SEED) -> List[List[int]]:
    """Replay the manual harness's per-layer random Wk column draw, exactly.

    The manual path seeds the *global* ``random`` module once per sentence
    (``random.seed(DEFAULT_SEED)``) and then draws ``k`` indices per layer, in layer order,
    inside ``manual_self_attention_new``. ``random.Random(seed)`` seeds an independent
    Mersenne Twister with the identical state, so replaying the same call sequence
    reproduces the identical draw without disturbing global RNG state.

    Two details are load-bearing:

    * ``hidden_size`` must match the manual path's ``hidden_states.size(-1)``. ``randint``
      consumes a *width-dependent* number of raw bits, so a wrong width desynchronises the
      stream from the very first draw.
    * Duplicates are **preserved** here (the manual path can and does draw the same column
      twice). They are de-duplicated at the edit site instead — see
      :meth:`NNsightEngine._wk_correction`.

    Returns ``num_layers`` lists of ``k`` indices.
    """
    rng = random.Random(seed)
    return [[rng.randint(0, hidden_size - 1) for _ in range(k)]
            for _ in range(num_layers)]


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve(root: Any, path: str) -> Any:
    """Resolve a dotted attribute path (``"model.decoder.layers"``)."""
    return functools.reduce(getattr, path.split("."), root)


def load_nnsight_model(spec: ArchSpec, model_name: str, *, dtype, tokenizer,
                       device=None, remote: bool = False, guard: Optional[Callable] = None):
    """Wrap ``model_name`` in an NNsight ``LanguageModel``, preserving harness semantics.

    ``guard(hf_model, model_name)`` is the harness's own structural check (e.g. OPT's
    post-LayerNorm rejection, Qwen's ``q_proj.bias`` requirement); it runs against the real
    HF module tree at ``lm._model``.

    ``tokenizer`` is passed through rather than letting NNsight build its own: the GPT-2
    harness uses the *slow* ``GPT2Tokenizer``, and the dataset sampler filters/truncates by
    token count, so a different tokenizer would silently change which sentences are
    evaluated and make any manual-vs-nnsight comparison meaningless.

    ``device_map`` is deliberately never passed — it installs accelerate hooks that
    conflict with the harnesses' explicit ``.to(device)`` and can leave modules on meta.
    """
    from nnsight import LanguageModel

    kwargs = {"attn_implementation": "eager"}
    if spec.dtype_mode == "from_pretrained":
        kwargs["torch_dtype"] = dtype

    lm = LanguageModel(model_name, tokenizer=tokenizer, dispatch=not remote, **kwargs)

    if guard is not None:
        guard(lm._model, model_name)

    if not remote:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lm._model.to(device)
        if spec.dtype_mode == "post_load_to":
            lm._model.to(dtype)
        lm._model.eval()

    return lm


# ═══════════════════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════════════════


class NNsightEngine:
    """Runs the E1/E2 interventions on the real HF forward via NNsight tracing."""

    def __init__(self, lm, spec: ArchSpec, *, remote: bool = False):
        self.lm = lm
        self.spec = spec
        self.remote = remote
        self.hf = lm._model
        self.plans = build_edit_plans(spec)

        impl = getattr(self.hf.config, "_attn_implementation", None)
        if impl != "eager":
            raise RuntimeError(
                f"NNsight engine requires attn_implementation='eager' (got {impl!r}). "
                "Under sdpa/flash the eager attention path is skipped and the attention "
                "probabilities are None."
            )

        self.blocks = _resolve(self.hf, spec.blocks_path)
        self.num_layers = len(self.blocks)
        cfg = self.hf.config
        self.hidden = int(getattr(cfg, "n_embd", None) or getattr(cfg, "hidden_size", 0))

    # --- envoy accessors (mirror the HF tree through the NNsight proxy) -----------

    def _blocks_envoy(self):
        return _resolve(self.lm, self.spec.blocks_path)

    def _block(self, i):
        return self._blocks_envoy()[i]

    def _attn(self, block):
        return _resolve(block, self.spec.attn_path)

    def _hf_attn(self, i):
        return _resolve(self.blocks[i], self.spec.attn_path)

    # --- precomputation (outside the trace) ---------------------------------------

    def _q_bias(self, i):
        """The query-bias vector to subtract for (b), or None when the arch has none."""
        attn = self._hf_attn(i)
        if self.spec.qkv_layout == "fused":
            bias = _resolve(attn, self.spec.qkv_path).bias
            return None if bias is None else bias[:self.hidden].detach()
        bias = _resolve(attn, self.spec.q_path).bias
        # GPT-Neo: None -> (b) is a structural no-op, matching the manual path.
        return None if bias is None else bias.detach()

    def _wk_correction(self, i, cols: Sequence[int]):
        """``corr`` such that ``k_slice -= x[..., S] @ corr`` == zeroing ``Wk[:, S]``.

        ``cols`` is de-duplicated here: zeroing a weight column twice is idempotent, but
        subtracting its contribution twice is not. The manual (j) control draws with
        replacement, so duplicates genuinely occur.
        """
        S = sorted(set(int(c) for c in cols))
        attn = self._hf_attn(i)
        H = self.hidden
        if self.spec.qkv_layout == "fused":
            # c_attn.weight is Conv1D [H, 3H]; wk == weight[:, H:2H].t(), so
            # wk[:, S].T == weight[S, H:2H].
            corr = _resolve(attn, self.spec.qkv_path).weight[S, H:2 * H]
        else:
            corr = _resolve(attn, self.spec.k_path).weight[:, S].T
        return S, corr.detach()

    def _swap_directions(self, kind: str, inputs) -> Optional[tuple]:
        """Unit vectors for the (d)/(e) component swap, from the harness's own EPE/PE.

        Supplied by the caller via ``swap_dirs`` in practice; kept here for the
        self-contained smoke path.
        """
        return None

    # --- payload -------------------------------------------------------------------

    def _position_ids(self, plan: EditPlan, seq_len: int, device):
        """RoPE position ids for the Qwen positional interventions.

        Mirrors ``intervention_analysis_qwen.py`` exactly: (c) gives token 0 the position
        of token 1; (d)/(e) swap positions 0 and 1; (h) sets every position to 0 so the
        rotation is the identity.
        """
        pos = torch.arange(seq_len, device=device)
        if plan.pos_ids == "remove_first" and seq_len > 1:
            pos = pos.clone()
            pos[0] = pos[1].clone()
        elif plan.pos_ids == "swap01" and seq_len > 1:
            pos = pos.clone()
            pos[0], pos[1] = pos[1].clone(), pos[0].clone()
        elif plan.pos_ids == "zero_all":
            pos = torch.zeros(seq_len, dtype=torch.long, device=device)
        return pos.unsqueeze(0)

    def _payload(self, plan: EditPlan, inputs) -> dict:
        ids = inputs["input_ids"]
        device = ids.device
        seq_len = ids.shape[-1]

        mask = inputs.get("attention_mask", None) if hasattr(inputs, "get") else None
        if mask is None:
            mask = torch.ones_like(ids)

        # An explicit attention_mask is mandatory, not defensive: masking_utils only runs
        # its packed-sequence detection when attention_mask is None, and every positional
        # intervention below uses non-monotonic position ids that would trip it.
        payload = {"input_ids": ids, "attention_mask": mask}

        if self.spec.positional == "rope_ids":
            payload["position_ids"] = self._position_ids(plan, seq_len, device)
        if self.spec.needs_output_attentions:
            payload["output_attentions"] = True
        return payload

    # --- the trace -----------------------------------------------------------------

    def run_intervention(self, plan: EditPlan, inputs, *,
                         massive_coords: Optional[Sequence[int]] = None,
                         random_columns: Optional[List[List[int]]] = None,
                         swap_dirs: Optional[tuple] = None) -> List[torch.Tensor]:
        """Run one intervention and return per-layer ``[num_heads, seq, seq]`` probs (CPU).

        Everything that does not depend on activations (bias slices, Wk corrections, swap
        directions) is precomputed here so the trace body stays a straight line.
        """
        spec = self.spec
        H = self.hidden
        L = self.num_layers
        payload = self._payload(plan, inputs)

        # --- precompute per-layer edit operands ---
        q_bias = [self._q_bias(i) for i in range(L)] if plan.zero_bq else None

        wk_ops = None
        if plan.wk_zero == "massive":
            if massive_coords is None:
                raise ValueError("intervention (i) requires massive_coords")
            wk_ops = [self._wk_correction(i, massive_coords) for i in range(L)]
        elif plan.wk_zero == "random":
            if random_columns is None:
                raise ValueError("intervention (j) requires random_columns")
            wk_ops = [self._wk_correction(i, random_columns[i]) for i in range(L)]

        saved: List[Any] = []

        with self.lm.trace(payload, remote=self.remote):
            # 1. Embeddings — wte executes before wpe in every additive arch.
            if plan.te_edit == "zero_first":
                _resolve(self.lm, spec.wte_path).output[0, 0] = 0
            if plan.pe_edit is not None and spec.wpe_path is not None:
                wpe = _resolve(self.lm, spec.wpe_path)
                if plan.pe_edit == "remove_first":
                    wpe.output[0, 0] = wpe.output[0, 1]
                elif plan.pe_edit == "zero_all":
                    wpe.output[:] = 0

            # 2. Layers. Access order below is the block's execution order
            #    (pre_ln -> q/k proj -> attn -> mlp); reordering raises MissedProviderError.
            for i in range(L):
                block = self._block(i)
                attn = self._attn(block)

                x = _resolve(block, spec.pre_ln_path).output

                if plan.zero_bq and q_bias[i] is not None:
                    if spec.qkv_layout == "fused":
                        _resolve(attn, spec.qkv_path).output[..., :H] -= q_bias[i]
                    else:
                        _resolve(attn, spec.q_path).output[...] -= q_bias[i]

                if wk_ops is not None:
                    S, corr = wk_ops[i]
                    delta = x[..., S] @ corr
                    if spec.qkv_layout == "fused":
                        _resolve(attn, spec.qkv_path).output[..., H:2 * H] -= delta
                    else:
                        _resolve(attn, spec.k_path).output[...] -= delta

                saved.append(attn.output[1].save())

                if plan.mlp_edit is not None:
                    mlp = _resolve(block, spec.mlp_out_path)
                    if plan.mlp_edit == "zero_all":
                        mlp.output[:] = 0
                    elif i == 0 and swap_dirs is not None:
                        out = mlp.output
                        v0, v1 = swap_dirs
                        if spec.mlp_out_layout == "NH":
                            r0, r1 = out[0], out[1]
                        else:
                            r0, r1 = out[0, 0], out[0, 1]
                        alpha = (r0 * v0).sum()
                        shift = alpha * v0 - alpha * v1
                        delta = torch.zeros_like(out)
                        if spec.mlp_out_layout == "NH":
                            delta[0] = -shift
                            delta[1] = shift
                        else:
                            delta[0, 0] = -shift
                            delta[0, 1] = shift
                        mlp.output = out + delta

        out = []
        for s in saved:
            if s is None:
                raise RuntimeError(
                    "Attention probabilities came back None. The model is not running "
                    "eager attention, or this architecture needs output_attentions=True."
                )
            out.append(s[0].detach().float().cpu())
        return out

    # --- public API ----------------------------------------------------------------

    def run_all(self, inputs, *, massive_coords: Optional[Sequence[int]] = None,
                swap_dirs: Optional[Dict[str, tuple]] = None,
                verbose: bool = True) -> Dict[str, List[torch.Tensor]]:
        """Run every intervention; mirrors each harness's ``run_all_interventions``.

        ``swap_dirs`` maps ``"int_d"``/``"int_e"`` to the ``(v0_hat, v1_hat)`` unit vectors
        for the component swap; the caller supplies them because each harness derives its
        EPE/PE differently. ``massive_coords`` is threaded to (i) and sizes (j)'s
        size-matched random control.
        """
        swap_dirs = swap_dirs or {}
        k = len(massive_coords) if massive_coords is not None else 0
        random_columns = (draw_random_wk_columns(self.num_layers, k, self.hidden)
                          if k else None)

        results = {}
        for key in INTERVENTION_ORDER:
            plan = self.plans[key]
            if verbose:
                print(f"  Running {key} (nnsight)...")
            # Deliberately not wrapped in torch.no_grad(): NNsight defers the trace body to
            # __exit__ and manages grad itself, so an outer no_grad creates the slice views
            # in one grad mode and mutates them in another ("A view was created in no_grad
            # mode and is being modified inplace with grad mode enabled"). Saved
            # activations are detached on the way out instead.
            results[key] = self.run_intervention(
                plan, inputs,
                massive_coords=massive_coords,
                random_columns=random_columns,
                swap_dirs=swap_dirs.get(key),
            )
        return results

    def engine_info(self) -> dict:
        """Provenance fields for ``run_config.json``."""
        import nnsight
        import transformers
        return {
            "name": "nnsight",
            "nnsight_version": nnsight.__version__,
            "remote": self.remote,
            "attn_implementation": "eager",
            "attn_probs_source": f"{self.spec.blocks_path}[i].{self.spec.attn_path}.output[1]",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Parity harness
# ═══════════════════════════════════════════════════════════════════════════════

# fp32 tolerances. The two engines are algebraically identical but not bit-identical
# (Conv1D addmm vs F.linear; `/sqrt(d)` vs `*d**-0.5`; OPT's relocated query scaling), so
# ~1e-6 on raw probabilities is expected. The gate is on the BOS metric, because that is
# the number that reaches the CSV — which is written to 6 decimal places.
METRIC_ATOL = 1e-5
METRIC_RTOL = 1e-4
ATTN_ATOL = 1e-6
ATTN_RTOL = 1e-5


def verify_parity(engine: "NNsightEngine", manual_runner: Callable, inputs_list: Sequence,
                  band, num_layers: int, *, massive_coords=None, swap_dirs=None,
                  metric_atol: float = METRIC_ATOL, metric_rtol: float = METRIC_RTOL,
                  strict: bool = True) -> dict:
    """Compare the NNsight engine against the manual harness, per intervention.

    Two tiers, following the conventions of ``evaluation_robustness_analysis.verify_parity``:

    * *informational* — max abs deviation on the raw ``[heads, seq, seq]`` probabilities;
    * *gate* — abs/rel deviation on the per-sentence BOS metric, i.e. the number that
      actually lands in the Table-1 CSV.

    ``strict=False`` downgrades the gate to advisory. Callers should do this for fp16/bf16,
    where the two paths are genuinely *different algorithms* rather than rounding variants:
    GPT-Neo upcasts q/k to fp32 inside attention and OPT/Qwen softmax in fp32, while the
    manual path does neither. A half-precision mismatch is expected and is not evidence
    that either engine is wrong.

    A divergence in fp32 *is* meaningful: the real HF forward is ground truth, so a
    mismatch means the hand-rolled re-implementation deviates from the model the paper
    claims to describe.
    """
    from intervention_analysis import compute_bos_attention_metric

    ls, le = band
    rows = []
    for key in INTERVENTION_ORDER:
        plan = engine.plans[key]
        max_attn_d, max_abs_d, max_rel_d = 0.0, 0.0, 0.0

        k = len(massive_coords) if massive_coords is not None else 0
        random_columns = draw_random_wk_columns(engine.num_layers, k, engine.hidden) if k else None

        for inputs in inputs_list:
            manual = manual_runner(inputs)[key]
            nn_out = engine.run_intervention(
                plan, inputs, massive_coords=massive_coords,
                random_columns=random_columns,
                swap_dirs=(swap_dirs or {}).get(key),
            )
            max_attn_d = max(max_attn_d,
                             max(float((a - b).abs().max()) for a, b in zip(manual, nn_out)))
            bm = compute_bos_attention_metric(manual, num_layers, "mid", layer_start=ls, layer_end=le)
            bn = compute_bos_attention_metric(nn_out, num_layers, "mid", layer_start=ls, layer_end=le)
            max_abs_d = max(max_abs_d, abs(bm - bn))
            max_rel_d = max(max_rel_d, abs(bm - bn) / max(abs(bm), 1e-12))

        passed = max_abs_d <= metric_atol + metric_rtol * abs(max_abs_d) or max_abs_d <= metric_atol
        rows.append({
            "intervention": key,
            "status": "pass" if passed else ("fail" if strict else "advisory"),
            "max_abs_attention_difference": max_attn_d,
            "max_abs_metric_deviation": max_abs_d,
            "max_rel_metric_deviation": max_rel_d,
            "metric_atol": metric_atol,
            "metric_rtol": metric_rtol,
            "note": "manual vs nnsight, BOS-metric gate",
        })

    report = {
        "engine": engine.engine_info(),
        "architecture": engine.spec.name,
        "strict": strict,
        "n_sentences": len(inputs_list),
        "band": [int(ls), int(le)],
        "rows": rows,
        "all_rows_pass": all(r["status"] == "pass" for r in rows),
    }
    if strict and not report["all_rows_pass"]:
        failed = [r["intervention"] for r in rows if r["status"] != "pass"]
        raise AssertionError(
            f"NNsight parity failed for {engine.spec.name}: {failed}. "
            "The real HF forward is ground truth here — investigate the manual path."
        )
    return report
