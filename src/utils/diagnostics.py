"""Per-layer-group gradient-norm logging.

Models that want grouped reporting declare their groups by exposing a
`layer_groups` attribute — a dict mapping group_name → list of parameter-name
prefixes. Models without it get a single "all" group.

Diagnostic rule of thumb: if encoder gradient norm < (1/10) of decoder
gradient norm at convergence, the encoder is not learning regardless of what
the loss curve looks like. This callback emits the numbers needed to evaluate
that ratio post-hoc.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import lightning.pytorch as pl
import torch


def _matches_any_prefix(name: str, prefixes: List[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def _resolve_groups(pl_module: pl.LightningModule) -> Dict[str, List[str]]:
    groups = getattr(pl_module, "layer_groups", None)
    if groups is None:
        return {"all": [""]}  # empty-string prefix matches every parameter
    if not isinstance(groups, Mapping):
        raise TypeError(
            f"`layer_groups` must be a Mapping[str, List[str]]; got {type(groups).__name__}"
        )
    return {k: list(v) for k, v in groups.items()}


class GradientNormLogger(pl.Callback):
    """Logs per-layer-group L2 gradient norms each train step.

    Without this, an encoder grad-starvation failure (Tier 2/3 risk per proposal
    v2.1 §"Sparse encoder gradients") manifests as "loss curve looks fine but
    accuracy plateaus" — silently misleading. The fix (per the proposal) is the
    1:10 ratio diagnostic; this callback emits the numbers it needs.

    Logs `gradnorm/<group>` and `gradnorm/__total__` to whatever logger
    Lightning is configured with (TensorBoard by default).
    """

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self._step_counter = 0
        self._cached_param_groups: Optional[Dict[str, List[torch.nn.Parameter]]] = None
        self._cached_param_names: Optional[Dict[str, List[str]]] = None

    def _build_param_groups(self, pl_module: pl.LightningModule) -> None:
        spec = _resolve_groups(pl_module)
        named = list(pl_module.named_parameters())
        unmatched = []
        groups: Dict[str, List[torch.nn.Parameter]] = {g: [] for g in spec}
        names: Dict[str, List[str]] = {g: [] for g in spec}
        for name, param in named:
            if not param.requires_grad:
                continue
            placed = False
            for group_name, prefixes in spec.items():
                if _matches_any_prefix(name, prefixes):
                    groups[group_name].append(param)
                    names[group_name].append(name)
                    placed = True
                    if len(spec) > 1:
                        break  # avoid double-counting when groups partition the params
            if not placed and len(spec) > 1:
                unmatched.append(name)
        self._cached_param_groups = groups
        self._cached_param_names = names
        if unmatched:
            # Surface as a one-shot warning via the logger's text channel; not fatal.
            pl_module.print(
                f"[GradientNormLogger] {len(unmatched)} parameter(s) match no group "
                f"(e.g. {unmatched[:3]}). They will be excluded from per-group norms."
            )

    @staticmethod
    def _l2_norm(params: List[torch.nn.Parameter]) -> Optional[float]:
        sq = 0.0
        seen = False
        for p in params:
            if p.grad is None:
                continue
            seen = True
            # detach() — we only want the value, not a node in the graph.
            sq = sq + p.grad.detach().pow(2).sum().item()
        if not seen:
            return None
        return float(sq ** 0.5)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self._step_counter += 1
        if self._step_counter % self.log_every_n_steps != 0:
            return
        if self._cached_param_groups is None:
            self._build_param_groups(pl_module)
        assert self._cached_param_groups is not None

        total_sq = 0.0
        for group_name, params in self._cached_param_groups.items():
            norm = self._l2_norm(params)
            if norm is None:
                continue
            pl_module.log(
                f"gradnorm/{group_name}", norm,
                on_step=True, on_epoch=False, prog_bar=False,
            )
            total_sq += norm * norm
        if total_sq > 0:
            pl_module.log(
                "gradnorm/__total__", float(total_sq ** 0.5),
                on_step=True, on_epoch=False, prog_bar=False,
            )

    def get_param_group_norms(self, pl_module: pl.LightningModule) -> Dict[str, Optional[float]]:
        """Test/diagnostic helper — compute group norms once on demand."""
        if self._cached_param_groups is None:
            self._build_param_groups(pl_module)
        assert self._cached_param_groups is not None
        return {g: self._l2_norm(params) for g, params in self._cached_param_groups.items()}
