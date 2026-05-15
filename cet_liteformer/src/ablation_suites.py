from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

VariantSpec = Tuple[str, str, Dict[str, Any]]


def _deepcopy_cfg(base: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(base)


def variants_model_components(base_cfg: Dict[str, Any]) -> List[VariantSpec]:
    """
    Incremental ladder (weak → strong) on the same data pipeline and broad training setup.
    Each step adds a modeling component relative to simpler predecessors.
    """
    specs: List[VariantSpec] = []

    # 1 — vector baseline
    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "MLPBaseline"
    specs.append(
        (
            "01_mlp_baseline",
            "MLP (2 hidden layers): no feature-sequence or attention.",
            c,
        )
    )

    # 2 — standard tabular transformer
    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "StandardTransformerBaseline"
    specs.append(
        (
            "02_standard_transformer",
            "Standard Transformer: same tokenizer + multi-head self-attention + plain FFN; no correntropy, no entropy gate.",
            c,
        )
    )

    # 3 — CET stack, standard attention, no entropy gate
    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "CET-LiteFormer"
    c["model"]["use_entropy_gate"] = False
    c["model"]["use_correntropy_attention"] = False
    c["model"]["attention_type"] = "standard"
    c["model"]["learnable_sigma"] = False
    specs.append(
        (
            "03_cet_standard_attention_no_gate",
            "CET-LiteFormer blocks with scaled-dot-product attention; entropy gate off.",
            c,
        )
    )

    # 4 — correntropy attention, still no gate
    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "CET-LiteFormer"
    c["model"]["use_entropy_gate"] = False
    c["model"]["use_correntropy_attention"] = True
    c["model"]["attention_type"] = "correntropy_rbf"
    c["model"]["learnable_sigma"] = False
    specs.append(
        (
            "04_cet_correntropy_no_gate",
            "Add Gaussian RBF / correntropy self-attention (no entropy gate yet).",
            c,
        )
    )

    # 5 — entropy gate + MI prior
    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "CET-LiteFormer"
    c["model"]["use_entropy_gate"] = True
    c["model"]["use_correntropy_attention"] = True
    c["model"]["attention_type"] = "correntropy_rbf"
    c["model"]["learnable_sigma"] = False
    specs.append(
        (
            "05_cet_correntropy_entropy_gate",
            "Add entropy feature gate with MI prior on top of correntropy attention.",
            c,
        )
    )

    # 6 — learnable correntropy scale (matches typical full model)
    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "CET-LiteFormer"
    c["model"]["use_entropy_gate"] = True
    c["model"]["use_correntropy_attention"] = True
    c["model"]["attention_type"] = "correntropy_rbf"
    c["model"]["learnable_sigma"] = True
    specs.append(
        (
            "06_cet_full_learnable_sigma",
            "Full CET-LiteFormer stack: correntropy + entropy gate + learnable sigma (same capacity as default config).",
            c,
        )
    )

    return specs


def variants_training_objectives(base_cfg: Dict[str, Any]) -> List[VariantSpec]:
    """
    Same full CET model; ablate training objectives only (loss shaping).
    """
    specs: List[VariantSpec] = []

    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "CET-LiteFormer"
    c["training"]["use_focal_loss"] = False
    c["training"]["exit_loss_lambda"] = 0.0
    c["training"]["gate_l1_lambda"] = 0.0
    specs.append(("T1_ce_only", "Weighted cross-entropy only (no focal, no exit aux, no gate L1).", c))

    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "CET-LiteFormer"
    c["training"]["use_focal_loss"] = True
    c["training"]["exit_loss_lambda"] = 0.0
    c["training"]["gate_l1_lambda"] = 0.0
    specs.append(("T2_plus_focal_loss", "Add focal loss (gamma from config).", c))

    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "CET-LiteFormer"
    c["training"]["use_focal_loss"] = True
    c["training"]["exit_loss_lambda"] = float(base_cfg["training"].get("exit_loss_lambda", 0.22))
    c["training"]["gate_l1_lambda"] = 0.0
    specs.append(("T3_plus_exit_supervision", "Add intermediate exit-head supervision (deep supervision).", c))

    c = _deepcopy_cfg(base_cfg)
    c["model"]["name"] = "CET-LiteFormer"
    specs.append(("T4_full_training_objectives", "Full objective: focal + exit supervision + gate L1 (config defaults).", c))

    return specs


def variants_legacy(base_cfg: Dict[str, Any]) -> List[VariantSpec]:
    """Previous one-off ablations + baselines (not strictly incremental)."""
    specs: List[VariantSpec] = []

    def cpy() -> Dict[str, Any]:
        return _deepcopy_cfg(base_cfg)

    c = cpy()
    specs.append(("full_cet_liteformer", "Full model (config as-is).", c))

    v = cpy()
    v["model"]["use_entropy_gate"] = False
    specs.append(("no_entropy_gate", "Disable entropy gate.", v))

    v = cpy()
    v["model"]["use_correntropy_attention"] = False
    v["model"]["attention_type"] = "standard"
    specs.append(("standard_attention_instead_of_correntropy", "Standard dot-product attention.", v))

    v = cpy()
    v["model"]["use_early_exit"] = False
    specs.append(("no_early_exit", "Disable inference early exit (train still uses exit heads if loss says so).", v))

    v = cpy()
    v["training"]["use_focal_loss"] = False
    specs.append(("no_focal_loss", "Cross-entropy instead of focal.", v))

    v = cpy()
    v["training"]["gate_l1_lambda"] = 0.0
    specs.append(("no_gate_sparsity", "Remove gate L1 penalty.", v))

    v = cpy()
    v["model"]["name"] = "MLPBaseline"
    specs.append(("shallow_mlp_baseline", "MLP baseline.", v))

    v = cpy()
    v["model"]["name"] = "StandardTransformerBaseline"
    specs.append(("standard_transformer_baseline", "Standard Transformer baseline.", v))

    return specs


def get_variant_specs(base_cfg: Dict[str, Any], suite: str) -> List[VariantSpec]:
    suite = (suite or "model_components").strip().lower()
    if suite in ("model", "model_components", "components"):
        return variants_model_components(base_cfg)
    if suite in ("training", "training_objectives", "losses"):
        return variants_training_objectives(base_cfg)
    if suite in ("legacy", "old"):
        return variants_legacy(base_cfg)
    if suite == "all":
        return variants_model_components(base_cfg) + variants_training_objectives(base_cfg)
    raise ValueError(
        f"Unknown ablation suite: {suite}. "
        f"Use model_components | training_objectives | legacy | all"
    )
