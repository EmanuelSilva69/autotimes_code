"""
Testes_eficiencia.modules - Modulos de melhoria arquitetural para o
benchmark (Ablation Study) do framework AutoTimes.

Exporta todas as classes de melhoria:
    - RobustRevIN                 (normalizacao robusta mediana/MAD)
    - VolatilityAwareLogCoshLoss  (loss Log-Cosh atenuada por volatilidade)
    - OverlappingPatchEmbedding   (patches sobrepostos via torch.unfold)
    - build_discriminative_optimizer / build_cosine_warm_scheduler
    - VolatilityScaledFusionGate  (porta de fusao cross-attention + variancia)
"""

from .robust_revin import RobustRevIN
from .robust_revin import RobustRevIN
from .volatility_loss import VolatilityAwareLogCoshLoss, HybridVolatilityLoss
from .overlapping_embed import OverlappingPatchEmbedding
from .discriminative_optim import build_discriminative_optimizer, build_cosine_warm_scheduler
from .volatility_gate import VolatilityScaledFusionGate

__all__ = [
    "RobustRevIN",
    "VolatilityAwareLogCoshLoss",
    "HybridVolatilityLoss",
    "OverlappingPatchEmbedding",
    "build_discriminative_optimizer",
    "build_cosine_warm_scheduler",
    "VolatilityScaledFusionGate",
]
