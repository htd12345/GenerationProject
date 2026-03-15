"""
核心模块初始化
Core module initialization
"""

from .unified_semantic import (
    SemanticRepresentation,
    BaseEncoder,
    TextEncoder,
    TableEncoder,
    UnifiedSemanticModel,
    SemanticIndex
)
from .cross_modal_alignment import (
    AlignmentResult,
    ProjectionHead,
    ContrastiveLoss,
    CrossModalAttention,
    CrossModalAligner,
    AlignmentTrainer
)
from .query_decomposition import (
    QueryType,
    DecompositionStrategy,
    SubQuery,
    QueryDecomposition,
    QueryClassifier,
    ComplexityEstimator,
    QueryDecomposer
)
from .multi_hop_reasoning import (
    ReasoningState,
    ReasoningType,
    ReasoningStep,
    ReasoningChain,
    KnowledgeStore,
    ReasoningPathPlanner,
    InferenceEngine,
    MultiHopReasoner
)
from .explainability import (
    ExplanationLevel,
    ExplanationFormat,
    Evidence,
    ReasoningNode,
    ExplanationResult,
    ExplanationFormatter,
    ExplainabilityEngine,
    ConfidenceCalibrator
)

__all__ = [
    # Unified Semantic
    "SemanticRepresentation",
    "BaseEncoder",
    "TextEncoder",
    "TableEncoder",
    "UnifiedSemanticModel",
    "SemanticIndex",

    # Cross-Modal Alignment
    "AlignmentResult",
    "ProjectionHead",
    "ContrastiveLoss",
    "CrossModalAttention",
    "CrossModalAligner",
    "AlignmentTrainer",

    # Query Decomposition
    "QueryType",
    "DecompositionStrategy",
    "SubQuery",
    "QueryDecomposition",
    "QueryClassifier",
    "ComplexityEstimator",
    "QueryDecomposer",

    # Multi-Hop Reasoning
    "ReasoningState",
    "ReasoningType",
    "ReasoningStep",
    "ReasoningChain",
    "KnowledgeStore",
    "ReasoningPathPlanner",
    "InferenceEngine",
    "MultiHopReasoner",

    # Explainability
    "ExplanationLevel",
    "ExplanationFormat",
    "Evidence",
    "ReasoningNode",
    "ExplanationResult",
    "ExplanationFormatter",
    "ExplainabilityEngine",
    "ConfidenceCalibrator"
]
