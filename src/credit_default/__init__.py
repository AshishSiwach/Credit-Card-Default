"""Credit card default prediction package.

Leakage-safe, pipeline-driven model for predicting next-month default
on the UCI credit-card-clients data. See the README for the metric
rationale and the precision-floor decision rule.
"""

__all__ = ["data", "features", "pipeline", "train", "evaluate", "compare"]
