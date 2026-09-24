"""进阶判责链路的结构化契约。"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from models.schemas import Contract, Probability


class ClassificationResult(Contract):
    """报案分类输出：意图、复杂度与理由（供模型路由使用）。"""
    intent: Literal['claim', 'consult', 'status'] = 'claim'
    complex: bool = False
    reason: str = Field(default='', max_length=500)


class ExpertJudgment(Contract):
    """专家意见输出契约；模型必须输出该结构才能进入融合。"""
    expert: Literal['damage', 'risk', 'liability']
    recommendation: Literal['accept', 'reject', 'review', 'investigate']
    confidence: Probability
    risk_score: Probability | None = None
    rationale: str = Field(min_length=1, max_length=2000)
    clause_ids: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)


class TriageExplanation(Contract):
    """分流说明（供人工队列与审计）。"""
    band: Literal['high', 'mid', 'low']
    rationale: str = Field(min_length=1, max_length=1000)


__all__ = ['ClassificationResult', 'ExpertJudgment', 'TriageExplanation']
