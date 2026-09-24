"""方向一+方向三：模型路由与多级降级兜底。

- route(): 按任务画像选择模型档位，案件复杂度信号（大额/欺诈线索）可升档。
- ainvoke(): 按链执行 pro→main→fast，全部失败时落到调用方注入的规则兜底，
  全程记录路由决策、降级事件，供链路追踪与审计输出。
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

# 复杂度升级关键词：命中即把判责类任务升到pro档。
COMPLEXITY_KEYWORDS = ('欺诈', '骗保', '诉讼', '伤残', '死亡', '重大', '拒赔争议', '多人受伤')
HIGH_AMOUNT_THRESHOLD = 50_000.0


@dataclass(frozen=True)
class TaskProfile:
    """任务画像：首选档位、降级链与路由理由。"""
    name: str
    primary_tier: str
    chain: tuple[str, ...]
    reason: str


TASK_PROFILES: dict[str, TaskProfile] = {
    'classification': TaskProfile('classification', 'fast', ('fast',), '意图与复杂度分类是低难度任务，使用轻量档'),
    'rag_answer': TaskProfile('rag_answer', 'main', ('main', 'fast'), '条款问答需要中等推理能力，失败降级轻量档'),
    'expert_damage': TaskProfile('expert_damage', 'main', ('main', 'fast'), '定损分析中等复杂度'),
    'expert_risk': TaskProfile('expert_risk', 'pro', ('pro', 'main', 'fast'), '反欺诈分析高复杂度，首选强推理档'),
    'expert_liability': TaskProfile('expert_liability', 'pro', ('pro', 'main', 'fast'), '责任判定是核心判责任务，首选强推理档'),
    'decision': TaskProfile('decision', 'pro', ('pro', 'main'), '最终判责建议需强推理，失败降级主档'),
    'report': TaskProfile('report', 'main', ('main', 'fast'), '报告生成重表达，中等档即可'),
}


@dataclass
class RouteDecision:
    """一次路由决策的完整记录。"""
    task: str
    tier: str
    chain: tuple[str, ...]
    reason: str
    escalated: bool = False
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {'task': self.task, 'tier': self.tier, 'chain': list(self.chain),
                'reason': self.reason, 'escalated': self.escalated, 'latency_ms': round(self.latency_ms, 1)}


@dataclass
class FallbackEvent:
    """一次降级事件：从哪个档位、因什么错误、降到哪个档位。"""
    from_tier: str
    to_tier: str
    error: str

    def as_dict(self) -> dict[str, Any]:
        return {'from': self.from_tier, 'to': self.to_tier, 'error': self.error}


@dataclass
class ModelCallResult:
    """模型调用结果：内容 + 路由与降级证据。"""
    content: str
    route: RouteDecision
    attempts: int = 1
    degraded: bool = False
    fallback_events: list[FallbackEvent] = field(default_factory=list)
    rule_based: bool = False


def complexity_signals(claim: dict[str, Any] | None) -> dict[str, Any]:
    """从报案数据提取复杂度信号；只报告事实，不虚构信号。"""
    claim = claim or {}
    amount = float(claim.get('amount') or 0)
    text = str(claim.get('description') or '')
    keywords = [word for word in COMPLEXITY_KEYWORDS if word in text]
    return {'amount': amount, 'high_amount': amount > HIGH_AMOUNT_THRESHOLD,
            'complex_keywords': keywords, 'complex': amount > HIGH_AMOUNT_THRESHOLD or bool(keywords)}


class ModelRouter:
    """模型路由器：model_factory(tier) 由调用方注入，测试可传假模型，live传工厂。

    路由两级决策：
    1. 任务画像决定首选档位与降级链；
    2. 复杂度信号（大额/欺诈线索）把非fast任务升到pro档（升级保持链顺序约束见_escalated_chain）。
    """

    def __init__(self, model_factory: Callable[[str], Any]) -> None:
        self.model_factory = model_factory
        self.routing_history: list[RouteDecision] = []

    def route(self, task: str, *, claim: dict[str, Any] | None = None) -> RouteDecision:
        if task not in TASK_PROFILES:
            raise ValueError(f'未知路由任务：{task}，可用：{sorted(TASK_PROFILES)}')
        profile = TASK_PROFILES[task]
        signals = complexity_signals(claim)
        tier, escalated, reason = profile.primary_tier, False, profile.reason
        if signals['complex'] and profile.primary_tier != 'fast':
            tier = 'pro'
            escalated = True
            reason = f'复杂度信号触发升档（大额={signals["high_amount"]}，关键词={signals["complex_keywords"]}），路由至强推理档'
        chain = self._escalated_chain(profile.chain, tier)
        decision = RouteDecision(task, tier, chain, reason, escalated)
        self.routing_history.append(decision)
        return decision

    @staticmethod
    def _escalated_chain(chain: tuple[str, ...], primary: str) -> tuple[str, ...]:
        """升档后以primary开头、保留链内其余档位且去重，保持降级顺序。"""
        rest = tuple(tier for tier in chain if tier != primary)
        return (primary, *rest)

    async def ainvoke(self, task: str, messages: Sequence[Any], *,
                      claim: dict[str, Any] | None = None,
                      rule_fallback: Callable[[], str] | None = None) -> ModelCallResult:
        """按路由链调用模型；异常逐档降级，最终可落规则兜底。

        注意：这里的降级只捕获模型调用异常；解析失败由 structured_retry 处理，
        两类失败不混在一起，避免把网络错误伪装成格式问题。
        """
        decision = self.route(task, claim=claim)
        start = time.monotonic()
        events: list[FallbackEvent] = []
        for index, tier in enumerate(decision.chain):
            try:
                model = self.model_factory(tier)
                response = await model.ainvoke(list(messages))
                decision.latency_ms = (time.monotonic() - start) * 1000
                return ModelCallResult(content=str(response.content), route=decision,
                                       attempts=index + 1, fallback_events=events)
            except Exception as exc:  # noqa: BLE001 - 降级链需捕获所有模型调用失败
                next_tier = decision.chain[index + 1] if index + 1 < len(decision.chain) else 'rules' if rule_fallback else 'exhausted'
                events.append(FallbackEvent(tier, next_tier, f'{type(exc).__name__}: {exc}'[:200]))
        decision.latency_ms = (time.monotonic() - start) * 1000
        if rule_fallback is not None:
            return ModelCallResult(content=rule_fallback(), route=decision,
                                   attempts=len(decision.chain), degraded=True,
                                   fallback_events=events, rule_based=True)
        raise RuntimeError(f'模型链路全部失败（{decision.chain}），且未配置规则兜底：{[e.as_dict() for e in events]}')


__all__ = ['TASK_PROFILES', 'TaskProfile', 'RouteDecision', 'FallbackEvent',
           'ModelCallResult', 'ModelRouter', 'complexity_signals']
