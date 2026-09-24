"""进阶判责工作流的服务层：离线确定性服务 + live组合服务。

- OfflineJudgeServices：固定场景合成输出，不调用真实模型（教学/评测/CI用）；
  保单查询与条款检索走真实模拟API与真实检索逻辑。
- LiveJudgeServices：组合 ModelRouter（路由+降级）+ structured_call（校验重试）
  + PromptVersionManager（版本化提示），是三个方向能力的真实执行链。
"""
from __future__ import annotations

import json
from typing import Any

from advanced.clause_store import ClauseStore, build_clause_query
from advanced.model_router import ModelRouter
from advanced.prompt_versioning import PromptVersionManager
from advanced.schemas import ClassificationResult, ExpertJudgment
from advanced.structured_retry import StructuredOutputError, structured_call

EXPERT_ROLES = ('damage', 'risk', 'liability')
EXPERT_TASK = {'damage': 'expert_damage', 'risk': 'expert_risk', 'liability': 'expert_liability'}
ROLE_CN = {'damage': '定损', 'risk': '反欺诈', 'liability': '责任'}


def _exclusion_hit(claim: dict[str, Any], policy: dict[str, Any] | None) -> str | None:
    """报案描述与保单免责标记的交集；只报事实命中，不做扩大解释。"""
    if not policy:
        return None
    text = str(claim.get('description') or '')
    signals = {'dui_exclusion': ('醉酒', '饮酒', '酒驾'),
               'license_suspended': ('驾驶证被暂扣', '无证驾驶', '驾驶证吊销')}
    for tag in policy.get('exclusion_tags') or []:
        for keyword in signals.get(tag, ()):
            if keyword in text:
                return tag
    return None


class JudgeServices:
    """判责图依赖的服务集合；子类实现具体行为。"""

    demo = False

    def __init__(self, clause_store: ClauseStore, policy_client: Any = None,
                 router: ModelRouter | None = None,
                 prompts: PromptVersionManager | None = None) -> None:
        self.clause_store = clause_store
        self.policy_client = policy_client
        self.router = router
        self.prompts = prompts

    # -- 共用真实能力 --------------------------------------------------
    async def policy_lookup(self, claim: dict[str, Any]) -> dict[str, Any]:
        if self.policy_client is not None:
            return self.policy_client.get_policy(str(claim.get('policy_id') or ''))
        raise RuntimeError('policy_client_not_configured')

    def retrieve_clauses(self, claim: dict[str, Any], policy: dict[str, Any] | None) -> list[Any]:
        query = build_clause_query(claim, policy)
        return self.clause_store.retrieve(query, k=4)

    async def classify(self, claim: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def run_expert(self, role: str, claim: dict[str, Any], policy: dict[str, Any] | None,
                         clause_context: str, classification: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class OfflineJudgeServices(JudgeServices):
    """离线确定性服务：专家意见按场景固定，显式标记demo，不冒充模型结果。"""

    demo = True

    # 场景表：expert → (recommendation, confidence, risk_score, missing)
    SCENARIOS: dict[str, dict[str, tuple[str, float, float | None, list[str]]]] = {
        'auto': {
            'damage': ('accept', .95, None, []),
            'risk': ('accept', .92, .08, []),
            'liability': ('accept', .90, None, []),
        },
        'auto_minor': {
            'damage': ('accept', .96, None, []),
            'risk': ('accept', .95, .05, []),
            'liability': ('accept', .94, None, []),
        },
        'review_missing': {
            'damage': ('accept', .80, None, ['维修发票']),
            'risk': ('accept', .75, .10, []),
            'liability': ('accept', .85, None, []),
        },
        'reject_expired': {
            'damage': ('reject', .35, None, []),
            'risk': ('reject', .40, .30, []),
            'liability': ('reject', .30, None, []),
        },
        'reject_exclusion': {
            'damage': ('reject', .40, None, []),
            'risk': ('reject', .45, .35, []),
            'liability': ('reject', .38, None, []),
        },
        'risk_review': {
            'damage': ('accept', .70, None, []),
            'risk': ('investigate', .60, .55, []),
            'liability': ('accept', .75, None, []),
        },
        'over_limit': {
            'damage': ('accept', .85, None, []),
            'risk': ('accept', .90, .10, []),
            'liability': ('accept', .88, None, []),
        },
    }

    def __init__(self, scenario: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if scenario not in self.SCENARIOS:
            raise ValueError(f'未知离线场景：{scenario}，可用：{sorted(self.SCENARIOS)}')
        self.scenario = scenario

    async def policy_lookup(self, claim: dict[str, Any]) -> dict[str, Any]:
        # 未注入HTTP客户端时直接读进程内保单库，离线测试/CI零依赖。
        if self.policy_client is None:
            from advanced.mock_policy_api import POLICY_DB
            policy_id = str(claim.get('policy_id') or '')
            record = POLICY_DB.get(policy_id)
            if record is None:
                raise RuntimeError(f'模拟保单API·进程内：保单不存在：{policy_id}')
            return dict(record, source='模拟保单API·进程内')
        return await super().policy_lookup(claim)

    async def classify(self, claim: dict[str, Any]) -> dict[str, Any]:
        if self.router is None:
            raise RuntimeError('offline服务也需要router以记录路由决策')
        decision = self.router.route('classification', claim=claim)
        return {'intent': 'claim',
                'complex': bool(claim.get('amount') and float(claim['amount']) > 50000),
                'reason': '离线规则分类：金额与关键词固定映射', 'demo': True,
                'route': decision.as_dict()}

    async def run_expert(self, role: str, claim: dict[str, Any], policy: dict[str, Any] | None,
                         clause_context: str, classification: dict[str, Any]) -> dict[str, Any]:
        decision = self.router.route(EXPERT_TASK[role], claim=claim)
        recommendation, confidence, risk_score, missing = self.SCENARIOS[self.scenario][role]
        clause_ids = [doc.metadata['doc_id'] for doc in
                      self.clause_store.retrieve(build_clause_query(claim, policy), k=2, kind='exclusion')] \
            if recommendation == 'reject' else []
        spec = self.prompts.get('expert') if self.prompts else None
        return {'expert': role, 'recommendation': recommendation, 'confidence': confidence,
                'risk_score': risk_score,
                'rationale': f'离线合成{ROLE_CN[role]}意见（场景={self.scenario}），未调用模型',
                'clause_ids': clause_ids, 'missing_information': missing,
                'model_tier': decision.tier, 'model_chain': list(decision.chain),
                'prompt_version': f'{spec.version}@{spec.content_hash()}' if spec else '未配置',
                'attempts': 1, 'degraded': False, 'rule_based': False, 'demo': True,
                'route': decision.as_dict()}


class _RouterModelAdapter:
    """把ModelRouter包装成structured_call可用的模型接口。

    每次ainvoke都完整执行路由+降级链；解析失败重试时重新路由，
    路由与降级证据累积在evidence里供图状态记录。
    """

    def __init__(self, router: ModelRouter, task: str, claim: dict[str, Any],
                 rule_fallback: Any = None) -> None:
        self.router, self.task, self.claim, self.rule_fallback = router, task, claim, rule_fallback
        self.evidence: list[dict[str, Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        from langchain_core.messages import AIMessage
        result = await self.router.ainvoke(self.task, messages, claim=self.claim,
                                           rule_fallback=self.rule_fallback)
        self.evidence.append(result.route.as_dict() | {
            'attempts': result.attempts, 'degraded': result.degraded,
            'rule_based': result.rule_based,
            'fallback_events': [event.as_dict() for event in result.fallback_events]})
        return AIMessage(content=result.content)


class LiveJudgeServices(JudgeServices):
    """live组合服务：路由+降级+结构化重试+版本化Prompt一次跑通。"""

    demo = False

    def __init__(self, *, model_factory: Any, json_mode: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.router is None or self.prompts is None:
            raise ValueError('live服务需要router与prompts')
        self.model_factory = model_factory
        self.json_mode = json_mode
        # 路由器拿到的是已绑定JSON输出格式的模型；降级重路由时同样带绑定。
        self.router.model_factory = self._bind_model

    def _bind_model(self, tier: str) -> Any:
        model = self.model_factory(tier)
        return model.bind(response_format={'type': 'json_object'}, stop=[]) if self.json_mode else model

    def _rule_classification(self) -> str:
        return json.dumps({'intent': 'claim', 'complex': False,
                           'reason': '模型链路降级：规则兜底按普通报案处理'}, ensure_ascii=False)

    def _rule_expert(self, role: str) -> str:
        """规则兜底专家意见：保守建议人工复核，不编造分析。"""
        return json.dumps({'expert': role, 'recommendation': 'review', 'confidence': .5,
                           'rationale': '模型链路降级，规则兜底：无法自动判责，建议人工复核',
                           'clause_ids': [], 'missing_information': ['模型链路不可用']}, ensure_ascii=False)

    async def classify(self, claim: dict[str, Any]) -> dict[str, Any]:
        prompt_text, prompt_meta = self.prompts.render(
            'classification',
            {'claim_text': str(claim.get('description') or ''), 'amount': str(claim.get('amount') or '未知')},
            context=f"classify:{claim.get('claim_id')}")
        from langchain_core.messages import HumanMessage
        adapter = _RouterModelAdapter(self.router, 'classification', claim, self._rule_classification)
        try:
            result = await structured_call(adapter, ClassificationResult,
                                           [HumanMessage(content=prompt_text)])
        except StructuredOutputError as exc:
            # 解析重试也失败：使用规则兜底结果，标记降级。
            fallback = ClassificationResult.model_validate_json(self._rule_classification())
            return fallback.model_dump() | {'demo': False, 'degraded': True,
                                            'parse_attempts': exc.attempts,
                                            'route': adapter.evidence[0] if adapter.evidence else None,
                                            'prompt': prompt_meta}
        return result.value.model_dump() | {
            'route': adapter.evidence[0] if adapter.evidence else None,
            'retry': {'attempts': result.attempts, 'errors': result.errors[:2]} if result.attempts > 1 else None,
            'prompt': prompt_meta, 'all_routes': adapter.evidence}

    async def run_expert(self, role: str, claim: dict[str, Any], policy: dict[str, Any] | None,
                         clause_context: str, classification: dict[str, Any]) -> dict[str, Any]:
        prompt_text, prompt_meta = self.prompts.render(
            'expert', {'role': ROLE_CN[role], 'claim_text': str(claim.get('description') or ''),
                       'policy_info': json.dumps(policy or {}, ensure_ascii=False, default=str)[:1500],
                       'clauses': clause_context},
            context=f"expert:{role}:{claim.get('claim_id')}")
        from langchain_core.messages import HumanMessage, SystemMessage
        schema_text = json.dumps(ExpertJudgment.model_json_schema(), ensure_ascii=False)
        messages = [SystemMessage(content='你是理赔专家，只输出符合Schema的JSON，不编造证据。Schema：' + schema_text),
                    HumanMessage(content=prompt_text)]
        adapter = _RouterModelAdapter(self.router, EXPERT_TASK[role], claim,
                                      lambda: self._rule_expert(role))
        try:
            result = await structured_call(adapter, ExpertJudgment, messages)
            opinion = result.value
        except StructuredOutputError as exc:
            opinion = ExpertJudgment.model_validate_json(self._rule_expert(role))
            degraded = True
            parse_attempts = exc.attempts
        else:
            degraded = bool(adapter.evidence and adapter.evidence[-1].get('degraded'))
            parse_attempts = result.attempts
        if opinion.expert != role:
            # 角色不符按解析失败处理，走规则兜底。
            opinion = ExpertJudgment.model_validate_json(self._rule_expert(role))
            degraded = True
        output = opinion.model_dump()
        evidence = adapter.evidence[-1] if adapter.evidence else {}
        return output | {'model_tier': evidence.get('tier', 'unknown'),
                         'model_chain': evidence.get('chain', []),
                         'prompt_version': f"{prompt_meta['version']}@{prompt_meta['content_hash']}",
                         'attempts': parse_attempts, 'degraded': degraded,
                         'rule_based': bool(evidence.get('rule_based')), 'demo': False,
                         'route': evidence,
                         'fallback_events': evidence.get('fallback_events', []),
                         'parse_retries': parse_attempts - 1}


__all__ = ['JudgeServices', 'OfflineJudgeServices', 'LiveJudgeServices',
           'EXPERT_ROLES', 'EXPERT_TASK', 'ROLE_CN', '_exclusion_hit']
