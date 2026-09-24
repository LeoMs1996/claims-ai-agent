"""方向二核心：LangGraph多节点判责工作流 + 置信度阈值三分流。

课堂版 agents/claim_agent.build_claim_graph 已是LangGraph状态机（Send并行
三专家 + HITL），本工作流在其基础上叠加进阶改造：

节点链：classify → policy_lookup → rag_retrieve → experts(并行三专家)
        → fuse_confidence → [置信度分流] → auto_approve / human_review /
        auto_reject → finalize

分流规则（第四章核心逻辑，阈值可配置）：
- 高置信度（≥ high，默认0.85）→ 自动判责（受理建议，不执行付款）
- 中置信度（≥ mid，默认0.5） → 转人工审核（入队，可interrupt恢复）
- 低置信度（< mid）          → 拒赔建议（必须引用免责条款，否则升级人工）

合规护栏（优先于阈值，全部记录在triage_reason）：
- 保单查询降级/系统错误 → 人工；专家建议调查 → 至多中置信；
- 缺失材料 → 至多中置信（0.75上限）；超保额 → 至多中置信（0.80上限）；
- 保单失效/免责命中 → 至多低置信（0.45上限，配合条款引用走拒赔建议）。
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from advanced.clause_store import ClauseStore
from advanced.services import EXPERT_ROLES, JudgeServices, _exclusion_hit

DEFAULT_HIGH_THRESHOLD = 0.85
DEFAULT_MID_THRESHOLD = 0.50
CAP_MISSING_MATERIALS = 0.75
CAP_OVER_COVERAGE = 0.80
CAP_INVESTIGATE = 0.80
CAP_POLICY_INVALID = 0.45


class JudgeState(TypedDict, total=False):
    """判责工作流状态；专家结果用operator.add汇聚并行Send结果。"""
    claim_id: str
    claim_data: dict[str, Any]
    classification: dict[str, Any]
    policy_info: dict[str, Any]
    policy_degraded: bool
    policy_error: str
    clause_hits: list[dict[str, Any]]
    clause_context: str
    expert_results: Annotated[list[dict[str, Any]], operator.add]
    route_decisions: Annotated[list[dict[str, Any]], operator.add]
    fallback_events: Annotated[list[dict[str, Any]], operator.add]
    retry_events: Annotated[list[dict[str, Any]], operator.add]
    confidence: float
    confidence_components: dict[str, Any]
    band: str
    triage_reason: str
    decision: str
    phase: str
    message: str
    citations: list[dict[str, Any]]
    human_queue: dict[str, Any]
    verdict: dict[str, Any]
    demo: bool
    error: str


@dataclass(frozen=True)
class TriageThresholds:
    """分流阈值；生产按回测数据校准，不拍脑袋。"""
    high: float = DEFAULT_HIGH_THRESHOLD
    mid: float = DEFAULT_MID_THRESHOLD

    def __post_init__(self) -> None:
        if not 0 < self.mid < self.high < 1:
            raise ValueError(f'阈值必须满足 0 < mid < high < 1，当前：{self.mid}/{self.high}')


def fuse_confidence(experts: list[dict[str, Any]], claim: dict[str, Any],
                    policy: dict[str, Any] | None, *, policy_degraded: bool = False,
                    thresholds: TriageThresholds | None = None) -> dict[str, Any]:
    """置信度融合：最弱专家为基线 + 一致性修正 + 合规护栏上限。

    返回 confidence/components/band/triage_reason，供图状态与审计使用。
    """
    thresholds = thresholds or TriageThresholds()
    confs = [float(expert.get('confidence', 0)) for expert in experts]
    base = min(confs) if confs else 0.0
    recommendations = [str(expert.get('recommendation')) for expert in experts]
    unique = len(set(recommendations)) if recommendations else 0
    agreement_adjustment = {0: 0.0, 1: 0.05, 2: -0.10, 3: -0.20}[min(unique, 3)]
    score = base + agreement_adjustment

    caps: list[tuple[str, float]] = []
    if policy_degraded:
        caps.append(('保单查询降级，无法核实保单有效性', CAP_MISSING_MATERIALS))
    elif policy and policy.get('status') and policy['status'] != '有效':
        caps.append((f"保单状态={policy['status']}，保险责任无法成立", CAP_POLICY_INVALID))
    if any(expert.get('missing_information') for expert in experts):
        caps.append(('专家指出材料缺失', CAP_MISSING_MATERIALS))
    amount = float(claim.get('amount') or 0)
    if policy and amount > float(policy.get('coverage_limit') or 0):
        caps.append(('索赔金额超过保单限额，超额部分需协商', CAP_OVER_COVERAGE))
    if 'investigate' in recommendations:
        caps.append(('有专家建议调查核实', CAP_INVESTIGATE))
    hit = _exclusion_hit(claim, policy)
    if hit:
        caps.append((f'报案描述命中保单免责标记（{hit}）', CAP_POLICY_INVALID))

    final = min([score, *[cap for _, cap in caps]])
    final = max(0.0, min(1.0, final))

    # 分流：阈值 + 高置信额外要求三方一致受理。
    if final >= thresholds.high and recommendations and all(rec == 'accept' for rec in recommendations):
        band, reason = 'high', f'置信度{final:.2f}≥{thresholds.high}且三专家一致受理'
    elif final >= thresholds.mid:
        band, reason = 'mid', (f'置信度{thresholds.mid}≤{final:.2f}<{thresholds.high}，转人工审核')
        if final >= thresholds.high:
            reason = f'置信度{final:.2f}虽达高线但专家意见不一致，降级转人工'
    else:
        band, reason = 'low', f'置信度{final:.2f}<{thresholds.mid}，形成拒赔建议'
    if caps:
        reason += '；护栏：' + '；'.join(name for name, _ in caps)

    return {'confidence': round(final, 4), 'band': band, 'triage_reason': reason,
            'confidence_components': {
                'base_min_expert': round(base, 4),
                'agreement_adjustment': agreement_adjustment,
                'expert_confidences': [round(c, 3) for c in confs],
                'recommendations': recommendations,
                'caps_applied': [{'reason': name, 'cap': cap} for name, cap in caps],
            }}


def build_judge_graph(services: JudgeServices, *,
                      thresholds: TriageThresholds | None = None,
                      human_review_mode: str = 'queue',
                      checkpointer: Any = None) -> Any:
    """构建进阶判责图。

    human_review_mode：
    - 'queue'（默认）：中置信度生成本地待审队列载荷后结束（demo/评测用）；
    - 'interrupt'：LangGraph interrupt挂起，等Command(resume=...)人工决定。
    """
    if human_review_mode not in ('queue', 'interrupt'):
        raise ValueError(f'未知人工审核模式：{human_review_mode}')
    thresholds = thresholds or TriageThresholds()
    graph = StateGraph(JudgeState)

    # -- 1. 分类节点（flash档；复杂度信号供模型路由升档） ----------------
    async def classify(state: JudgeState) -> dict[str, Any]:
        claim = state['claim_data']
        result = await services.classify(claim)
        routes = [result['route']] if result.get('route') else []
        return {'classification': result, 'demo': services.demo,
                'route_decisions': routes}

    # -- 2. 保单查询节点（新工具：模拟保单API） --------------------------
    async def policy_lookup(state: JudgeState) -> dict[str, Any]:
        claim = state['claim_data']
        if not claim.get('policy_id'):
            return {'policy_degraded': True, 'policy_error': '报案缺少保单号',
                    'policy_info': {}, 'phase': 'policy_missing'}
        try:
            info = await services.policy_lookup(claim)
            return {'policy_info': dict(info, query_source='模拟保单API'), 'policy_degraded': False}
        except Exception as exc:  # noqa: BLE001 - 工具失败必须显式降级，不能假设保单有效
            return {'policy_degraded': True, 'policy_error': f'{type(exc).__name__}: {exc}'[:200],
                    'policy_info': {}}

    # -- 3. RAG条款检索节点（方向二：模块10能力） ------------------------
    def rag_retrieve(state: JudgeState) -> dict[str, Any]:
        documents = services.retrieve_clauses(state['claim_data'], state.get('policy_info') or None)
        citations = ClauseStore.citations(documents)
        return {'clause_hits': citations,
                'clause_context': ClauseStore.format_for_prompt(documents)}

    # -- 4. 三专家并行（Send扇出，复用课堂版的并行模式） ------------------
    def dispatch_experts(state: JudgeState) -> list[Send]:
        return [Send('expert', {'role': role, 'claim_data': state['claim_data'],
                                'policy_info': state.get('policy_info') or {},
                                'policy_degraded': state.get('policy_degraded', False),
                                'clause_context': state.get('clause_context', ''),
                                'classification': state.get('classification', {})})
                for role in EXPERT_ROLES]

    async def expert(state: dict[str, Any]) -> dict[str, Any]:
        try:
            result = await services.run_expert(
                state['role'], state['claim_data'], state['policy_info'] or None,
                state.get('clause_context', ''), state.get('classification', {}))
        except Exception as exc:  # noqa: BLE001 - 单专家失败不拖垮整案，按保守意见继续
            result = {'expert': state['role'], 'recommendation': 'review', 'confidence': 0.5,
                      'rationale': f'专家调用失败：{type(exc).__name__}', 'clause_ids': [],
                      'missing_information': ['专家链路异常'], 'model_tier': 'unknown',
                      'attempts': 0, 'degraded': True, 'rule_based': False,
                      'fallback_events': [{'from': state['role'], 'to': 'error',
                                           'error': type(exc).__name__}]}
        events = []
        retries = []
        routes = []
        if result.get('route'):
            routes.append(result['route'])
        events.extend(result.get('fallback_events', []))
        if result.get('parse_retries'):
            retries.append({'node': f"expert:{result['expert']}", 'kind': 'structured_retry',
                            'retries': result['parse_retries']})
        return {'expert_results': [result],
                'route_decisions': routes, 'fallback_events': events, 'retry_events': retries}
    def aggregate(state: JudgeState) -> dict[str, Any]:
        fused = fuse_confidence(state.get('expert_results', []), state['claim_data'],
                                state.get('policy_info') or None,
                                policy_degraded=state.get('policy_degraded', False),
                                thresholds=thresholds)
        return {'confidence': fused['confidence'], 'band': fused['band'],
                'triage_reason': fused['triage_reason'],
                'confidence_components': fused['confidence_components']}

    # -- 5. 分流条件边（方向二：置信度阈值三分流） ------------------------
    def route_triage(state: JudgeState) -> str:
        if state.get('error') or state.get('policy_degraded'):
            # 工具降级属于系统风险，人工优先于任何自动决定。
            return 'human_review'
        return {'high': 'auto_approve', 'mid': 'human_review', 'low': 'auto_reject'}[state['band']]

    # -- 6. 三个出口节点 --------------------------------------------------
    def auto_approve(state: JudgeState) -> dict[str, Any]:
        return {'decision': 'accept', 'phase': 'complete',
                'message': f"高置信度自动判责：形成受理建议（置信度{state['confidence']:.2f}），未执行付款。",
                'citations': [hit for hit in state.get('clause_hits', [])
                              if hit['doc_id'] in {cid for expert_result in state.get('expert_results', [])
                                                   for cid in expert_result.get('clause_ids', [])}]}

    def human_review(state: JudgeState) -> dict[str, Any]:
        queue = {'claim_id': state['claim_id'], 'band': state.get('band', 'mid'),
                 'confidence': state.get('confidence'),
                 'reason': state.get('triage_reason') or state.get('policy_error', '需人工核实'),
                 'expert_summary': [{'expert': item['expert'], 'recommendation': item['recommendation'],
                                     'confidence': item['confidence'],
                                     'missing': item.get('missing_information', [])}
                                    for item in state.get('expert_results', [])],
                 'suggested_checks': ['补充核实保单状态', '核对缺失材料', '复核专家分歧点'],
                 'policy_error': state.get('policy_error', '')}
        if human_review_mode == 'queue':
            return {'decision': 'review', 'phase': 'awaiting_human',
                    'message': '中置信度/护栏触发：已生成人工审核任务并入队。',
                    'human_queue': queue, 'citations': state.get('clause_hits', [])[:2]}
        answer = interrupt({'kind': 'judge_review', **{k: v for k, v in queue.items() if k != 'suggested_checks'}})
        if not isinstance(answer, dict) or answer.get('decision') not in ('review', 'accept', 'reject'):
            raise ValueError('人工审核决定不合法')
        if not answer.get('reviewer') or not answer.get('reason'):
            raise ValueError('人工审核必须记录审核员与理由')
        if answer['decision'] == 'reject' and not answer.get('clause_basis'):
            raise ValueError('拒赔必须提供条款依据')
        return {'decision': answer['decision'], 'phase': 'complete',
                'human_queue': queue | {'resolved_by': answer['reviewer'], 'resolution': answer},
                'message': f"人工审核完成：{answer['decision']}（{answer['reason']}）"}

    def auto_reject(state: JudgeState) -> dict[str, Any]:
        # 合规护栏：自动拒赔必须能引用免责/效力类条款，否则升级人工。
        exclusions = [hit for hit in state.get('clause_hits', []) if hit['kind'] == 'exclusion']
        if not exclusions:
            return {'decision': 'review', 'phase': 'awaiting_human',
                    'message': '低置信度但未检索到免责条款依据，拒赔转人工核实（不能无依据拒赔）。',
                    'human_queue': {'claim_id': state['claim_id'], 'band': 'low',
                                    'confidence': state.get('confidence'),
                                    'reason': state.get('triage_reason', ''),
                                    'suggested_checks': ['人工核实免责条款适用性']}}
        supported = {cid for expert_result in state.get('expert_results', [])
                     for cid in expert_result.get('clause_ids', [])}
        citations = [hit for hit in exclusions if hit['doc_id'] in supported] or exclusions[:2]
        return {'decision': 'reject', 'phase': 'complete',
                'message': (f"低置信度拒赔建议（置信度{state['confidence']:.2f}），引用免责条款："
                            + '、'.join(hit['article'] for hit in citations)
                            + '。真实生产仍需授权人工复核后发出。'),
                'citations': citations}

    # -- 7. 定稿节点：输出完整可审计判决记录 ------------------------------
    def finalize(state: JudgeState) -> dict[str, Any]:
        verdict = {'claim_id': state['claim_id'], 'decision': state.get('decision'),
                   'phase': state.get('phase'), 'band': state.get('band'),
                   'confidence': state.get('confidence'),
                   'triage_reason': state.get('triage_reason'),
                   'confidence_components': state.get('confidence_components'),
                   'citations': state.get('citations', []),
                   'model_routing': state.get('route_decisions', []),
                   'fallback_events': state.get('fallback_events', []),
                   'prompt_versions': sorted({item['prompt_version'] for item in state.get('expert_results', [])
                                              if item.get('prompt_version') and item['prompt_version'] != '未配置'}),
                   'expert_opinions': [{'expert': item.get('expert'),
                                        'recommendation': item.get('recommendation'),
                                        'confidence': item.get('confidence'),
                                        'tier': item.get('model_tier'),
                                        'degraded': item.get('degraded', False)}
                                       for item in state.get('expert_results', [])],
                   'human_queue': state.get('human_queue'),
                   'message': state.get('message', ''), 'demo': state.get('demo', False)}
        return {'verdict': verdict}

    graph.add_node('classify', classify)
    graph.add_node('policy_lookup', policy_lookup)
    graph.add_node('rag_retrieve', rag_retrieve)
    graph.add_node('expert', expert)
    graph.add_node('aggregate', aggregate)
    graph.add_node('auto_approve', auto_approve)
    graph.add_node('human_review', human_review)
    graph.add_node('auto_reject', auto_reject)
    graph.add_node('finalize', finalize)

    graph.add_edge(START, 'classify')
    graph.add_edge('classify', 'policy_lookup')
    graph.add_edge('policy_lookup', 'rag_retrieve')
    graph.add_conditional_edges('rag_retrieve', dispatch_experts, ['expert'])
    graph.add_edge('expert', 'aggregate')
    graph.add_conditional_edges('aggregate', route_triage,
                                {'auto_approve': 'auto_approve', 'human_review': 'human_review',
                                 'auto_reject': 'auto_reject'})
    for node in ('auto_approve', 'human_review', 'auto_reject'):
        graph.add_edge(node, 'finalize')
    graph.add_edge('finalize', END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def initial_state(claim_id: str, description: str, policy_id: str | None,
                  amount: float | None = None) -> dict[str, Any]:
    """构造图初始状态。"""
    if not description.strip():
        raise ValueError('报案描述不能为空')
    return {'claim_id': claim_id, 'demo': False,
            'claim_data': {'claim_id': claim_id, 'description': description.strip(),
                           'policy_id': policy_id, 'amount': amount},
            'expert_results': [], 'route_decisions': [], 'fallback_events': [], 'retry_events': []}


__all__ = ['build_judge_graph', 'fuse_confidence', 'TriageThresholds', 'JudgeState',
           'initial_state', 'DEFAULT_HIGH_THRESHOLD', 'DEFAULT_MID_THRESHOLD']
