"""进阶判责工作流测试：三分流、护栏、保单降级、人工队列、HITL与评测。"""
import asyncio

import pytest
from langgraph.types import Command

from advanced.clause_store import ClauseStore
from advanced.evals import evaluate, load_dataset
from advanced.judge_graph import (TriageThresholds, build_judge_graph,
                                  fuse_confidence, initial_state)
from advanced.model_router import ModelRouter
from advanced.prompt_versioning import PromptVersionManager
from advanced.services import OfflineJudgeServices

STORE = ClauseStore()
PROMPTS = PromptVersionManager()


def offline(scenario: str) -> OfflineJudgeServices:
    return OfflineJudgeServices(scenario, clause_store=STORE, router=ModelRouter(lambda tier: None),
                                prompts=PROMPTS)


def run(services, claim_id='CLM-TEST-001', policy_id='POL-2024-001', amount=1000.0,
        description='测试报案', **graph_kwargs):
    graph = build_judge_graph(services, **graph_kwargs)
    config = {'configurable': {'thread_id': claim_id}, 'recursion_limit': 30}
    return asyncio.run(graph.ainvoke(
        initial_state(claim_id, description, policy_id, amount), config=config))


# ---------- 置信度三分流 ----------

def test_high_confidence_auto_accept():
    state = run(offline('auto'))
    assert state['band'] == 'high' and state['decision'] == 'accept'
    assert state['verdict']['model_routing'], '判决记录必须包含路由证据'
    assert state['verdict']['prompt_versions']


def test_mid_confidence_routes_to_human_queue():
    state = run(offline('review_missing'))
    assert state['band'] == 'mid' and state['decision'] == 'review'
    assert state['phase'] == 'awaiting_human'
    queue = state['human_queue']
    assert queue['claim_id'] == 'CLM-TEST-001' and queue['band'] == 'mid'
    assert any(item['missing'] for item in queue['expert_summary'])
    assert state['confidence_components']['caps_applied'], '护栏原因必须可追溯'


def test_low_confidence_reject_with_clause_citations():
    state = run(offline('reject_exclusion'), policy_id='POL-2024-003',
                description='醉酒驾驶发生事故')
    assert state['band'] == 'low' and state['decision'] == 'reject'
    citations = state['citations']
    assert citations and all(c['kind'] == 'exclusion' for c in citations)
    assert citations[0]['article'] in ('第五条', '第六条')


def test_expired_policy_reject_cites_effect_termination():
    state = run(offline('reject_expired'), policy_id='POL-2024-002')
    assert state['decision'] == 'reject'
    assert any(c['article'] == '第二条' for c in state['citations'])


def test_investigate_recommendation_caps_to_human():
    state = run(offline('risk_review'), policy_id='POL-2024-003',
                description='多次出险报案需调查')
    assert state['band'] == 'mid' and state['decision'] == 'review'
    reasons = [cap['reason'] for cap in state['confidence_components']['caps_applied']]
    assert any('调查' in reason for reason in reasons)


def test_over_coverage_amount_capped():
    state = run(offline('over_limit'), policy_id='POL-2024-004', amount=600000.0)
    assert state['band'] == 'mid' and state['confidence'] <= 0.80


# ---------- 融合函数边界 ----------

def test_fuse_confidence_threshold_boundaries():
    experts = [{'confidence': .90, 'recommendation': 'accept', 'missing_information': []},
               {'confidence': .90, 'recommendation': 'accept', 'missing_information': []},
               {'confidence': .90, 'recommendation': 'accept', 'missing_information': []}]
    claim = {'amount': 1000, 'description': '普通案件'}
    policy = {'status': '有效', 'coverage_limit': 500000, 'exclusion_tags': []}
    # 0.90+0.05=0.95 ≥0.85 高置信
    assert fuse_confidence(experts, claim, policy)['band'] == 'high'
    # 高置信但意见分歧：0.90-0.10=0.80 → mid
    experts[0]['recommendation'] = 'review'
    assert fuse_confidence(experts, claim, policy)['band'] == 'mid'
    # 低置信：min=0.4 -0.10 → low
    experts[0]['confidence'] = .4
    assert fuse_confidence(experts, claim, policy)['band'] == 'low'


def test_triage_thresholds_validation():
    with pytest.raises(ValueError, match='阈值'):
        TriageThresholds(high=0.5, mid=0.85)
    # 阈值可调：抬高high线把临界案件挤到人工
    state = run(offline('auto'), thresholds=TriageThresholds(high=0.99, mid=0.5))
    assert state['decision'] == 'review'


# ---------- 工具失败与护栏 ----------

def test_policy_api_failure_degrades_to_human():
    class BrokenClient:
        def get_policy(self, policy_id):
            raise RuntimeError('模拟保单系统宕机')

    services = offline('auto')
    services.policy_client = BrokenClient()
    state = run(services)
    assert state['policy_degraded'] and state['decision'] == 'review'
    assert '宕机' in state['policy_error']


def test_missing_policy_id_degrades():
    state = run(offline('auto'), policy_id=None)
    assert state['decision'] == 'review' and state['phase'] == 'awaiting_human'


def test_auto_reject_without_clause_escalates_to_human():
    """检索不到免责条款时不能自动拒赔：改写条款命中，模拟无依据场景。"""
    services = offline('reject_expired')
    services.retrieve_clauses = lambda claim, policy: []  # 模拟RAG无结果
    state = run(services, policy_id='POL-2024-002')
    assert state['decision'] == 'review'
    assert '拒赔转人工' in state['message']


# ---------- HITL interrupt模式 ----------

def test_human_review_interrupt_and_resume():
    graph = build_judge_graph(offline('review_missing'), human_review_mode='interrupt')
    config = {'configurable': {'thread_id': 'hitl'}, 'recursion_limit': 30}
    asyncio.run(graph.ainvoke(
        initial_state('CLM-HITL-1', '描述', 'POL-2024-001', 500), config=config))
    snapshot = graph.get_state(config)
    assert snapshot.next == ('human_review',)
    with pytest.raises(ValueError, match='拒赔必须提供条款依据'):
        asyncio.run(graph.ainvoke(Command(resume={'decision': 'reject', 'reviewer': 'r1',
                                                  'reason': '无据拒赔'}), config=config))
    state = asyncio.run(graph.ainvoke(
        Command(resume={'decision': 'accept', 'reviewer': '审核员A', 'reason': '材料补齐'}), config=config))
    assert state['decision'] == 'accept' and state['verdict']['human_queue']['resolved_by'] == '审核员A'


# ---------- 评测数据集 ----------

def test_eval_dataset_valid_and_offline_accuracy():
    cases = load_dataset()
    assert len(cases) >= 7 and {'auto', 'review', 'reject'} <= {c['golden_band'] for c in cases}
    report = asyncio.run(evaluate(upload=False))
    assert report['summary']['accuracy'] == 1.0
    matrix = report['summary']['confusion_matrix(golden→predicted)']
    for gold in ('auto', 'review', 'reject'):
        assert sum(matrix[gold].values()) == matrix[gold][gold], '离线管线必须全部命中'
