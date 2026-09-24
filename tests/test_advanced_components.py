"""进阶组件单测：路由降级、结构化重试、Prompt版本、条款RAG、模拟保单API工具。"""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from advanced.clause_store import ClauseStore, build_clause_query
from advanced.model_router import ModelRouter, complexity_signals
from advanced.mock_policy_api import create_app, start_background_server
from advanced.policy_api_tool import PolicyApiClient, build_policy_api_tools
from advanced.prompt_versioning import PromptSpec, PromptVersionManager
from advanced.structured_retry import StructuredOutputError, structured_call


# ---------- 模型路由与降级 ----------

class FakeModel:
    def __init__(self, fail: bool = False, content: str = '{}') -> None:
        self.fail, self.content = fail, content

    async def ainvoke(self, messages):
        if self.fail:
            raise RuntimeError('模拟模型不可用')
        return AIMessage(content=self.content)


def test_route_by_task_profile_and_escalation():
    router = ModelRouter(lambda tier: FakeModel())
    decision = router.route('classification')
    assert decision.tier == 'fast' and decision.chain == ('fast',)
    # 大额案件把判责任务升档到pro，且降级链保持有序去重。
    claim = {'description': '重大事故', 'amount': 120000}
    decision = router.route('expert_liability', claim=claim)
    assert decision.tier == 'pro' and decision.escalated
    assert decision.chain == ('pro', 'main', 'fast')
    signals = complexity_signals(claim)
    assert signals['complex'] and signals['high_amount']


def test_fallback_chain_degrades_to_fast_then_rules():
    calls = []

    def factory(tier: str):
        calls.append(tier)
        # pro和main都失败，fast成功。
        return FakeModel(fail=tier in ('pro', 'main'), content='降级后结果')

    router = ModelRouter(factory)
    result = asyncio.run(router.ainvoke('expert_liability', [HumanMessage(content='分析')]))
    assert result.content == '降级后结果' and result.attempts == 3
    assert not result.rule_based
    assert [event.from_tier for event in result.fallback_events] == ['pro', 'main']
    assert calls == ['pro', 'main', 'fast']


def test_rule_fallback_when_whole_chain_fails():
    router = ModelRouter(lambda tier: FakeModel(fail=True))
    result = asyncio.run(router.ainvoke('expert_risk', [HumanMessage(content='分析')],
                                        rule_fallback=lambda: '{"rule": true}'))
    assert result.rule_based and result.degraded
    assert json.loads(result.content) == {'rule': True}
    assert len(result.fallback_events) == 3  # pro→main→fast→rules


def test_router_rejects_unknown_task():
    with pytest.raises(ValueError, match='未知路由任务'):
        ModelRouter(lambda tier: FakeModel()).route('nonexistent')


# ---------- 结构化输出校验重试 ----------

class SampleSchema(BaseModel):
    claim_id: str
    amount: float = Field(ge=0)


class FlakyJsonModel:
    """第一次输出坏JSON，收到反馈后输出合法JSON。"""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content='这不是JSON，而是模型的自由发挥')
        assert '校验错误' in messages[-1].content, '反馈消息必须包含校验错误'
        return AIMessage(content=json.dumps({'claim_id': 'CLM-1', 'amount': 100}))


def test_structured_retry_recovers_with_feedback():
    model = FlakyJsonModel()
    result = asyncio.run(structured_call(model, SampleSchema, [HumanMessage(content='初判')]))
    assert result.attempts == 2 and result.value.claim_id == 'CLM-1'
    assert result.errors and '校验' not in result.value.claim_id


def test_structured_retry_raises_after_limit():
    model = FlakyJsonModel()
    model_content = ['坏输出'] * 3

    async def always_bad(_messages):
        return AIMessage(content=model_content.pop(0))

    model.ainvoke = always_bad  # type: ignore[method-assign]
    with pytest.raises(StructuredOutputError) as excinfo:
        asyncio.run(structured_call(model, SampleSchema, [HumanMessage(content='初判')],
                                    max_feedback_retries=2))
    assert excinfo.value.attempts == 3


def test_structured_retry_invalid_limit():
    with pytest.raises(ValueError, match='0到3'):
        asyncio.run(structured_call(FakeModel(), SampleSchema, [], max_feedback_retries=9))


# ---------- Prompt版本管理 ----------

def test_prompt_register_activate_rollback(tmp_path):
    manager = PromptVersionManager(specs=[
        PromptSpec('demo', '1.0.0', '你好{name}', '基线', 'active', ('name',)),
    ], directory=tmp_path)
    manager.register('demo', '1.1.0', '您好，{name}！', '语气优化')
    assert manager.get('demo').version == '1.0.0', '新版本默认canary不直接生效'
    manager.activate('demo', '1.1.0')
    assert manager.get('demo').version == '1.1.0'
    assert manager.get('demo', '1.0.0').status == 'retired'
    rolled = manager.rollback('demo')
    assert rolled.version == '1.0.0', '回滚后1.0.0重新激活'
    text, meta = manager.render('demo', {'name': '张三'}, context='test')
    assert '张三' in text and meta['version'] == '1.0.0' and meta['content_hash']
    assert manager.usage_log and manager.usage_log[-1]['context'] == 'test'


def test_prompt_version_immutable_and_persisted(tmp_path):
    manager = PromptVersionManager(specs=[
        PromptSpec('demo', '1.0.0', '模板A：{value}', '基线', 'active', ('value',)),
    ], directory=tmp_path)
    with pytest.raises(ValueError, match='已存在'):
        manager.register('demo', '1.0.0', '模板B', '试图覆盖')
    reloaded = PromptVersionManager(directory=tmp_path)
    assert reloaded.get('demo').template == '模板A：{value}'
    with pytest.raises(ValueError, match='渲染缺少变量'):
        reloaded.render('demo', {})


def test_prompt_builtin_versions_available():
    manager = PromptVersionManager()
    assert manager.get('expert').version == '1.0.0'
    assert any(spec['version'] == '1.1.0' and spec['status'] == 'canary'
               for spec in manager.list_versions('expert'))


# ---------- 条款RAG ----------

def test_clause_retrieval_ranks_relevant_exclusion_first():
    store = ClauseStore()
    hits = store.retrieve('醉酒驾驶出险 血液酒精含量超标 免责', k=3)
    assert hits[0].metadata['article'] == '第五条'
    assert hits[0].metadata['kind'] == 'exclusion'
    assert all('doc_id' in doc.metadata for doc in hits)


def test_clause_query_builder_injects_policy_signals():
    claim = {'description': '路边剐蹭'}
    policy = {'status': '已过期', 'exclusion_tags': ['expired_policy'], 'claims_90d': 0}
    query = build_clause_query(claim, policy)
    assert '效力终止' in query
    hits = ClauseStore().retrieve(query, k=2)
    assert hits[0].metadata['article'] == '第二条'


# ---------- 模拟保单API与工具 ----------

@pytest.fixture(scope='module')
def policy_api():
    server = start_background_server(create_app(), port=8123)
    import time
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield 'http://127.0.0.1:8123'
    server.should_exit = True


def test_policy_api_client_and_tool(policy_api):
    client = PolicyApiClient(policy_api)
    record = client.get_policy('POL-2024-001')
    assert record['status'] == '有效' and record['coverage_limit'] == 500000
    tools = build_policy_api_tools(client)
    result = tools[0].invoke({'policy_id': 'POL-2024-003'})
    assert result['exclusion_tags'] == ['dui_exclusion', 'license_suspended']
    history = client.get_claim_history('POL-2024-003')
    assert len(history['records']) == 3


def test_policy_api_404_not_retried_and_wrapped(policy_api):
    client = PolicyApiClient(policy_api, retries=2)
    from advanced.policy_api_tool import PolicyApiError
    with pytest.raises(PolicyApiError) as excinfo:
        client.get_policy('POL-NOT-EXIST')
    assert excinfo.value.status_code == 404
    tool = build_policy_api_tools(client)[0]
    from langchain_core.tools import ToolException
    with pytest.raises(ToolException):
        tool.invoke({'policy_id': 'POL-NOT-EXIST'})
