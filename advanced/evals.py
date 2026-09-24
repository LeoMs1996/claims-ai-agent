"""方向三：判责链路评测（Eval）。

- 数据集：evals/judge_dataset.jsonl，黄金标注三分流（auto/review/reject）。
- 离线模式：用OfflineJudgeServices跑全图，验证"专家信号→融合→阈值分流"
  的管线回归（阈值改动/护栏逻辑破坏会被抓住）。
- live模式（--live）：同一数据集跑真实模型链路，度量模型判责与黄金标注的一致率。
- 配置LANGSMITH_API_KEY时自动把数据集与逐案结果上传LangSmith数据集/实验。
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from advanced.clause_store import ClauseStore
from advanced.judge_graph import build_judge_graph, initial_state
from advanced.model_router import ModelRouter
from advanced.prompt_versioning import PromptVersionManager
from advanced.services import LiveJudgeServices, OfflineJudgeServices
from advanced.tracing import langsmith_enabled

LOGGER = logging.getLogger('claims.advanced.evals')

DATASET_PATH = Path(__file__).resolve().parent.parent / 'evals' / 'judge_dataset.jsonl'
BANDS = ('auto', 'review', 'reject')
DECISION_TO_BAND = {'accept': 'auto', 'review': 'review', 'reject': 'reject'}


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    seen: set[str] = set()
    for case in cases:
        missing = {'claim_id', 'scenario', 'golden_band', 'description'} - case.keys()
        if missing:
            raise ValueError(f'{path.name}存在缺失字段案例：{missing}')
        if case['claim_id'] in seen:
            raise ValueError(f'数据集claim_id重复：{case["claim_id"]}')
        seen.add(case['claim_id'])
        if case['golden_band'] not in BANDS:
            raise ValueError(f'非法golden_band：{case["golden_band"]}')
    return cases


def build_offline_services(scenario: str, clause_store: ClauseStore,
                           router: ModelRouter, prompts: PromptVersionManager,
                           policy_client: Any = None) -> OfflineJudgeServices:
    return OfflineJudgeServices(scenario, clause_store=clause_store,
                                policy_client=policy_client, router=router, prompts=prompts)


async def run_case(case: dict[str, Any], services: Any, *, thread: str = 'eval') -> dict[str, Any]:
    graph = build_judge_graph(services)
    config = {'configurable': {'thread_id': f"{thread}:{case['claim_id']}"}, 'recursion_limit': 30}
    state = await graph.ainvoke(
        initial_state(case['claim_id'], case['description'], case.get('policy_id'),
                      case.get('amount')), config=config)
    verdict = state['verdict']
    predicted = DECISION_TO_BAND.get(verdict['decision'], 'review')
    return {'claim_id': case['claim_id'], 'golden': case['golden_band'], 'predicted': predicted,
            'correct': predicted == case['golden_band'],
            'decision': verdict['decision'], 'confidence': verdict['confidence'],
            'band': verdict.get('band'), 'message': verdict.get('message', '')}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    confusion = {gold: {pred: 0 for pred in BANDS} for gold in BANDS}
    for r in results:
        confusion[r['golden']][r['predicted']] += 1
    per_band = {}
    for band in BANDS:
        gold_count = sum(1 for r in results if r['golden'] == band)
        if gold_count:
            precision_denominator = sum(1 for r in results if r['predicted'] == band) or 1
            per_band[band] = {
                'support': gold_count,
                'recall': round(sum(1 for r in results if r['golden'] == band and r['correct']) / gold_count, 4),
                'precision': round(sum(1 for r in results if r['predicted'] == band and r['correct']) / precision_denominator, 4)}
    return {'total': total, 'correct': correct, 'accuracy': round(correct / total, 4) if total else 0.0,
            'confusion_matrix(golden→predicted)': confusion, 'per_band': per_band}


def upload_to_langsmith(cases: list[dict[str, Any]], results: list[dict[str, Any]],
                        summary: dict[str, Any], *, dataset_name: str = 'claims_judge_eval') -> bool:
    """尽力上传LangSmith：数据集+逐案运行记录；失败不阻断本地评测。"""
    if not langsmith_enabled():
        return False
    try:
        from langsmith import Client
        client = Client()
        dataset = client.create_dataset(dataset_name=dataset_name,
                                        description='进阶判责链路评测数据集（三分流黄金标注）')
        client.create_examples(
            inputs=[{'claim_id': c['claim_id'], 'description': c['description'],
                     'policy_id': c.get('policy_id'), 'amount': c.get('amount')} for c in cases],
            outputs=[{'golden_band': c['golden_band'], 'golden_decision': c['golden_decision']}
                     for c in cases],
            dataset_id=dataset.id)
        for result in results:
            client.create_run(name='judge_eval_case', run_type='chain',
                              inputs={'claim_id': result['claim_id']},
                              outputs={'predicted': result['predicted'], 'golden': result['golden'],
                                       'correct': result['correct'], 'confidence': result['confidence']})
        client.create_run(name='judge_eval_summary', run_type='chain', inputs={},
                          outputs={'summary': {k: v for k, v in summary.items()}})
        LOGGER.info('评测结果已上传LangSmith数据集：%s', dataset_name)
        return True
    except Exception as exc:  # noqa: BLE001 - 上传失败不影响本地结论
        LOGGER.warning('LangSmith上传失败（不影响本地评测）：%s: %s', type(exc).__name__, exc)
        return False


async def evaluate(*, live: bool = False, policy_client: Any = None,
                   model_factory: Any = None, upload: bool = True) -> dict[str, Any]:
    """跑完整评测并返回报告；离线确定性，live依赖配置的模型端点。"""
    cases = load_dataset()
    clause_store = ClauseStore()
    prompts = PromptVersionManager()
    results = []
    for case in cases:
        if live:
            if model_factory is None:
                raise ValueError('live评测需要model_factory')
            router = ModelRouter(model_factory)
            services = LiveJudgeServices(model_factory=model_factory, clause_store=clause_store,
                                         policy_client=policy_client, router=router, prompts=prompts)
        else:
            router = ModelRouter(lambda tier: None)  # 离线路由只做决策记录，不调用模型
            services = build_offline_services(case['scenario'], clause_store, router, prompts,
                                              policy_client=policy_client)
        results.append(await run_case(case, services))
    summary = summarize(results)
    if upload:
        summary['langsmith_uploaded'] = upload_to_langsmith(cases, results, summary)
    return {'summary': summary, 'results': results}


def format_report(report: dict[str, Any]) -> str:
    summary, results = report['summary'], report['results']
    lines = ['=== 判责链路评测报告 ===',
             f"准确率：{summary['correct']}/{summary['total']} = {summary['accuracy']:.1%}",
             f"LangSmith上传：{summary.get('langsmith_uploaded', False)}", '',
             '混淆矩阵（行=黄金，列=预测）：']
    header = '\t'.join(BANDS)
    lines.append(f'\t{header}')
    for gold in BANDS:
        row = summary['confusion_matrix(golden→predicted)'][gold]
        lines.append(f'{gold}\t' + '\t'.join(str(row[pred]) for pred in BANDS))
    lines.append('')
    for result in results:
        mark = '✓' if result['correct'] else '✗'
        lines.append(f"{mark} {result['claim_id']} 黄金={result['golden']} 预测={result['predicted']}"
                     f"（{result['decision']}，置信度{result['confidence']}）")
    return '\n'.join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='使用真实模型链路（需配置模型端点）')
    parser.add_argument('--no-upload', action='store_true')
    args = parser.parse_args()
    model_factory = None
    if args.live:
        from config import ClaimLLMFactory
        model_factory = lambda tier: ClaimLLMFactory.create(tier, max_retries=0)  # noqa: E731
    report = asyncio.run(evaluate(live=args.live, model_factory=model_factory, upload=not args.no_upload))
    print(format_report(report))


if __name__ == '__main__':
    main()
