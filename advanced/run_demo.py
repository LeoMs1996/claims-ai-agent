"""进阶改造演示入口：一条命令跑通三个方向的组合链路。

离线演示（默认，无任何外部依赖）：
    python -m advanced.run_demo
    python -m advanced.run_demo --scenario reject_exclusion
    python -m advanced.run_demo --eval

live演示（需.env配置的模型端点，真实模型路由+降级+结构化重试）：
    python -m advanced.run_demo --live --claim-id CLM-LIVE-001

演示会：启动模拟保单API(8010) → 跑LangGraph判责图 → 打印路由决策、
降级/重试事件、RAG条款引用、置信度融合与三分流结果、本地链路追踪。
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from advanced.clause_store import ClauseStore
from advanced.evals import evaluate as run_eval
from advanced.evals import format_report as format_eval_report
from advanced.judge_graph import TriageThresholds, build_judge_graph, initial_state
from advanced.model_router import ModelRouter
from advanced.mock_policy_api import create_app, start_background_server, wait_until_ready
from advanced.policy_api_tool import PolicyApiClient
from advanced.prompt_versioning import PromptVersionManager
from advanced.services import LiveJudgeServices, OfflineJudgeServices
from advanced.tracing import LocalTraceCollector, setup_tracing

DEFAULT_SCENARIOS = {
    'auto': ('CLM-ADV-001', 'POL-2024-001', 1200.0,
             '停车场低速倒车刮擦护栏，无人员受伤。维修发票、损失清单、事故证明材料均已提交齐全，申请车辆维修理赔。'),
    'review_missing': ('CLM-ADV-003', 'POL-2024-001', 8000.0,
                       '夜间小区内双方轻微碰撞，责任明确，但维修发票尚未提交。'),
    'reject_expired': ('CLM-ADV-004', 'POL-2024-002', 5600.0,
                       '车辆停在路边被剐蹭，保单已过期未续保，申请理赔。'),
    'reject_exclusion': ('CLM-ADV-005', 'POL-2024-003', 9800.0,
                         '驾驶人醉酒驾驶发生单方事故撞上护栏，血液酒精含量超标。'),
    'risk_review': ('CLM-ADV-006', 'POL-2024-003', 4500.0,
                    '三个月内第三次出险报案，事故经过与前两次高度相似，反欺诈需调查。'),
    'over_limit': ('CLM-ADV-007', 'POL-2024-004', 600000.0,
                   '重大交通事故车辆全损，维修与施救费用远超保单限额，涉及诉讼风险。'),
}


def section(title: str) -> None:
    print('\n' + '=' * 12 + f' {title} ' + '=' * 12)


def print_verdict(verdict: dict[str, Any]) -> None:
    print(f"判决：{verdict['decision']}（band={verdict['band']}，置信度={verdict['confidence']}）")
    print(f"分流理由：{verdict['triage_reason']}")
    print(f"说明：{verdict['message']}")
    suffix = '（专家意见为离线合成，未调用模型）' if verdict['demo'] else '（真实模型链路输出）'
    print(f"演示模式：{verdict['demo']}{suffix}")
    print('\n置信度融合：')
    print(json.dumps(verdict['confidence_components'], ensure_ascii=False, indent=2))
    print('\n专家意见与模型档位：')
    for opinion in verdict['expert_opinions']:
        degrade = ' [降级]' if opinion.get('degraded') else ''
        print(f"  - {opinion['expert']}: {opinion['recommendation']}"
              f"（置信度{opinion['confidence']}，tier={opinion.get('tier')}{degrade}）")
    print('\n模型路由决策：')
    for route in verdict['model_routing']:
        print(f"  - {route['task']} → {route['tier']}（链{route['chain']}，"
              f"升档={route['escalated']}）{route['reason'][:60]}")
    if verdict['fallback_events']:
        print('\n降级/重试事件：')
        print(json.dumps(verdict['fallback_events'], ensure_ascii=False, indent=2))
    if verdict['prompt_versions']:
        print('\nPrompt版本：', verdict['prompt_versions'])
    if verdict['citations']:
        print('\n条款引用：')
        for citation in verdict['citations']:
            print(f"  - {citation['article']}《{citation['title']}》[{citation['doc_id']}]：{citation['excerpt'][:60]}…")
    if verdict.get('human_queue'):
        print('\n人工审核队列载荷：')
        print(json.dumps(verdict['human_queue'], ensure_ascii=False, indent=2)[:800])


async def run_single(scenario: str, *, live: bool, api_url: str, port: int,
                     high: float, mid: float, show_trace: bool) -> dict[str, Any]:
    claim_id, policy_id, amount, description = DEFAULT_SCENARIOS[scenario]
    clause_store = ClauseStore()
    prompts = PromptVersionManager()
    client = PolicyApiClient(api_url)
    collector = LocalTraceCollector()
    try:
        if live:
            from config import ClaimLLMFactory
            model_factory = lambda tier: ClaimLLMFactory.create(tier, max_retries=0)  # noqa: E731
            router = ModelRouter(model_factory)
            services = LiveJudgeServices(model_factory=model_factory, clause_store=clause_store,
                                         policy_client=client, router=router, prompts=prompts)
        else:
            router = ModelRouter(lambda tier: None)
            services = OfflineJudgeServices(scenario, clause_store=clause_store,
                                            policy_client=client, router=router, prompts=prompts)
        graph = build_judge_graph(services, thresholds=TriageThresholds(high=high, mid=mid))
        config = {'configurable': {'thread_id': claim_id}, 'recursion_limit': 30}
        if show_trace:
            config['callbacks'] = [collector]
        state = await graph.ainvoke(initial_state(claim_id, description, policy_id, amount),
                                    config=config)
        section(f'场景 {scenario} 判决结果')
        print_verdict(state['verdict'])
        if show_trace:
            section('本地链路追踪')
            print(collector.report()[:3000])
        return state['verdict']
    finally:
        client.close()


async def main_async(args: argparse.Namespace) -> None:
    tracing_mode = setup_tracing()
    section('环境')
    print(f'追踪模式：{tracing_mode}'
          + ('（配置LANGSMITH_API_KEY可自动上报LangSmith）' if tracing_mode == 'local' else ''))
    print(f'模式：{"live真实模型" if args.live else "offline离线演示（专家意见为合成数据）"}')

    server = None
    if not args.no_api:
        section('启动模拟保单API')
        server = start_background_server(create_app(), port=args.port)
        wait_until_ready(f'http://127.0.0.1:{args.port}')
        print(f'模拟保单API就绪：http://127.0.0.1:{args.port}（健康检查/保单查询/出险历史/故障注入）')

    try:
        if args.eval:
            section('运行评测数据集')
            client = PolicyApiClient(f'http://127.0.0.1:{args.port}')
            try:
                report = await run_eval(live=args.live, policy_client=client, upload=not args.no_upload)
            finally:
                client.close()
            print(format_eval_report(report))
        else:
            scenarios = [args.scenario] if args.scenario else list(DEFAULT_SCENARIOS)
            for scenario in scenarios:
                await run_single(scenario, live=args.live, api_url=f'http://127.0.0.1:{args.port}',
                                 port=args.port, high=args.high, mid=args.mid, show_trace=args.trace)
    finally:
        if server is not None:
            server.should_exit = True
            print('\n模拟保单API已停止。')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scenario', choices=sorted(DEFAULT_SCENARIOS), help='单场景演示；缺省跑全部场景')
    parser.add_argument('--eval', action='store_true', help='运行evals/judge_dataset.jsonl评测')
    parser.add_argument('--no-upload', action='store_true', help='评测结果不上传LangSmith')
    parser.add_argument('--live', action='store_true', help='使用.env配置的真实模型端点')
    parser.add_argument('--port', type=int, default=8010, help='模拟保单API端口')
    parser.add_argument('--no-api', action='store_true', help='不启动模拟保单API（需自行提供）')
    parser.add_argument('--trace', action='store_true', help='打印本地链路追踪明细')
    parser.add_argument('--high', type=float, default=0.85, help='高置信度阈值（默认0.85）')
    parser.add_argument('--mid', type=float, default=0.50, help='中置信度阈值（默认0.50）')
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
