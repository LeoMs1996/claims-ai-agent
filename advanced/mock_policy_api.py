"""方向一：模拟保单查询 API。

独立 FastAPI 服务，提供多张合成保单、出险历史与故障注入，
用于验证保单查询工具的超时、重试与降级链路。
数据全部为教学合成，不代表任何真实保单系统。

独立启动：python -m advanced.mock_policy_api --port 8010
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

LOGGER = logging.getLogger('claims.advanced.policy_api')

# 合成保单库：覆盖自动通过 / 过期 / 免责条款 / 超保额 / 高风险等教学场景。
POLICY_DB: dict[str, dict[str, Any]] = {
    'POL-2024-001': {
        'policy_id': 'POL-2024-001', 'policy_type': '机动车商业险',
        'status': '有效', 'coverage_limit': 500000.0, 'deductible': 500.0,
        'effective_date': '2024-01-01', 'expiry_date': '2026-12-31',
        'exclusion_tags': [], 'claims_90d': 0, 'driver_restriction': '无',
        'source': '模拟保单API·合成数据',
    },
    'POL-2024-002': {
        'policy_id': 'POL-2024-002', 'policy_type': '机动车商业险',
        'status': '已过期', 'coverage_limit': 300000.0, 'deductible': 800.0,
        'effective_date': '2022-01-01', 'expiry_date': '2023-12-31',
        'exclusion_tags': ['expired_policy'], 'claims_90d': 0, 'driver_restriction': '无',
        'source': '模拟保单API·合成数据',
    },
    'POL-2024-003': {
        'policy_id': 'POL-2024-003', 'policy_type': '机动车商业险',
        'status': '有效', 'coverage_limit': 400000.0, 'deductible': 1000.0,
        'effective_date': '2023-06-01', 'expiry_date': '2026-06-01',
        'exclusion_tags': ['dui_exclusion', 'license_suspended'],
        'claims_90d': 3, 'driver_restriction': '禁止代驾及无有效驾驶证人员驾驶',
        'source': '模拟保单API·合成数据',
    },
    'POL-2024-004': {
        'policy_id': 'POL-2024-004', 'policy_type': '机动车商业险(低限额)',
        'status': '有效', 'coverage_limit': 200000.0, 'deductible': 2000.0,
        'effective_date': '2024-03-01', 'expiry_date': '2026-03-01',
        'exclusion_tags': [], 'claims_90d': 0, 'driver_restriction': '无',
        'source': '模拟保单API·合成数据',
    },
    'POL-2024-005': {
        'policy_id': 'POL-2024-005', 'policy_type': '机动车商业险',
        'status': '有效', 'coverage_limit': 600000.0, 'deductible': 0.0,
        'effective_date': '2024-01-01', 'expiry_date': '2026-12-31',
        'exclusion_tags': [], 'claims_90d': 0, 'driver_restriction': '无',
        'source': '模拟保单API·合成数据',
    },
}

CLAIM_HISTORY_DB: dict[str, list[dict[str, Any]]] = {
    'POL-2024-003': [
        {'claim_id': 'CLM-HIST-0031', 'date': '2026-01-12', 'type': '单方刮擦', 'amount': 3200.0, 'status': '已赔付'},
        {'claim_id': 'CLM-HIST-0032', 'date': '2026-02-03', 'type': '双方碰撞', 'amount': 8800.0, 'status': '已赔付'},
        {'claim_id': 'CLM-HIST-0033', 'date': '2026-02-27', 'type': '玻璃破碎', 'amount': 1500.0, 'status': '审核中'},
    ],
}


class PolicyRecord(BaseModel):
    """保单查询响应契约。"""
    model_config = {'extra': 'forbid'}
    policy_id: str = Field(min_length=1, max_length=64)
    policy_type: str
    status: str
    coverage_limit: float = Field(ge=0)
    deductible: float = Field(ge=0)
    effective_date: str
    expiry_date: str
    exclusion_tags: list[str]
    claims_90d: int = Field(ge=0)
    driver_restriction: str
    source: str


def create_app(*, fail_rate: float = 0.0, latency_ms: int = 0, seed: int = 7) -> FastAPI:
    """构建模拟API应用；fail_rate/latency_ms 用于故障注入验证降级。"""
    app = FastAPI(title='模拟保单查询API', version='1.0.0',
                  description='课堂项目的模拟保单接口，仅返回合成教学数据。')
    state = {'fail_rate': fail_rate, 'latency_ms': latency_ms, 'rng': random.Random(seed)}

    def _maybe_fault() -> None:
        if state['latency_ms']:
            time.sleep(state['latency_ms'] / 1000.0)
        if state['fail_rate'] > 0 and state['rng'].random() < state['fail_rate']:
            raise HTTPException(status_code=503, detail='模拟保单系统过载，请稍后重试')

    @app.get('/health')
    def health() -> dict[str, Any]:
        return {'status': 'ok', 'fail_rate': state['fail_rate'], 'latency_ms': state['latency_ms']}

    @app.get('/api/policies/{policy_id}', response_model=PolicyRecord)
    def get_policy(policy_id: str) -> dict[str, Any]:
        _maybe_fault()
        record = POLICY_DB.get(policy_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f'保单不存在：{policy_id}')
        return record

    @app.get('/api/policies/{policy_id}/claims')
    def get_claim_history(policy_id: str, limit: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
        _maybe_fault()
        if policy_id not in POLICY_DB:
            raise HTTPException(status_code=404, detail=f'保单不存在：{policy_id}')
        records = CLAIM_HISTORY_DB.get(policy_id, [])[:limit]
        return {'policy_id': policy_id, 'records': records,
                'source': '模拟保单API·合成历史，不代表真实投保人'}

    @app.post('/admin/fault')
    def set_fault(fail_rate: float = Query(ge=0, le=1), latency_ms: int = Query(ge=0, le=5000)) -> dict[str, Any]:
        """演示/测试用故障注入开关；生产环境不得暴露此类端点。"""
        state['fail_rate'], state['latency_ms'] = fail_rate, latency_ms
        return {'fail_rate': state['fail_rate'], 'latency_ms': state['latency_ms']}

    return app


def start_background_server(app: FastAPI, host: str = '127.0.0.1', port: int = 8010) -> Any:
    """在守护线程内运行uvicorn，返回server对象供测试关闭。"""
    import threading

    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level='warning')
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name=f'mock-policy-api-{port}')
    thread.start()
    return server


def wait_until_ready(url: str, timeout: float = 5.0) -> None:
    """阻塞等待健康检查通过，超时显式报错。"""
    import httpx
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url.rstrip('/') + '/health', timeout=1.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001 - 启动期轮询需捕获所有传输错误
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f'模拟保单API未就绪：{url}，最后错误：{last_error}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument('--fail-rate', type=float, default=float(os.getenv('POLICY_API_FAIL_RATE', '0')))
    parser.add_argument('--latency-ms', type=int, default=int(os.getenv('POLICY_API_LATENCY_MS', '0')))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    app = create_app(fail_rate=args.fail_rate, latency_ms=args.latency_ms)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
