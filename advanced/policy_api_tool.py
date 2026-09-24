"""方向一：保单查询工具（模拟API客户端 + LangChain Tool）。

通过HTTP调用 advanced.mock_policy_api 提供的模拟保单系统，
带超时与有限重试；失败显式抛错，不编造保单数据。
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.tools import ToolException, tool
from pydantic import Field

from models.schemas import Contract

LOGGER = logging.getLogger('claims.advanced.policy_tool')

DEFAULT_TIMEOUT = 3.0
DEFAULT_RETRIES = 2


class PolicyApiError(RuntimeError):
    """保单API调用失败；携带HTTP状态便于上游分流。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PolicyApiClient:
    """模拟保单API的轻量客户端：超时、有限重试、错误分类。"""

    def __init__(self, base_url: str = 'http://127.0.0.1:8010',
                 *, timeout: float = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES) -> None:
        if not base_url.startswith(('http://', 'https://')):
            raise ValueError('base_url必须是HTTP(S)地址')
        if timeout <= 0 or retries < 0:
            raise ValueError('timeout必须为正数且retries不能为负')
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retries = retries
        self._client = httpx.Client(timeout=timeout)

    def _get(self, path: str) -> dict[str, Any]:
        """GET请求；5xx/超时按次数重试，4xx不重试直接失败。"""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._client.get(self.base_url + path)
                if response.status_code >= 500:
                    last_error = PolicyApiError(
                        f'保单API服务错误：HTTP {response.status_code}', response.status_code)
                elif response.status_code == 404:
                    raise PolicyApiError(f'保单不存在：{path}', 404)
                elif response.status_code != 200:
                    raise PolicyApiError(f'保单API异常：HTTP {response.status_code}', response.status_code)
                else:
                    return response.json()
            except PolicyApiError as exc:
                if exc.status_code == 404 or attempt == self.retries:
                    raise
                last_error = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = PolicyApiError(f'保单API网络错误：{type(exc).__name__}')
                if attempt == self.retries:
                    raise last_error from exc
        raise last_error or PolicyApiError('保单API未知错误')

    def get_policy(self, policy_id: str) -> dict[str, Any]:
        if not policy_id.strip():
            raise ValueError('policy_id不能为空')
        return self._get(f'/api/policies/{policy_id.strip()}')

    def get_claim_history(self, policy_id: str, limit: int = 10) -> dict[str, Any]:
        return self._get(f'/api/policies/{policy_id.strip()}/claims?limit={limit}')

    def close(self) -> None:
        self._client.close()


class PolicyApiInput(Contract):
    """工具入参契约。"""
    policy_id: str = Field(min_length=1, max_length=64, description='待查询的保单号')


class PolicyHistoryInput(Contract):
    policy_id: str = Field(min_length=1, max_length=64, description='保单号')
    limit: int = Field(default=10, ge=1, le=100, description='最多返回记录条数')


def build_policy_api_tools(client: PolicyApiClient) -> list[Any]:
    """构造可绑定到tool-calling Agent的保单工具集。"""

    @tool(args_schema=PolicyApiInput)
    def query_policy_api(policy_id: str) -> dict:
        """查询保单实时状态、保额、免赔额与免责标记；失败时须转人工核实，不能假设保单有效。"""
        try:
            return client.get_policy(policy_id)
        except Exception as exc:
            raise ToolException(f'保单查询失败：{type(exc).__name__}: {exc}，需人工核实') from exc

    @tool(args_schema=PolicyHistoryInput)
    def query_policy_claim_history(policy_id: str, limit: int = 10) -> dict:
        """查询保单近端理赔历史；查询失败不等于没有历史记录。"""
        try:
            return client.get_claim_history(policy_id, limit)
        except Exception as exc:
            raise ToolException(f'历史查询失败：{type(exc).__name__}: {exc}') from exc

    return [query_policy_api, query_policy_claim_history]


__all__ = ['PolicyApiClient', 'PolicyApiError', 'build_policy_api_tools',
           'PolicyApiInput', 'PolicyHistoryInput']
