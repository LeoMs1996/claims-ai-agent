"""方向三：LangSmith链路追踪（模块03）+ 离线可用的本地追踪。

背景：课堂版在config.py里显式关闭了LangChain自动上报（当时选型LangFuse）。
本模块提供显式opt-in开关：
- 配置了LANGSMITH_API_KEY时，setup_tracing()重新打开tracing并指定项目名，
  LangGraph每个节点的输入输出、模型调用、工具调用自动上报LangSmith；
- 未配置时启用LocalTraceCollector，同样按运行树记录节点/LLM/工具事件，
  演示与测试无需任何外部服务即可看到完整链路追踪。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

LOGGER = logging.getLogger('claims.advanced.tracing')


def setup_tracing(project: str = 'claims-advanced-judge') -> str:
    """初始化追踪，返回实际模式：langsmith / local / off。

    必须在config.py导入之后调用：config在导入时会关闭LANGSMITH环境开关，
    这里按用户显式配置重新打开，避免默认外泄请求数据。
    """
    api_key = os.getenv('LANGSMITH_API_KEY', '').strip()
    if api_key:
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGSMITH_TRACING'] = 'true'
        os.environ['LANGCHAIN_PROJECT'] = project
        os.environ['LANGSMITH_PROJECT'] = project
        LOGGER.info('LangSmith追踪已开启，项目：%s', project)
        return 'langsmith'
    if os.getenv('ADVANCED_TRACING', 'local').lower() != 'off':
        LOGGER.info('未配置LANGSMITH_API_KEY，使用本地链路追踪（ADVANCED_TRACING=off可关闭）')
        return 'local'
    return 'off'


def langsmith_enabled() -> bool:
    return bool(os.getenv('LANGSMITH_API_KEY', '').strip())


class LocalTraceCollector(BaseCallbackHandler):
    """进程内运行树采集：链/LLM/工具的输入输出、耗时与错误。

    用法：graph.ainvoke(state, config={'callbacks': [collector]})，
    结束后调用 report() 得到结构化轨迹与汇总。
    """

    def __init__(self, *, preview_chars: int = 160) -> None:
        self.preview_chars = preview_chars
        self.records: list[dict[str, Any]] = []
        self._starts: dict[str, float] = {}

    # -- 辅助 ----------------------------------------------------------
    def _preview(self, payload: Any) -> str:  # noqa: ANN401
        text = repr(payload)
        return text[:self.preview_chars] + ('…' if len(text) > self.preview_chars else '')

    def _close(self, run_id: str, name: str, run_type: str, output: Any = None,
               error: str | None = None) -> None:
        elapsed = (time.monotonic() - self._starts.pop(run_id, time.monotonic())) * 1000
        self.records.append({'type': run_type, 'name': name, 'duration_ms': round(elapsed, 1),
                             'output': self._preview(output), 'error': error})

    # -- 链（含LangGraph节点） -----------------------------------------
    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: dict[str, Any],
                       *, run_id: Any, **kwargs: Any) -> None:
        serialized = serialized or {}
        name = str(serialized.get('name') or kwargs.get('name') or 'chain')
        self._starts[str(run_id)] = time.monotonic()
        self.records.append({'type': 'chain_start', 'name': name,
                             'input': self._preview(inputs)})

    def on_chain_end(self, outputs: dict[str, Any], *, run_id: Any, **kwargs: Any) -> None:
        self._close(str(run_id), '', 'chain', outputs)

    def on_chain_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._close(str(run_id), '', 'chain', error=type(error).__name__)

    # -- LLM ------------------------------------------------------------
    def on_llm_start(self, serialized: dict[str, Any] | None, prompts: list[str],
                     *, run_id: Any, **kwargs: Any) -> None:
        serialized = serialized or {}
        self._starts[str(run_id)] = time.monotonic()
        invocation = kwargs.get('invocation_params') or {}
        self.records.append({'type': 'llm_start', 'name': str(invocation.get('model') or serialized.get('name', 'llm')),
                             'input': self._preview(prompts[-1] if prompts else '')})

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        text = ''
        try:
            text = response.generations[0][0].text
        except (AttributeError, IndexError, TypeError):
            pass
        self._close(str(run_id), '', 'llm', text)

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._close(str(run_id), '', 'llm', error=type(error).__name__)

    # -- 工具 ------------------------------------------------------------
    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str,
                      *, run_id: Any, **kwargs: Any) -> None:
        serialized = serialized or {}
        self._starts[str(run_id)] = time.monotonic()
        self.records.append({'type': 'tool_start', 'name': str(serialized.get('name') or kwargs.get('name', 'tool')),
                             'input': self._preview(input_str)})

    def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self._close(str(run_id), '', 'tool', output)

    def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        self._close(str(run_id), '', 'tool', error=type(error).__name__)

    # -- 汇总 ------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        finished = [r for r in self.records if r['type'] in ('chain', 'llm', 'tool')]
        errors = [r for r in finished if r.get('error')]
        return {'events': len(self.records), 'runs': len(finished),
                'errors': len(errors),
                'llm_calls': sum(1 for r in finished if r['type'] == 'llm'),
                'tool_calls': sum(1 for r in finished if r['type'] == 'tool'),
                'total_ms': round(sum(r['duration_ms'] for r in finished), 1)}

    def report(self) -> str:
        """人类可读的轨迹表，演示与本地排障直接打印。"""
        lines = [f"本地链路追踪：{self.summary()}"]
        for record in self.records:
            if record['type'] == 'chain_start':
                lines.append(f"[start] {record['name']} 输入={record.get('input', '')}")
            elif record['type'] in ('chain', 'llm', 'tool'):
                status = f"错误={record['error']}" if record.get('error') else f"输出={record.get('output', '')}"
                lines.append(f"[end {record['type']}] {record['duration_ms']}ms {status}")
        return '\n'.join(lines)


__all__ = ['setup_tracing', 'langsmith_enabled', 'LocalTraceCollector']
