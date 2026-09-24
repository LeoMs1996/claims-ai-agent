"""方向三：结构化输出校验失败自动重试（模块06）。

课堂版 models/parsers.SafePydanticOutputParser 依赖调用方注入修复函数；
本模块实现完整闭环：解析失败 → 把校验错误回传给模型 → 重新生成 → 再校验，
有限次数后仍失败则显式抛StructuredOutputError，由上层降级（规则兜底/转人工），
绝不把半结构化输出当成合法判责结果。

网络/认证类错误不属于格式问题，直接向上抛出，不消耗重试次数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ValidationError


class StructuredOutputError(RuntimeError):
    """结构化输出经反馈重试仍未通过校验。"""

    def __init__(self, message: str, attempts: int, errors: list[str]) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.errors = errors


@dataclass
class RetryAttempt:
    """一次尝试的记录：原始输出摘要 + 校验错误。"""
    index: int
    output_preview: str
    error: str


@dataclass
class StructuredResult:
    """成功结果：解析值 + 重试证据。"""
    value: BaseModel
    attempts: int = 1
    errors: list[str] = field(default_factory=list)
    retry_log: list[RetryAttempt] = field(default_factory=list)


FEEDBACK_TEMPLATE = (
    '你上一次输出未通过结构化校验。\n'
    '校验错误：{error}\n'
    '上一次输出：{output}\n'
    '请重新输出JSON对象，严格符合以下Schema，不要输出解释、Markdown代码块或多余文本：\n{schema}'
)


async def structured_call(model: Any, schema: type[BaseModel], messages: list[Any],
                          *, max_feedback_retries: int = 2) -> StructuredResult:
    """调用模型并校验；校验失败带着错误反馈重问模型，有限次数。

    参数：
        model: 已绑定json输出格式的聊天模型（ainvoke可用）。
        schema: Pydantic模型，作为唯一校验标准。
        messages: 初始消息列表（不会被修改）。
        max_feedback_retries: 反馈重试上限（总尝试次数 = 1 + 该值）。
    """
    if max_feedback_retries < 0 or max_feedback_retries > 3:
        raise ValueError('反馈重试次数必须在0到3之间')
    parser = PydanticOutputParser(pydantic_object=schema)
    schema_json = schema.model_json_schema()
    conversation = list(messages)
    errors: list[str] = []
    retry_log: list[RetryAttempt] = []

    for attempt in range(1, max_feedback_retries + 2):
        response = await model.ainvoke(conversation)
        # 网络错误在此行之前就会抛出，不会进入下面的格式重试逻辑。
        raw = str(response.content).strip()
        try:
            value = parser.parse(raw)
            return StructuredResult(value=value, attempts=attempt,
                                    errors=errors, retry_log=retry_log)
        except (OutputParserException, ValidationError) as exc:
            error_text = str(exc)[:500]
            errors.append(error_text)
            retry_log.append(RetryAttempt(attempt, raw[:200], error_text))
            if attempt > max_feedback_retries:
                break
            # 只有格式失败才追问；对话按轮次追加，保留完整纠错上下文。
            conversation = conversation + [
                AIMessage(content=raw),
                HumanMessage(content=FEEDBACK_TEMPLATE.format(
                    error=error_text, output=raw[:1000], schema=schema_json)),
            ]
    raise StructuredOutputError(
        f'结构化输出经{len(errors)}次尝试仍未通过{schema.__name__}校验：{errors[-1] if errors else "未知错误"}',
        attempts=len(errors), errors=errors)


def retry_events(result: StructuredResult, *, node: str) -> list[dict[str, Any]]:
    """把重试证据转成图状态事件，供追踪与审计。"""
    if result.attempts <= 1 and not result.errors:
        return []
    return [{'node': node, 'schema': type(result.value).__name__,
             'attempts': result.attempts, 'errors': [e[:200] for e in result.errors]}]


__all__ = ['structured_call', 'StructuredOutputError', 'StructuredResult',
           'RetryAttempt', 'retry_events', 'FEEDBACK_TEMPLATE']
