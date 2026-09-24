"""方向二：轻量条款RAG检索（模块10能力，离线可用）。

课堂版 tools/rag_retrieval.py 依赖 Milvus，联调需要基础设施。
本模块提供进程内混合检索（字符bigram TF-IDF向量 + BM25，RRF融合），
复用同一 HybridRetriever 抽象：生产环境把 vector_search 换成
ClaimVectorStore 即可，检索融合逻辑零改动。

条款数据为教学合成文本，见 data/clauses/motor_commercial_v3.md。
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from tools.rag_retrieval import HybridRetriever

CLAUSE_FILE = Path(__file__).resolve().parent.parent / 'data' / 'clauses' / 'motor_commercial_v3.md'
ARTICLE_PATTERN = re.compile(r'^##\s*(第[一二三四五六七八九十百]+条)\|([a-z_]+)\|(.+)$', re.M)


class InMemoryVectorIndex:
    """字符bigram TF-IDF + 余弦相似度；确定性、无网络依赖。"""

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.doc_ids = [str(doc.metadata['doc_id']) for doc in documents]
        self._doc_vectors, self._idf = self._build(documents)

    @staticmethod
    def _terms(text: str) -> list[str]:
        cleaned = re.sub(r'\s+', '', text.lower())
        return [cleaned[i:i + 2] for i in range(len(cleaned) - 1)] or [cleaned]

    def _build(self, documents: list[Document]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        doc_freq: Counter[str] = Counter()
        term_docs: list[Counter[str]] = []
        for doc in documents:
            terms = Counter(self._terms(doc.page_content))
            term_docs.append(terms)
            doc_freq.update(terms.keys())
        total = max(len(documents), 1)
        idf = {term: math.log((total + 1) / (count + 1)) + 1.0 for term, count in doc_freq.items()}
        vectors = {}
        for doc_id, terms in zip(self.doc_ids, term_docs):
            vector = {term: (count / sum(terms.values())) * idf.get(term, 0.0) for term, count in terms.items()}
            norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
            vectors[doc_id] = {term: weight / norm for term, weight in vector.items()}
        return vectors, idf

    def search(self, query: str, k: int = 5) -> list[Document]:
        terms = Counter(self._terms(query))
        if not terms or not self.documents:
            return []
        total = sum(terms.values())
        query_vector = {term: (count / total) * self._idf.get(term, 0.0) for term, count in terms.items()}
        norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0
        query_vector = {term: weight / norm for term, weight in query_vector.items()}

        def cosine(doc_id: str) -> float:
            vector = self._doc_vectors[doc_id]
            small, large = (query_vector, vector) if len(query_vector) < len(vector) else (vector, query_vector)
            return sum(weight * large.get(term, 0.0) for term, weight in small.items())

        ranked = sorted((doc_id for doc_id in self.doc_ids), key=cosine, reverse=True)[:k]
        by_id = dict(zip(self.doc_ids, self.documents))
        return [by_id[doc_id] for doc_id in ranked]


class ClauseStore:
    """条款库：加载、按类检索、格式化为提示上下文。

    metadata 包含 doc_id（内容SHA256，可追溯）、article、kind（coverage/
    exclusion/claims/dispute）、insurance_type、version、effective_date。
    """

    INSURANCE_TYPE = 'motor_commercial'
    VERSION = 'v3.0-teaching'
    EFFECTIVE_DATE = '2024-01-01'

    def __init__(self, clause_file: Path = CLAUSE_FILE) -> None:
        text = clause_file.read_text(encoding='utf-8')
        matches = list(ARTICLE_PATTERN.finditer(text))
        if not matches:
            raise ValueError(f'条款文件无有效条文：{clause_file}')
        documents: list[Document] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            article, kind, title = match.group(1), match.group(2), match.group(3).strip()
            body = text[start:end].strip()
            content = f'{article} {title}\n{body}'
            doc_id = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
            documents.append(Document(page_content=content, metadata={
                'doc_id': doc_id, 'article': article, 'kind': kind, 'title': title,
                'insurance_type': self.INSURANCE_TYPE, 'version': self.VERSION,
                'effective_date': self.EFFECTIVE_DATE,
                'source': str(clause_file),
            }))
        self.documents = documents
        self.index = InMemoryVectorIndex(documents)
        # 向量0.5/BM25 0.5：教学条款库规模小，关键词信号与语义信号同等重要。
        self.retriever = HybridRetriever(self.index.search, documents, weights=(.5, .5))

    def retrieve(self, query: str, k: int = 4, *, kind: str | None = None) -> list[Document]:
        """混合检索；kind过滤在融合后进行，保证排序稳定。"""
        if not query.strip():
            raise ValueError('检索查询不能为空')
        results = self.retriever.retrieve(query, k=k * 3 if kind else k,
                                          insurance_type=self.INSURANCE_TYPE, version=self.VERSION)
        if kind:
            results = [doc for doc in results if doc.metadata['kind'] == kind][:k]
        return results

    def exclusion_docs(self) -> list[Document]:
        return [doc for doc in self.documents if doc.metadata['kind'] == 'exclusion']

    def find(self, doc_id: str) -> Document | None:
        return next((doc for doc in self.documents if doc.metadata['doc_id'] == doc_id), None)

    @staticmethod
    def format_for_prompt(documents: list[Document]) -> str:
        """压缩为带doc_id的引用上下文；生成端只能引用这里的doc_id。"""
        if not documents:
            return '（未检索到相关条款）'
        return '\n'.join(
            f"[{doc.metadata['doc_id']}] {doc.metadata['article']}《{doc.metadata['title']}》：{doc.page_content}"
            for doc in documents)

    @staticmethod
    def citations(documents: list[Document]) -> list[dict[str, Any]]:
        return [{'doc_id': doc.metadata['doc_id'], 'article': doc.metadata['article'],
                 'kind': doc.metadata['kind'], 'title': doc.metadata['title'],
                 'excerpt': doc.page_content[:120]} for doc in documents]


def build_clause_query(claim: dict[str, Any], policy: dict[str, Any] | None) -> str:
    """由案件事实构造检索查询：描述 + 保单状态信号，不虚构关键词。"""
    parts = [str(claim.get('description') or '')]
    if policy:
        if policy.get('status') and policy['status'] != '有效':
            parts.append('保险期间届满后 效力终止 不属于保险责任')
        for tag in policy.get('exclusion_tags') or []:
            parts.append({'dui_exclusion': '饮酒驾驶 免责', 'license_suspended': '驾驶证被暂扣吊销 无证驾驶免责',
                          'expired_policy': '保险期间届满 效力终止'}.get(tag, tag))
        if (policy.get('claims_90d') or 0) >= 3:
            parts.append('多次报案 重复索赔 欺诈核查')
    return ' '.join(part for part in parts if part.strip())


__all__ = ['ClauseStore', 'InMemoryVectorIndex', 'build_clause_query', 'CLAUSE_FILE']
