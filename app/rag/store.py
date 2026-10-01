"""A tiny, self-contained vector store.

Design note: the plan reached for ChromaDB + sentence-transformers. That pulls
in a ~2GB torch stack, makes the lab painful to install offline, and makes the
embedding non-deterministic across machines - none of which a teaching lab
wants. So this is a deliberately small, fully-readable embedder: character
n-gram term frequencies, cosine similarity. It reproduces the properties the
LLM09 attacks exploit - a shared geometric space, similarity scores that leak,
retrieval that can be poisoned or jammed - without hiding them behind a library.

Real deployments do not do it this way. That is fine: the vulnerability being
taught lives in *how similarity search is used*, not in which library computes
it, and every LLM09 mitigation here maps straight onto a production control.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


def embed(text: str, n: int = 3) -> Counter[str]:
    """Character n-gram term-frequency vector (sparse, as a Counter)."""
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if len(cleaned) < n:
        cleaned = cleaned.ljust(n)
    grams = [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]
    return Counter(grams)


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[g] * b[g] for g in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Record:
    id: str
    text: str
    tenant: str
    metadata: dict = field(default_factory=dict)
    vector: Counter[str] = field(default_factory=Counter)


@dataclass
class Hit:
    record: Record
    score: float


class VectorStore:
    """An in-memory similarity index.

    Whether a query is scoped to a tenant *before* or *after* the search is the
    LLM09 lesson, so the store exposes both: `search` runs across everything and
    lets the caller filter afterward (the vulnerable pattern), while
    `search_scoped` filters inside the search (the fix).
    """

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def add(self, id: str, text: str, tenant: str, metadata: dict | None = None) -> None:
        self._records[id] = Record(
            id=id, text=text, tenant=tenant, metadata=metadata or {}, vector=embed(text)
        )

    def delete(self, id: str) -> None:
        self._records.pop(id, None)

    def clear_source(self, source: str) -> None:
        for rid in [r.id for r in self._records.values() if r.metadata.get("source") == source]:
            del self._records[rid]

    def __len__(self) -> int:
        return len(self._records)

    def _rank(self, query: str, pool: list[Record], k: int) -> list[Hit]:
        qv = embed(query)
        scored = [Hit(record=r, score=cosine(qv, r.vector)) for r in pool]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Rank across EVERY record, regardless of tenant.

        This is the vulnerable primitive: similarity runs over the whole index
        and any access decision happens later, in the application.
        """
        return self._rank(query, list(self._records.values()), k)

    def search_scoped(self, query: str, tenant: str, k: int = 3) -> list[Hit]:
        """Rank only within one tenant - authorization inside the query."""
        pool = [r for r in self._records.values() if r.tenant == tenant]
        return self._rank(query, pool, k)
