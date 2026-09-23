"""Decision-level fusion and ranking, shared by serving and offline score calibration.

No queries, gold judgments or benchmark IDs belong in this module. Reranker inputs are composite
representations; returned passages remain genuine corpus chunks with their original quote offsets.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .corpus import Passage


@dataclass(frozen=True)
class RankConfig:
    citations: float = 0.25
    bge: float = 0.5
    bger: float = 0.25
    authority_cap: float = 2.0
    native_language: float = 0.25
    bge_margin: float = 2.0
    headnote_weight: float = 0.5


# Frozen development-calibrated serving configuration. Keep RankConfig's exploratory
# defaults separate so old candidate-cache replays remain explicit and reproducible.
DEPLOYED_RANK_CONFIG = RankConfig(
    citations=0.5, bge=1.0, bger=0.0, authority_cap=5.0,
    native_language=0.25, bge_margin=2.0, headnote_weight=1.0,
)


def fuse_decisions(pools: dict[str, list[Passage]], identity: Callable[[str], str],
                   limit: int = 120, floor: int = 40) -> list[tuple[Passage, list[Passage]]]:
    """Document-level RRF, with a small floor from each retriever to avoid candidate starvation.

    A judgment gets one contribution per retriever, however many of its paragraphs were returned.
    The BGE search is a restricted view of dense retrieval, so it receives half an RRF vote.
    """
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, dict[str, Passage]] = defaultdict(dict)
    sources: dict[str, list[str]] = defaultdict(list)
    orders = []
    for name, hits in pools.items():
        order = []
        seen = set()
        for hit in hits:
            did = identity(hit.decision_id)
            evidence[did].setdefault(hit.chunk_id, hit)
            if did in seen:
                continue
            seen.add(did)
            order.append(did)
            scores[did] += (0.5 if name.startswith('bge') else 1.0) / (60 + len(order))
            sources[did].append(name)
        orders.append(order)
    fused = sorted(scores, key=lambda did: (-scores[did], did))
    reserved = {did for order in orders for did in order[:floor]}
    selected = sorted(reserved, key=lambda did: (-scores[did], did))[:limit]
    selected += [did for did in fused if did not in reserved][:max(0, limit - len(selected))]
    out = []
    for did in sorted(selected, key=lambda did: (-scores[did], did)):
        chunks = list(evidence[did].values())
        # The first dense/lexical match is real evidence, not the synthetic reranker document.
        representative = replace(chunks[0], score=scores[did], fusion_score=scores[did],
                                 retrieval_sources=tuple(sources[did]))
        out.append((representative, chunks))
    return out


def decision_text(summary, passages: list[Passage], max_chars: int = 6500) -> str:
    """Headnote first, followed by up to two distinct matching passages, in a bounded input."""
    heading = '\n'.join(x for x in (summary.docket, summary.court_label, summary.title,
                                   summary.legal_area) if x)[:500]
    headnote = (summary.regeste or '').strip()[:2800]
    pieces = [heading]
    if headnote:
        pieces.append('Headnote / Regeste:\n' + headnote)
    seen = {' '.join(headnote.split())} if headnote else set()
    used = 0
    for p in passages:
        normal = ' '.join(p.text.split())
        if not normal or normal in seen or (headnote and normal in ' '.join(headnote.split())):
            continue
        seen.add(normal)
        pieces.append('Matching passage:\n' + p.text[:1500])
        used += 1
        if used == 2:
            break
    return '\n\n'.join(pieces)[:max_chars]


def rank_decisions(hits: list[Passage], language: str | None, config: RankConfig) -> list[Passage]:
    """Authority is bounded, and never added to RRF/cosine scores on a reranker outage."""
    out = []
    for h in hits:
        if h.reranker_score is None:
            out.append(h)
            continue
        authority = config.citations * math.log1p(max(0, h.cited_by))
        court = h.authority_court or h.court
        authority += config.bge if court == 'bge' else config.bger if court == 'bger' else 0.0
        relevance = relevance_score(h, config)
        score = relevance + min(config.authority_cap, authority)
        if language and h.language == language:
            score += config.native_language
        out.append(replace(h, score=score))
    return sorted(out, key=lambda h: (-h.score, h.decision_id, h.chunk_id))


def relevance_score(hit: Passage, config: RankConfig) -> float:
    if hit.reranker_score is None:
        return hit.fusion_score
    if hit.passage_score is None:
        return hit.reranker_score
    return config.headnote_weight * hit.reranker_score + (1 - config.headnote_weight) * hit.passage_score


def select_decisions(hits: list[Passage], k: int, language: str | None, config: RankConfig,
                     identity: Callable[[str], str]) -> list[Passage]:
    """Keep relevance anchors and only reserve a BGE near its language's best relevance score.

    This preserves strong cantonal matches; an irrelevant BGE cannot take a reserved place solely
    through popularity. Reservations control inclusion, not the order of the final ranked list.
    """
    if k <= 0:
        return []
    ranked = rank_decisions(hits, language, config)
    by_language = defaultdict(list)
    for h in ranked:
        by_language[h.language].append(h)
    reserved = set()
    for lang, group in by_language.items():
        relevance = sorted(group, key=lambda h: (-relevance_score(h, config), h.decision_id, h.chunk_id))
        reserved.update(identity(h.decision_id) for h in relevance[:2 if lang == language else 1])
        best = relevance_score(relevance[0], config)
        bge = next((h for h in relevance if (h.authority_court or h.court) == 'bge'), None)
        if bge and bge.reranker_score is not None and best - relevance_score(bge, config) <= config.bge_margin:
            reserved.add(identity(bge.decision_id))
    chosen, seen = [], set()
    for h in [h for h in ranked if identity(h.decision_id) in reserved] + ranked:
        did = identity(h.decision_id)
        if did in seen:
            continue
        seen.add(did)
        chosen.append(h)
        if len(chosen) == k:
            break
    return sorted(chosen, key=lambda h: (-h.score, h.decision_id, h.chunk_id))


# ── Learned linear ranker ──────────────────────────────────────────────────────────────────────
# One weight per standardised feature, fitted offline on the development partition only
# (agent-eval/learned_rank_experiment.py fit). The features are computed here, in serving code, and
# written to the search diagnostics, so the offline fit and replay see exactly what serving sees.

FEATURES = ('rerank', 'passage', 'fusion', 'logcit', 'bge', 'bger', 'native', 'src_dense', 'src_bm25',
            'src_bge', 'src_headnotes', 'src_statute', 'explicit_article', 'prf_articles')


@dataclass(frozen=True)
class LinearRanker:
    features: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]

    @classmethod
    def load(cls, path) -> 'LinearRanker':
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        return cls(*(tuple(data[k]) for k in ('features', 'mean', 'scale', 'weights')))

    def score(self, features: dict[str, float]) -> float:
        return sum(w * (float(features[n]) - m) / s
                   for n, m, s, w in zip(self.features, self.mean, self.scale, self.weights))


def candidate_features(hits: list[Passage], language: str | None, config: RankConfig,
                       articles: dict[str, set[tuple[str, str]]], idf: Callable[[set], dict],
                       explicit: set[tuple[str, str]], prf_depth: int = 20) -> dict[str, dict[str, float]]:
    """Features per physical decision id. `articles` maps decision ids to the statute articles they cite.

    prf_articles: pseudo-relevance feedback over statutes. The articles cited by the top judgments of
    the hand-weighted ranking form an IDF-weighted profile; a candidate scores the share of it it cites.
    """
    ranked = rank_decisions(hits, language, config)
    profile: dict[tuple[str, str], float] = defaultdict(float)
    top = ranked[:prf_depth]
    weights = idf(set().union(*(articles.get(h.decision_id, set()) for h in top))) if top else {}
    for r, h in enumerate(top):
        for art in articles.get(h.decision_id, ()):
            profile[art] += weights.get(art, 0.0) / (r + 1)
    norm = sum(profile.values()) or 1.0
    out = {}
    for h in hits:
        cites = articles.get(h.decision_id, set())
        sources = set(h.retrieval_sources)
        court = h.authority_court or h.court
        out[h.decision_id] = {
            'rerank': h.reranker_score if h.reranker_score is not None else 0.0,
            'passage': h.passage_score if h.passage_score is not None else 0.0,
            'fusion': h.fusion_score * 100, 'logcit': math.log1p(max(0, h.cited_by)),
            'bge': float(court == 'bge'), 'bger': float(court == 'bger'),
            'native': float(bool(language) and h.language == language),
            'src_dense': float('dense' in sources), 'src_bm25': float('bm25' in sources),
            'src_bge': float('bge' in sources), 'src_headnotes': float('bge_headnotes' in sources),
            'src_statute': float('statute' in sources),
            'explicit_article': float(bool(explicit & cites)),
            'prf_articles': sum(profile.get(a, 0.0) for a in cites) / norm,
        }
    return out


def learned_select(hits: list[Passage], k: int, features: dict[str, dict[str, float]], ranker: LinearRanker,
                   identity: Callable[[str], str]) -> list[Passage]:
    """Top-k judgments by the learned score; one (best-scoring) record per canonical judgment."""
    scored = sorted((replace(h, score=ranker.score(features[h.decision_id])) for h in hits),
                    key=lambda h: (-h.score, h.decision_id, h.chunk_id))
    chosen, seen = [], set()
    for h in scored:
        did = identity(h.decision_id)
        if did not in seen:
            seen.add(did)
            chosen.append(h)
            if len(chosen) == k:
                break
    return chosen
