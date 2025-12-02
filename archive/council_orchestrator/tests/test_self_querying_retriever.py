import pytest
from pathlib import Path
from council_orchestrator.orchestrator.memory.cortex import SelfQueryingRetriever, ParentDocHit

class DummyIdx:
    def search_parent_docs(self, must, should, filters, k):
        return [
            {"doc_id":"D1","path":"docs/a.md","score":0.82,"snippet":"alpha beta gamma","sha256":"x"},
            {"doc_id":"D2","path":"docs/b.md","score":0.71,"snippet":"delta epsilon","sha256":"y"},
        ][:k]

class DummyCache:
    def peek(self, key): return None
    def hit_streak(self, key): return 0

def xxh(s): return f"key::{hash(s)%10000}"

@pytest.fixture
def retriever():
    return SelfQueryingRetriever(DummyIdx(), DummyCache(), xxh)

def test_plan_query_has_terms(retriever):
    q = retriever.plan_query("Improve RAG with parent doc retrieval", "COORDINATOR")
    assert q.intent == "retrieve_parent_docs"
    assert q.k > 0
    assert len(q.must_terms) >= 1

def test_parent_doc_retrieval_returns_hits(retriever):
    q = retriever.plan_query("alpha gamma", "AUDITOR")
    r = retriever.run_parent_doc_retrieval(q)
    assert r.retrieval_latency_ms >= 0
    assert len(r.parent_docs) >= 1

def test_novelty_high_when_low_overlap(retriever):
    sig = retriever.assess_novelty("unrelated zeta kappa theta", [])
    assert sig.is_novel is True
    assert sig.signal in {"medium","high"}

def test_conflict_signal_when_cache_stable(monkeypatch, retriever):
    def stable(_): return {"stable": True}
    retriever.cache.peek = stable
    conf = retriever.detect_conflict("same prompt")
    assert conf.conflicts_with

def test_memory_directive_conflict_wins(retriever, monkeypatch):
    def stable(_): return {"stable": True}
    retriever.cache.peek = stable
    md = retriever.propose_memory_directive(
        confidence=0.99, citations=["a","b"], novelty=retriever.assess_novelty("x",[]),
        conflict=retriever.detect_conflict("y"), cache_hit_streak=10
    )
    assert md.tier == "fast"

def test_memory_directive_promotes_to_slow(retriever, monkeypatch):
    class S(DummyCache):
        def peek(self, k): return None
        def hit_streak(self, k): return 4
    retriever.cache = S()
    md = retriever.propose_memory_directive(
        confidence=0.9, citations=["dummy", "content"],  # 2 citations that match parent doc snippet
        novelty=retriever.assess_novelty("alpha beta", []),
        conflict=retriever.detect_conflict("no conflict here"),
        cache_hit_streak=4,
        parent_docs=[ParentDocHit(doc_id="d1", path="", score=0.8, snippet="dummy content here")]  # Provide evidence
    )
    assert md.tier == "slow"

def test_memory_directive_no_evidence_guardrail(retriever):
    """Test that no-evidence guardrail forces fast tier."""
    md = retriever.propose_memory_directive(
        confidence=0.99, citations=[], novelty=retriever.assess_novelty("x",[]),
        conflict=retriever.detect_conflict("y"), cache_hit_streak=10, parent_docs=[]
    )
    assert md.tier == "fast"
    assert "No-evidence guardrail" in md.justification

def test_citation_overlap_validation(retriever):
    """Test citation overlap enforcement."""
    # Valid overlap - citation tokens found in retrieved snippet
    valid = retriever._validate_citation_overlap(
        ["alpha beta gamma"], [ParentDocHit(doc_id="d1", path="", score=0.8, snippet="alpha beta gamma delta")]
    )
    assert valid is True

    # Invalid overlap (citation tokens not in retrieved docs)
    invalid = retriever._validate_citation_overlap(
        ["zeta kappa theta"], [ParentDocHit(doc_id="d1", path="", score=0.8, snippet="alpha beta gamma")]
    )
    assert invalid is False

def test_rag_deduplication(retriever):
    """Test that near-duplicate RAG hits are collapsed."""
    hits = [
        {"doc_id": "d1", "snippet": "alpha beta gamma delta"},
        {"doc_id": "d2", "snippet": "alpha beta gamma epsilon"},  # Near duplicate
        {"doc_id": "d3", "snippet": "zeta kappa theta"}  # Different
    ]
    deduplicated = retriever._deduplicate_parent_docs(hits, jaccard_threshold=0.5)
    # Should keep d1 and d3, collapse d2 as duplicate of d1
    assert len(deduplicated) == 2
    assert any(h["doc_id"] == "d1" for h in deduplicated)
    assert any(h["doc_id"] == "d3" for h in deduplicated)